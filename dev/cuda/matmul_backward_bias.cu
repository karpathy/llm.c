/*
Kernels for matmul backward pass bias only.

Compile example:
nvcc -O3 matmul_backward_bias.cu -lineinfo -o matmul_backward_bias

./matmul_backward_bias 1
./matmul_backward_bias 2
./matmul_backward_bias 3
./matmul_backward_bias 4

ncu:
sudo ncu --set full --import-source yes -o bias -f ./matmul_backward_bias 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_bias_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                sum += dout_bt[o];
            }
        }
        dbias[o] = sum;
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void matmul_backward_bias_kernel1(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = dout + o;
    // thread coarsening
    float sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] += shared[0];
    }
}

// cooperative groups solution, one warp per output channel
__global__ void matmul_backward_bias_kernel2(float* dbias, const float* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // e.g. if block_size = 128, then we have 4 warps per block, each in charge of one output channel
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block (e.g. 4), meta_group_rank is the warp index (0,1,2,3)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= OC) { return; }
    int BT = B * T; // number of elements to reduce in total, per channel
    // first, thread coarsening to sum reduce the problem size from B*T to 32
    float sum = 0.0f;
    for(int i = warp.thread_rank(); i < BT; i += warp.size()) {
        sum += dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in this warp
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    // write the result to output (global memory)
    if(warp.thread_rank() == 0) {
        dbias[idx] += sum;
    }
}

__global__ void matmul_backward_bias_kernel3(float* dbias, const float* dout, int B, int T, int OC) {
    // dout is (B, T, OC), dbias is (OC)
    // in this version of the kernel the entire block of block_size is dedicated to one output channel
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum[32]; // block_size max is 1024 = 32 * 32 warps
    int BT = B * T; // number of elements to reduce in total, per channel
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simply one block per row
    // round 1: thread coarsening to reduce the problem size from B*T to 32
    float thread_sum = 0.0f;
    for(int i = threadIdx.x; i < BT; i += blockDim.x) {
        thread_sum += dout[i * OC + idx];
    }
    // now do a warp-level reduce to get the sum across the 32 threads in each warp
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    // store the warp sum in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{}); // sum(x)
    // write the result to output (global memory)
    if(threadIdx.x == 0) {
        dbias[idx] += block_sum;
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
__global__ void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    extern __shared__ float smem[]; // of size block_size (128)
    const int warp_id = threadIdx.x / warpSize; // warp index in the block, 0,1,2,3
    const int lane_id = threadIdx.x % warpSize; // thread index in the warp, 0,1,2,...,31
    const int tl = blockIdx.x * warpSize; // pointer to the start column for this block
    const int vstep = blockDim.x / warpSize; // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward_bias1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    dim3 block_dim(block_size);
    dim3 grid_dim(OC);
    size_t shared_mem_size = block_size * sizeof(float);
    matmul_backward_bias_kernel1<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, B, T, OC);
}

void matmul_backward_bias2(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    // block_size 512 seems best
    const int grid_size = ceil_div(OC * 32, block_size);
    matmul_backward_bias_kernel2<<<grid_size, block_size>>>(dbias, dout, B, T, OC);
}

void matmul_backward_bias3(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    // block_size 256 seems best
    matmul_backward_bias_kernel3<<<OC, block_size>>>(dbias, dout, B, T, OC);
}

void matmul_backward_bias4(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC, int block_size) {
    assert(OC % 32 == 0); // OC must be divisible by 32 for this kernel
    const int grid_size = OC / 32;
    matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
}

void matmul_backward_bias(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC, int block_size) {
    switch (kernel_num) {
        case 1:
            matmul_backward_bias1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 2:
            matmul_backward_bias2(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 3:
            matmul_backward_bias3(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        case 4:
            matmul_backward_bias4(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create host memory of random numbers
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);

    // move to GPU
    float* d_dbias;
    float* d_dout;
    cudaCheck(cudaMalloc(&d_dbias, OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dbias, dbias, OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * OC * sizeof(float), cudaMemcpyHostToDevice));

    // ncu debugging / profiling, do a single call
    // int block_size_debug;
    // if (kernel_num == 1) { block_size_debug = 512;
    // } else if (kernel_num == 2) { block_size_debug = 512;
    // } else { block_size_debug = 256; }
    // printf("kernel %d, block_size %d\n", kernel_num, block_size_debug);
    // matmul_backward_bias(kernel_num, NULL, NULL, d_dbias, d_dout, NULL, NULL, NULL, B, T, C, OC, block_size_debug);
    // exit(EXIT_SUCCESS);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // calculate the CPU reference
    matmul_backward_bias_cpu(NULL, NULL, dbias, dout, NULL, NULL, B, T, C, OC);

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // memset the bias to zero
        cudaCheck(cudaMemset(d_dbias, 0, OC * sizeof(float)));
        // calculate the GPU version
        matmul_backward_bias(kernel_num, NULL, NULL, d_dbias, d_dout, NULL, NULL, NULL, B, T, C, OC, 128);
        // compare
        printf("Checking correctness...\n");
        validate_result(d_dbias, dbias, "dbias", OC, 5e-3f);
        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now benchmark the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        float *d_dinp, *d_dweight, *d_inp, *d_weight, *d_ones;
        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_backward_bias, kernel_num,
                                            d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                            B, T, C, OC, block_size);
        printf("block_size %d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(dbias);
    free(dout);
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));

    return 0;
}