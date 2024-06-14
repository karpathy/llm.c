#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// Root Mean Square Layernorm Forward Pass
void rmsnorm_forward_cpu(
    float *out,
    float *rms,
    const float *inp,
    const float *weight,
    const float *bias,
    int B,
    int T,
    int C
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // compute RMS
            float sum_of_squares = 0.0f;
            for (int i = 0; i < C; i++) {
                sum_of_squares += x[i] * x[i];
            }
            float rms_val = rsqrtf(sum_of_squares / C);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = x[i] * rms_val; // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the rms for the backward pass later
            rms[b * T + t] = rms_val;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void rmsnorm_forward_kernel1(
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int N, 
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Seek to the input position inp[idx,:]
        const float* x = inp + idx * C;

        // Calculate the sum of squares
        float sum_of_squares = 0.0f;

        #pragma unroll
        for (int i = 0; i < C; i++) {
            sum_of_squares += x[i] * x[i];
        }

        // Compute RMS value
        sum_of_squares = sum_of_squares / C;
        float rms_val = rsqrtf(sum_of_squares);

        // Seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;

        #pragma unroll
        for (int i = 0; i < C; i++) {
            float n = x[i] * rms_val; // Normalized output
            float o = n * weight[i] + bias[i]; // Scale and shift it
            out_idx[i] = o; // Write
        }

        // Cache the RMS for the backward pass later
        rms[idx] = rms_val;
    }
}

__global__ void rms_val_kernel(
    float* rms, 
    const float* inp, 
    int N, 
    int C, 
    int block_size
) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, blocksize]
    const float *x = inp + idx * C;

    float sum_of_squares = 0.0f;

    #pragma unroll
    for (int i = tid; i < C; i += block_size) {
        sum_of_squares += x[i] * x[i];
    }
    shared[tid] = sum_of_squares;
    __syncthreads();

    #pragma unroll
    for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    if (tid == 0) {
        rms[idx] = rsqrt(shared[0] / C); // write back accumulated value in thread 0
    }
}

__global__ void rmsnorm_forward_kernel2(
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int B,
    int T, 
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bt = idx / C;
    int c = idx % C;
    
    float rms_val = rms[bt];
    float xi = inp[idx];
    float n = xi * rms_val;
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

__global__ void rmsnorm_forward_kernel3(
    float* __restrict__ out, 
    float* __restrict__ rms,
    const float* __restrict__ inp, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    int B,
    int T, 
    int C
) {
    namespace cg = cooperative_groups;
    constexpr unsigned WARP_SIZE = 32;

    int num_warps = blockDim.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x;

    __shared__ float shared[WARP_SIZE];
    const float *x = inp + idx * C;

    float thread_sum_of_squares = 0.0f;

    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum_of_squares += xi * xi;
    }

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    float warp_sum_of_squares = cg::reduce(warp, thread_sum_of_squares, cg::plus<float>{}); // sum(x * x)
    if (lane_id == 0) { 
        shared[warp_id] = warp_sum_of_squares; 
        __syncthreads();
    }

    warp_sum_of_squares = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
    float block_sum_of_squares = cg::reduce(warp, warp_sum_of_squares, cg::plus<float>{}); // sum(x * x)
    
    // compute rms
    float rms_val = rsqrtf(block_sum_of_squares / C);
    if (threadIdx.x == 0 && rms != nullptr) {
        __stcs(rms + idx, rms_val);
    }
    
    float *o = out + idx * C;
    
    #pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n =  __ldcs(x+i) * rms_val;
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void rmsnorm_forward1(
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int B, 
    int T, 
    int C,
    const int block_size
) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    rmsnorm_forward_kernel1<<<grid_size, block_size>>>(out, rms, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_forward2(
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int B, 
    int T, 
    int C,
    const int block_size
) {
    int N = B * T;
    // in rms, threads cooperate within blocks via reductions
    rms_val_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(rms, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    const int grid_size = ceil_div(B * T * C, block_size);
    rmsnorm_forward_kernel2<<<grid_size, block_size>>>(out, rms, inp, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_forward3(
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int B, 
    int T, 
    int C,
    const int block_size
) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    rmsnorm_forward_kernel3<<<grid_size, block_size>>>(out, rms, inp, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void rmsnorm_forward(
    int kernel_num,
    float* out, 
    float* rms,
    const float* inp, 
    const float* weight, 
    const float* bias,
    int B, 
    int T, 
    int C,
    const int block_size
) {
    switch (kernel_num) {
        case 1:
            rmsnorm_forward1(out, rms, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            rmsnorm_forward2(out, rms, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            rmsnorm_forward3(out, rms, inp, weight, bias, B, T, C, block_size);
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

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* rms = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_rms;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rms, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 3;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* rms_gpu = (float*)malloc(B * T * sizeof(float));

    rmsnorm_forward_cpu(out, rms, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rmsnorm_forward(kernel_num, d_out, d_rms, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_rms, rms, "rms", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(
                                repeat_times, 
                                rmsnorm_forward,
                                kernel_num, 
                                d_out, 
                                d_rms, 
                                d_inp, 
                                d_weight, 
                                d_bias,
                                B,
                                T, 
                                C, 
                                block_size
                            );

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(rms);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_rms));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}
