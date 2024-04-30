/*
Kernels for residual forward pass.

Compile example:
nvcc -O3 --use_fast_math residual_forward.cu -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
version 2 packs input into 128 bit memory reads
./residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// turn on bf16 as default, done up here for now
//#define ENABLE_BF16

#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#elif defined(ENABLE_FP16)
typedef half floatX;
typedef half floatN;
#else
typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;
// ----------------------------------------------------------------------------
// CPU code reference lol

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel1(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

__global__ void residual_forward_kernel2(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx);
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k)
        {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
        }
        store128(out + idx, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(float* out, const float* inp1, const float* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size)/x128::size;
    residual_forward_kernel2<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void residual_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp1,
                  const floatX* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
#endif
#if defined(ENABLE_BF16)
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
            break;
#endif
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
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);

    // create X host memory of random numbers
    floatX* inp1X = (floatX*)malloc(B * T * C * sizeof(float));
    floatX* inp2X = (floatX*)malloc(B * T * C * sizeof(float));

    for (int i = 0; i < B * T * C; i++) {
        inp1X[i] = (floatX)inp1[i];
        inp2X[i] = (floatX)inp2[i];
    }

    // move to GPU
    floatX* d_out;
    floatX* d_inp1;
    floatX* d_inp2;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMemcpy(d_inp1, inp1X, B * T * C * sizeof(floatX), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp2, inp2X, B * T * C * sizeof(floatX), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);


    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
#endif
#if defined(ENABLE_BF16)
        validate_result(d_out, out, "out", B * T * C, 1e-2f);
#endif
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, residual_forward,
                                              kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp1);
    free(inp2);
    free(inp1X);
    free(inp2X);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp1));
    cudaCheck(cudaFree(d_inp2));

    return 0;
}
