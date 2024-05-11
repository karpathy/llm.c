/*
Kernels for gelu backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_backward.cu -o gelu_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_backward 1

version 2 uses the Packed128 data structure
./gelu_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

__global__ void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }

        store128(dinp + i, packed_dinp);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                  floatX* dinp, 
                  const floatX* inp, 
                  const floatX* dout,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* dinp = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* dout = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, B * T * C);

    // move to GPU
    floatX* d_dinp;
    floatX* d_inp;
    floatX* d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));

    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_backward(kernel_num, d_dinp, d_inp, d_dout, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward,
                                              kernel_num, d_dinp, d_inp, d_dout,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp);
    free(inp);
    free(dout);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_dout));
    return 0;
}
