/*
Kernels for gelu backwards pass.

Compile example:
nvcc -O3 --use_fast_math gelu_backwards.cu -o gelu_backwards

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_backwards 1

version 2 is using float4 data type
./gelu_backwards 2

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}


// ----------------------------------------------------------------------------
// CPU code reference
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward_cpu(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

// ----------------------------------------------------------------------------
// GPU kernels

// Initial implementation
__global__ void gelu_backward_kernel1(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}


// Optimized GELU backward kernel using float4
__global__ void gelu_backward_kernel2(float4* dinp, const float4* inp, const float4* dout, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 4 * idx; // Each thread processes 4 floats

    if (i < N) {
        float4 x = inp[idx];
        float4 dout_vec = dout[idx];
        float4 dinp_vec;

        for (int j = 0; j < 4; ++j) {
            if (i + j < N) { // Check bounds for each float component
                float xj = vec_at(x, j);
                float cube = 0.044715f * xj * xj * xj;
                float tanh_arg = GELU_SCALING_FACTOR * (xj + cube);
                float tanh_out = tanhf(tanh_arg);
                float coshf_out = coshf(tanh_arg);
                float sech_out = 1.0f / (coshf_out * coshf_out);
                float local_grad = 0.5f * (1.0f + tanh_out) + xj * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * xj * xj);
                vec_at(dinp_vec, j) = local_grad * vec_at(dout_vec, j);
            }
        }
        dinp[idx] = dinp_vec;
    }
}


// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernel1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = ceil_div(N/4, block_size);
    gelu_backward_kernel2<<<grid_size, block_size>>>((float4 *)dinp, (float4 *)inp, (float4 *)dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                  float* dinp,
                  const float* inp,
                  float* dout,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(dinp, inp, dout, B*T*C);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B*T*C);
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
    int T = 1024; //Updated value to be a better time comparison to train times
    int C = 768*4;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* dinp = make_random_float(B * T * C);
    float* inp = make_random_float(B * T * C);

    // move to GPU
    float* d_out;
    float* d_dinp;
    float* d_inp;

    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, out, B * T * C);


    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_backward(kernel_num, d_dinp, d_inp, d_out, B, T, C, block_size);
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward,
                                              kernel_num, d_dinp, d_inp, d_out,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));

    return 0;
}