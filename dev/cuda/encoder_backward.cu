/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_backward.cu -o encoder_backward

version 1 is naive port from CPU code to kernel
parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward 1

version 2 is another naive port
parallelizes over C, loops over B,T; much slower than version 1
./encoder_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_backward_cpu(float* dwte, float* dwpe,
                            float* dout, int* inp,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
__global__ void encoder_backward_kernel1(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

// naive implementation that parallelizes over C and loops over B,T
// but it gets rid of atomics
__global__ void encoder_backward_kernel2(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) { return; } // guard
    int BT = B * T;
    for (int i = 0; i < BT; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_backward1(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_backward_kernel1<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward2(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int grid_size = ceil_div(C, block_size);
    encoder_backward_kernel2<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void encoder_backward(int kernel_num,
                     float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(dwte, dwpe, dout, inp, B, T, C, block_size);
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
    int V = 50257;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* dout = make_random_float(B * T * C);
    int* inp = make_random_int(B * T, V);
    float* dwte = make_zeros_float(V * C);
    float* dwpe = make_zeros_float(T * C);

    // move to GPU
    float* d_dout;
    int* d_inp;
    float* d_dwte;
    float* d_dwpe;
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_dwte, V * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dwpe, T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_dwte, 0, V * C * sizeof(float)));
        cudaCheck(cudaMemset(d_dwpe, 0, T * C * sizeof(float)));
        printf("Checking block size %d.\n", block_size);
        encoder_backward(kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        validate_result(d_dwte, dwte, "dwte", V * C, 1e-5f);
        validate_result(d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_backward,
                                              kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(dout);
    free(inp);
    free(dwte);
    free(dwpe);
    cudaFree(d_dout);
    cudaFree(d_inp);
    cudaFree(d_dwte);
    cudaFree(d_dwpe);

    return 0;
}
