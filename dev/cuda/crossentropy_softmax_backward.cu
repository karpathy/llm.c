/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_softmax_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_softmax_backward_cpu(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel that just parallelizes over B,T,V
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_softmax_backward1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    const int N = B * T * V;
    const int grid_size = ceil_div(N, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dlogits, dlosses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_softmax_backward(int kernel_num,
                           float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_softmax_backward1(dlogits, dlosses, probs, targets, B, T, V, block_size);
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
    int V = 50257;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* probs = make_random_float_01(B * T * V);
    int* targets = make_random_int(B * T, V);
    float* dlosses = make_random_float(B * T);
    float* dlogits = make_zeros_float(B * T * V);

    // move to GPU
    float* d_probs;
    int* d_targets;
    float* d_dlosses;
    float* d_dlogits;
    cudaCheck(cudaMalloc(&d_probs, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_dlogits, 0, B * T * V * sizeof(float)));
        printf("Checking block size %d.\n", block_size);
        crossentropy_softmax_backward(kernel_num, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_dlogits, dlogits, "dlogits", B * T * V, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                              kernel_num, d_dlogits, d_dlosses, d_probs, d_targets,
                                              B, T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_dlogits));

    return 0;
}