/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_forward.cu -o crossentropy_forward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_forward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_forward_cpu(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void crossentropy_forward_kernel1(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_forward1(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V,
                            const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(losses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_forward(int kernel_num,
                          float* losses,
                          const float* probs, const int* targets,
                          int B, int T, int V,
                          const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_forward1(losses, probs, targets, B, T, V, block_size);
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
    float* out = (float*)malloc(B * T * sizeof(float));
    float* probs = make_random_float_01(B * T * V);
    int* targets = make_random_int(B * T, V);

    // move to GPU
    float* d_out;
    float* d_probs;
    int* d_targets;
    cudaCheck(cudaMalloc(&d_out, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    crossentropy_forward_cpu(out, probs, targets, B, T, V);
    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        crossentropy_forward(kernel_num, d_out, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_out, out, "out", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_forward,
                                              kernel_num, d_out, d_probs, d_targets,
                                              B, T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f ns\n", block_size, elapsed_time, elapsed_time * 1'000'000 / (B*T));
    }

    // free memory
    free(out);
    free(probs);
    free(targets);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));

    return 0;
}