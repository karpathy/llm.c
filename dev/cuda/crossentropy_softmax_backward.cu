/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_softmax_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// CUDA utils

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_softmax_backward_cpu(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;
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
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        float* probs_bt = probs + b * T * V + t * V;
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
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V,
                           const int block_size) {
    const int N = B * T * V;
    const int grid_size = CEIL_DIV(N, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dlogits, dlosses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_softmax_backward(int kernel_num,
                           float* dlogits,
                           float* dlosses, float* probs, int* targets,
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
// random utils

float* make_zeros_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float));
    return arr;
}

float* make_random_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // [0,1)
    }
    return arr;
}

int* make_random_int(int N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % V;
    }
    return arr;
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
    float* probs = make_random_float(B * T * V);
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
    cudaCheck(cudaMemcpy(d_dlogits, dlogits, B * T * V * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    // crossentropy_forward_cpu(out, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);
    crossentropy_softmax_backward(kernel_num, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, 256);
    float* dlogits_gpu = (float*)malloc(B * T * V * sizeof(float));
    cudaCheck(cudaMemcpy(dlogits_gpu, d_dlogits, B * T * V * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B * T * V; i++) {
        // print the first few comparisons
        if (i < 10) {
            printf("%f %f\n", dlogits[i], dlogits_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(dlogits[i] - dlogits_gpu[i]) > 1e-5) {
            printf("Mismatch at %d: %f vs %f\n", i, dlogits[i], dlogits_gpu[i]);
            exit(1);
        }
    }
    printf("Results match at block_size=256!\n");

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        cudaCheck(cudaEventRecord(start, 0));
        for (int i = 0; i < repeat_times; i++) {
            crossentropy_softmax_backward(kernel_num, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time / repeat_times);
    }

    // free memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);
    free(dlogits_gpu);
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_dlogits));

    return 0;
}