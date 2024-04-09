/*
Kernels for residual forward pass.

Compile example:
nvcc -O3 --use_fast_math residual_forward.cu -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
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
// CPU code reference lol

void residual_forward_cpu(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(float* out, float* inp1, float* inp2, int N, const int block_size) {
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void residual_forward(int kernel_num,
                  float* out,
                  float* inp1,
                  float* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------
// random utils

float* make_random_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
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

    // move to GPU
    float* d_out;
    float* d_inp1;
    float* d_inp2;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp1, inp1, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp2, inp2, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);
    residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, 256);
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));
    cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B * T * C; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", out[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(out[i] - out_gpu[i]) > 1e-5) {
            printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
            exit(1);
        }
    }
    printf("Results match!\n");

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        cudaCheck(cudaEventRecord(start, 0));
        for (int i = 0; i < repeat_times; i++) {
            residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time / repeat_times, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp1);
    free(inp2);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp1));
    cudaCheck(cudaFree(d_inp2));

    return 0;
}