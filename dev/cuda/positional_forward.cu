/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math positional_forward.cu -o positional_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./positional_forward 1

version 2 is more optimized, parallelizes over all of B,T,C
./positional_forward 2
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

// GPT-2 positional encoder forward pass
void encoder_forward_cpu(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation into kernel, parallelize over B,T, loop over C
__global__ void encoder_forward_kernel1(float* out,
                               int* inp, float* wte, float* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;

    if (idx < N) {
        int b = idx / T;
        int t = idx % T;
        float* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        float* wte_ix = wte + ix * C;
        float* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++) {
            out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
}

// optimized implementation: parallelize over all of B,T,C
__global__ void encoder_forward_kernel2(float* out,
                               int* inp, float* wte, float* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        float* out_btc = out + b * T * C + t * C + c;
        float* wte_ix = wte + ix * C + c;
        float* wpe_tc = wpe + t * C + c;
        *out_btc = *wte_ix + *wpe_tc;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_forward1(float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward2(float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void encoder_forward(int kernel_num,
                     float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C,
                     const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
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
    float* out = (float*)malloc(B * T * C * sizeof(float));
    int* inp = make_random_int(B * T, V);
    float* wte = make_random_float(V * C);
    float* wpe = make_random_float(T * C);

    // move to GPU
    float* d_out;
    int* d_inp;
    float* d_wte;
    float* d_wpe;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, V * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_wpe, T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wte, wte, V * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_wpe, wpe, T * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);
    encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, 256);
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
            encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time / repeat_times, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));

    return 0;
}