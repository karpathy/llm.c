/*
Kernels for softmax forward pass.

Compile example:
nvcc -O3 --use_fast_math softmax_forward.cu -o softmax_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./softmax_forward 1

version 2 is a fused kernel that parallelizes over all of B,T,C
./softmax_forward 2
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

void softmax_forward_cpu(float* out, float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void softmax_forward_kernel1(float* out, float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}

__global__ void softmax_forward_kernel2(float* out, float* inp, int N, int C) {
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    extern __shared__ float shared[];
    int idx = blockIdx.x; // ranges [0, N)
    int tid = threadIdx.x; // ranges [0, block_size)
    int block_size = blockDim.x;
    float* x = inp + idx * C; // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    float offset = shared[0];
    __syncthreads();
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void softmax_forward1(float* out, float* inp, int N, int C, const int block_size) {
    const int grid_size = CEIL_DIV(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward2(float* out, float* inp, int N, int C, const int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, float* inp, int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
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

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * T * sizeof(float));
    float* inp = make_random_float(B * T * T);

    // move to GPU
    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * T * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * T * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    float* out_gpu = (float*)malloc(B * T * T * sizeof(float));

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        softmax_forward_cpu(out, inp, B * T, T);
        softmax_forward(kernel_num, d_out, d_inp, B * T, T, block_size);
        cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * T * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < B * T * T; i++) {
            // print the first few comparisons
            if (i < 5) {
                printf("%f %f\n", out[i], out_gpu[i]);
            }
            // ensure correctness for all elements
            if (fabs(out[i] - out_gpu[i]) > 1e-4) {
                printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
                exit(1);
            }
        }
        printf("Results match at block_size=%d\n", block_size);
    }

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        cudaCheck(cudaEventRecord(start, 0));
        for (int i = 0; i < repeat_times; i++) {
            softmax_forward(kernel_num, d_out, d_inp, B * T, T, block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time / repeat_times);
    }

    // free memory
    free(out);
    free(inp);
    free(out_gpu);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));

    return 0;
}