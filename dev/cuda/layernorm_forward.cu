/*
Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math layernorm_forward.cu -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2
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

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 float* inp, float* weight, float* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        float* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

__global__ void mean_kernel(float* mean, float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

__global__ void rstd_kernel(float* rstd, float* inp, float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

__global__ void normalization_kernel(float* out, float* inp, float* mean, float* rstd,
                                     float* weight, float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = CEIL_DIV(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
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
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);
    layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, 256);
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
    printf("Results match at block_size=256!\n");

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
            layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / (elapsed_time / repeat_times) / 1e6;

        printf("block_size %4d | time %f ms | bandwidth %f GB/s\n", block_size, elapsed_time / repeat_times, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(out_gpu);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}