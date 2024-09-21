/*
Kernels for layernorm backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt rmsnorm_backward.cu -o rmsnorm_backward
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void rmsnorm_forward_cpu(float* out, float* rstd,
                         const float* inp, const float* weight,
                         int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the variance
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                v += x[i] * x[i];
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float o = s * x[i] * weight[i];
                out_bt[i] = o; // write
            }
            // cache the rstd for the backward pass later
            rstd[b * T + t] = s;
        }
    }
}

void rmsnorm_backward_cpu(float* dinp, float* dweight,
                          const float* dout, const float* inp, const float* weight, const float* rstd,
                          int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float rstd_bt = rstd[b * T + t];

            float o = 0.0f;
            for (int i = 0; i < C; i++) {
                o += weight[i] * dout_bt[i] * inp_bt[i];
            }

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                // gradient contribution to weight
                dweight[i] += inp_bt[i] * rstd_bt * dout_bt[i];
                // gradient contribution to input
                dinp_bt[i] = (weight[i] * C / rstd_bt / rstd_bt * dout_bt[i] - o * inp_bt[i]) * rstd_bt * rstd_bt * rstd_bt/C;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// super naive kernel that just parallelizes over B,T and loops over C
__global__ void rmsnorm_backward_kernel1(float* dinp, float* dweight,
                                         const float* dout, const float* inp, const float* weight, const float* rstd,
                                         int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float rstd_bt = rstd[b * T + t];

    float o = 0.0f;
    for (int i = 0; i < C; i++) {
        o += weight[i] * dout_bt[i] * inp_bt[i];
    }

    // now iterate again and accumulate all the gradients
    for (int i = 0; i < C; i++) {
        // gradient contribution to weight
        atomicAdd(dweight + i, inp_bt[i] * rstd_bt * dout_bt[i]);
        // gradient contribution to input
        dinp_bt[i] = (weight[i] * C / rstd_bt / rstd_bt * dout_bt[i] - o * inp_bt[i]) * rstd_bt * rstd_bt * rstd_bt/C;
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_backward1(float* dinp, float* dweight,
                        const float* dout, const float* inp, const float* weight, const float* rstd,
                        int B, int T, int C, const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    rmsnorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dout, inp, weight, rstd, B, T, C);
}

// kernel version dispatch
void rmsnorm_backward(int kernel_num,
                      floatX* dinp, floatX* dweight,
                      const floatX* dout, const floatX* inp, const floatX* weight, const floatX* rstd,
                      int B, int T, int C,
                      const int block_size) {
    switch (kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            layernorm_backward1(dinp, dweight, dout, inp, weight, rstd, B, T, C, block_size);
            break;
#endif
    default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 1600;   // this is the problematic size

    // first do the forward pass in CPU
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    rmsnorm_forward_cpu(out, rstd, inp, weight, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    rmsnorm_backward_cpu(dinp, dweight, dout, inp, weight, rstd, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    floatX* d_dinp;
    floatX* d_dweight;
    floatX* d_dout;
    floatX* d_inp;
    floatX* d_weight;
    floatX* d_rstd;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(floatX)));
    // copy over the "inputs" to the backward call
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_rstd, rstd, B * T));

    // launch the kernel
    // removed 768 because it doesn't work for kernel9 despite being OK in train_gpt2.cu?!
    int block_sizes[] = {32, 64, 128, 256, 512, /*768,*/ 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(floatX)));

        rmsnorm_backward(kernel_num, d_dinp, d_dweight, d_dout, d_inp, d_weight, d_rstd,
                         B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, rmsnorm_backward, kernel_num,
                                              d_dinp, d_dweight, d_dout, d_inp, d_weight, d_rstd,
                                              B, T, C, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(out);
    free(rstd);
    free(inp);
    free(weight);
    free(dout);
    free(dinp);
    free(dweight);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_rstd));
    return 0;
}
