/*
Kernels for permute kernel backward pass.

Compile example:
nvcc -O3 --use_fast_math permute_backward.cu -o permute_backward

Test kernel:
./permute_backward

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code for permute backward pass

void permute_kernel_backward_cpu(float* dinp,
                                 const float* dq, const float* dk, const float* dv,
                                 int B, int N, int NH, int d) {
    for (int b = 0; b < B; ++b) {
        for (int nh_ = 0; nh_ < NH; ++nh_) {
            for (int n = 0; n < N; ++n) {
                for (int d_ = 0; d_ < d; ++d_) {
                    int idx = b * NH * N * d + nh_ * N * d + n * d + d_;
                    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
                    dinp[inp_idx] = dq[idx];
                    dinp[inp_idx + NH * d] = dk[idx];
                    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel for permute backward pass

__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

// ----------------------------------------------------------------------------
// Kernel launcher

void permute_kernel_backward_launcher(floatX* dinp, const floatX* dq, const floatX* dk, const floatX* dv,
                                      int B, int N, int NH, int d, int block_size) {
    int num_blocks = ceil_div(B * NH * N * d, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, N, NH, d);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Main function for testing

int main(int argc, char **argv) {
    setup_main();

    int B = 8; // Batch size
    int N = 1024; // Sequence length
    int NH = 12; // Number of heads
    int d = 64; // Dimension per head

    // create host memory of random numbers
    float* dinp_cpu = (float*)malloc(B * N * 3 * NH * d * sizeof(float));
    float* dinp_gpu = (float*)malloc(B * N * 3 * NH * d * sizeof(float));
    float* dq = make_random_float(B * N * NH * d);
    float* dk = make_random_float(B * N * NH * d);
    float* dv = make_random_float(B * N * NH * d);

    // run CPU version
    permute_kernel_backward_cpu(dinp_cpu, dq, dk, dv, B, N, NH, d);

    // move to GPU
    floatX* d_dinp;
    floatX* d_dq;
    floatX* d_dk;
    floatX* d_dv;
    cudaCheck(cudaMalloc(&d_dinp, B * N * 3 * NH * d * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dq, B * N * NH * d * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dk, B * N * NH * d * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dv, B * N * NH * d * sizeof(floatX)));

    cudaCheck(memcpy_convert(d_dq, dq, B * N * NH * d));
    cudaCheck(memcpy_convert(d_dk, dk, B * N * NH * d));
    cudaCheck(memcpy_convert(d_dv, dv, B * N * NH * d));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        permute_kernel_backward_launcher(d_dinp, d_dq, d_dk, d_dv, B, N, NH, d, block_size);

        cudaCheck(cudaMemcpy(dinp_gpu, d_dinp, B * N * 3 * NH * d * sizeof(floatX), cudaMemcpyDeviceToHost));

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp_cpu, "dinp", B * N * 3 * NH * d, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, permute_kernel_backward_launcher,
                                              d_dinp, d_dq, d_dk, d_dv,
                                              B, N, NH, d, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * N * NH * d * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp_cpu);
    free(dinp_gpu);
    free(dq);
    free(dk);
    free(dv);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dq));
    cudaCheck(cudaFree(d_dk));
    cudaCheck(cudaFree(d_dv));
    return 0;
}
