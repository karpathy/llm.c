/*
See repkv.cu for details. This is the backward pass of repkv forward.
Block size 128 seems fastest on H100
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "common.h"

// cpu reference code
void repkv_backward_cpu(float* dinp, const float* dout,
                       int B, int T, int C,
                       int hd, int qh, int kh, int vh) {
    // inp is (B, T, C)
    // out is (B, T, 3, NH, HD)
    // hd = head dimension
    // qh, kh, vh = number of query, key, value heads
    assert(C == hd * (qh + kh + vh));
    assert(kh == vh);
    int nrep = qh / kh; // number of times to replicate key/value vectors
    int Cout = hd * (qh * 3); // output channels

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* dx = dinp + b * T * C + t * C;
            // seek to the output position out[b,t,:]
            const float* dy = dout + b * T * Cout + t * Cout;
            // copy all the query vectors, no changes
            for (int i = 0; i < hd * qh; i++) { dx[i] = dy[i]; }
            dx += hd * qh; // advance input pointer
            dy += hd * qh; // advance output pointer
            // gather gradients from the key vectors
            for (int h = 0; h < kh; h++) {
                // init the gradient to 0
                for (int i = 0; i < hd; i++) { dx[i] = 0.0f; }
                for (int n = 0; n < nrep; n++) {
                    for (int i = 0; i < hd; i++) { dx[i] += dy[i]; }
                    dy += hd; // advance output pointer
                }
                dx += hd; // advance input pointer
            }
            // gather gradients from the value vectors
            for (int h = 0; h < vh; h++) {
                // init the gradient to 0
                for (int i = 0; i < hd; i++) { dx[i] = 0.0f; }
                for (int n = 0; n < nrep; n++) {
                    for (int i = 0; i < hd; i++) { dx[i] += dy[i]; }
                    dy += hd; // advance output pointer
                }
                dx += hd; // advance input pointer
            }
        }
    }
}

// kernels
__global__ void repkv_backward_kernel2(floatX* dinp, const floatX* dout,
                                int B, int N, int NH, int replicate_factor, int HD) {
    // we have a single tensor dout of shapae of (B, N 3 * NH * HD)
    // we want to reduce sum (for K and V) into  (B, N, (NH + 2*(NH/replicate_factor)) * HD)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // we use idx for dinp indexing
    int dinp_idx = idx; // keep backup

    int NKV = NH / replicate_factor;
    int nkv_factor = (replicate_factor + 2);    // replicate_factor is for (replicate_factor * NKV == NH), 2 for K V
                                                // use NKV size instead of NH size

    if (idx >= B * N * nkv_factor * NKV * HD) { return;}

    // decode the dinp index
    int d = idx % HD;
    idx /= HD;
    int nkv = idx % NKV;
    idx /= NKV;
    int c = idx % nkv_factor;
    idx /= nkv_factor;
    int n = idx % N;
    int b = idx / N;

    int dout_idx;
    int nh_total = 3 * NH;

    if (c >= 0 && c < replicate_factor) {
        dout_idx = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD + d;
        dinp[dinp_idx] = __ldcs(&dout[dout_idx]);
    } else if (c == replicate_factor) {
        float reduced_sum = 0.0f;
        dout_idx = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD * replicate_factor + d;
        for (int i = 0; i < replicate_factor; i++) {
            reduced_sum += __ldcs(&dout[dout_idx + HD * i]);
        }
        dinp[dinp_idx] = reduced_sum;
    } else {
        float reduced_sum = 0.0f;
        c = 2 * replicate_factor;   // we need this to align for dout_idx (full KV)
        dout_idx = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD * replicate_factor + d;
        for (int i = 0; i < replicate_factor; i++) {
            reduced_sum += __ldcs(&dout[dout_idx + HD * i]);
        }
        dinp[dinp_idx] = reduced_sum;
    }
}

// kernels
__global__ void repkv_backward_kernel1(floatX* dinp, const floatX* dout,
                                int B, int N, int NH, int replicate_factor, int HD) {
    // we have a single tensor dout of shapae of (B, N 3 * NH * HD)
    // we want to reduce sum (for K and V) into  (B, N, (NH + 2*(NH/replicate_factor)) * HD)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * 3 * NH * HD) { return;}
    int dout_idx = idx; // keep backup

    // decode the dout index
    int d = idx % HD;
    idx /= HD;
    int nh = idx % NH;
    idx /= NH;
    int c = idx % 3;
    idx /= 3;
    int n = idx % N;
    int b = idx / N;

    int dinp_idx;
    int nh_total = NH + 2 * (NH / replicate_factor);

    if (c == 0) {
        dinp_idx = b * N * nh_total * HD + n * nh_total * HD + 0 * NH * HD + nh * HD + d;
        dinp[dinp_idx] = __ldcs(&dout[dout_idx]);
    } else if (c == 1) {
        if (nh % replicate_factor == 0) {
            float reduced_sum = 0.0f;
            for (int i = 0; i < replicate_factor; i++) {
                reduced_sum += __ldcs(&dout[dout_idx+HD*i]);
            }

            dinp_idx = b * N * nh_total * HD + n * nh_total * HD + 1 * NH * HD + (nh / replicate_factor) * HD + d;
            dinp[dinp_idx] = reduced_sum;
        }

    } else {
        if (nh % replicate_factor == 0) {
            float reduced_sum = 0.0f;
            for (int i = 0; i < replicate_factor; i++) {
                reduced_sum += __ldcs(&dout[dout_idx+HD*i]);
            }
            dinp_idx = b * N * nh_total * HD + n * nh_total * HD + (NH * HD + (NH / replicate_factor) * HD) + (nh / replicate_factor) * HD + d;
            dinp[dinp_idx] = reduced_sum;
        }
    }
}

// kernel launchers
void repkv_backward2(floatX* dinp, const floatX* dout,
    const int B, const int T, const int NH, const int NH_KV, const int d, int block_size) {
    int total_threads = B * T * (NH + 2 * NH_KV) * d;
    int num_blocks = ceil_div(total_threads, block_size);
    int replicate_factor = NH / NH_KV;
    repkv_backward_kernel2<<<num_blocks, block_size>>>(dinp, dout, B, T, NH, replicate_factor, d);
    cudaCheck(cudaGetLastError());
}

// kernel launchers
void repkv_backward1(floatX* dinp, const floatX* dout,
    const int B, const int T, const int NH, const int NH_KV, const int d, int block_size) {
    int total_threads = B * T * (3 * NH) * d;
    int num_blocks = ceil_div(total_threads, block_size);
    int replicate_factor = NH / NH_KV;
    repkv_backward_kernel1<<<num_blocks, block_size>>>(dinp, dout, B, T, NH, replicate_factor, d);
    cudaCheck(cudaGetLastError());
}

// kernel dispatcher
void repkv_backward(int kernel_num,
                   floatX* dinp, const floatX* dout,
                   int B, int T, int NH, int NH_KV, int d,
                   int block_size) {
    switch (kernel_num) {
        case 1:
            repkv_backward1(dinp, dout, B, T, NH, NH_KV, d, block_size);
            break;
        case 2:
            repkv_backward2(dinp, dout, B, T, NH, NH_KV, d, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// tester
int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int hd = 128; // head dim
    int qh = 32; // num query heads
    int kh = 8; // num key heads
    int vh = 8; // num value heads

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    int Cout = hd * (qh * 3); // out, upstream channels
    int Cin = hd * (qh + kh + vh); // in, downstream channels

    // allocate (and fill) CPU memory
    float* dinp = (float*)malloc(B * T * Cin * sizeof(float));
    float* dout = make_random_float(B * T * Cout * sizeof(float));

    // allocate GPU memory
    float* d_dinp;
    float* d_inp;
    float* d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * Cin * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * Cin * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * Cout * sizeof(float)));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // CPU reference calculate
    repkv_backward_cpu(dinp, dout, B, T, Cin, hd, qh, kh, vh);

    // check the correctness of the kernel at all block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * Cout * sizeof(float), cudaMemcpyHostToDevice));
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        repkv_backward(kernel_num, d_dinp, d_dout, B, T, qh, kh, hd, block_size);
        validate_result(d_dinp, dinp, "out", B * T * Cin, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    // now benchmark
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, repkv_backward, kernel_num,
                                            d_dinp, d_dout, B, T, qh, kh, hd, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(dinp);
    free(dout);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_dout));

    return 0;
}

