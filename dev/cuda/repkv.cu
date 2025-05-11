/*
Layer that takes a QKV tensor of shape (B, T, C) and replicates the K,V
some number of times. For example, if B=4, T=64, C=6144, and we have that:
- head dimension (hd) is 128 channels
- query heads: 32
- key heads: 8
- value heads: 8
- so number of heads = 32 + 8 + 8 = 48, each of 128 channels, total of 6144 channels
We want to replicate the key/value vectors 4X, so that we get:
32 + 32 + 32 = 96 query, key, value heads, each of 128 channels, total of 12288 channels
Each of these vectors should be replicated by simple copying/concat 4X times.

Compile and run as:
make repkv
./repkv

block_size 128 seems fastest on H100
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "common.h"

// cpu reference code
void repkv_forward_cpu(float* out, const float* inp,
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
            const float* x = inp + b * T * C + t * C;
            // seek to the output position out[b,t,:]
            float* y = out + b * T * Cout + t * Cout;
            // copy all the query vectors, no changes
            for (int i = 0; i < hd * qh; i++) { y[i] = x[i]; }
            x += hd * qh; // advance input pointer
            y += hd * qh; // advance output pointer
            // copy key vectors, and replicate them nrep times
            for (int h = 0; h < kh; h++) {
                for (int n = 0; n < nrep; n++) {
                    for (int i = 0; i < hd; i++) { y[i] = x[i]; }
                    y += hd; // advance output pointer
                }
                x += hd; // advance input pointer
            }
            // copy value vectors, and replicate them nrep times
            for (int h = 0; h < vh; h++) {
                for (int n = 0; n < nrep; n++) {
                    for (int i = 0; i < hd; i++) { y[i] = x[i]; }
                    y += hd; // advance output pointer
                }
                x += hd; // advance input pointer
            }
        }
    }
}

// kernels
__global__ void repkv_forward_kernel2(floatX* replicated_qkv,
                               const floatX* gqa_qkv,
                               int B, int N, int NH, int replicate_factor, int HD) {
    // we have a single tensor gqa_qkv of shape (B, N, (NH + 2*(NH/replicate_factor)) * HD)
    // we want to replicate it into (B, N, 3 * NH * HD)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // we use idx for gqa_qkv indexing
    int inp_idx = idx; // keep backup

    int NKV = NH / replicate_factor;
    int nkv_factor = (replicate_factor + 2);    // replicate_factor is for (replicate_factor * NKV == NH), 2 for K V
                                                // use NKV size instead of NH size

    if (idx >= B * N * nkv_factor * NKV * HD) { return;}

    // decode the gqa_qkv index
    int d = idx % HD;
    idx /= HD;
    int nkv = idx % NKV;
    idx /= NKV;
    int c = idx % nkv_factor;
    idx /= nkv_factor;
    int n = idx % N;
    int b = idx / N;

    int idx_flat;
    int nh_total = 3 * NH;

    if (c >= 0 && c < replicate_factor) {
        idx_flat = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD + d;
        replicated_qkv[idx_flat] = __ldcs(&gqa_qkv[inp_idx]);
    } else if (c == replicate_factor) {
        idx_flat = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD * replicate_factor + d;
        for (int i = 0; i < replicate_factor; i++) {
            replicated_qkv[idx_flat + HD * i] = __ldcs(&gqa_qkv[inp_idx]);
        }
    } else {
        c = 2 * replicate_factor;   // we need this to align for dout_idx (full KV)
        idx_flat = b * N * nh_total * HD + n * nh_total * HD + c * NKV * HD + nkv * HD * replicate_factor + d;
        for (int i = 0; i < replicate_factor; i++) {
            replicated_qkv[idx_flat + HD * i] = __ldcs(&gqa_qkv[inp_idx]);
        }
    }
}

// kernels
__global__ void repkv_forward_kernel1(floatX* replicated_qkv,
                               const floatX* gqa_qkv,
                               int B, int N, int NH, int replicate_factor, int HD) {
    // we have a single tensor gqa_qkv of shape (B, N, (NH + 2*(NH/replicate_factor)) * HD)
    // we want to replicate it into (B, N, 3 * NH * HD)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * 3 * NH * HD) { return; }
    int idx_flat = idx; // keep backup

    // decode the output index
    int d = idx % HD;
    idx /= HD;
    int nh = idx % NH;
    idx /= NH;
    int c = idx % 3;
    idx /= 3;
    int n = idx % N;
    int b = idx / N;

    int inp_idx;
    int nh_total = NH + 2 * (NH / replicate_factor);
    if (c == 0) {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + 0 * NH * HD + nh * HD + d;
    } else if (c == 1) {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + 1 * NH * HD + (nh / replicate_factor) * HD + d;
    } else {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + (NH * HD + (NH / replicate_factor) * HD) + (nh / replicate_factor) * HD + d;
    }

    replicated_qkv[idx_flat] = __ldcs(&gqa_qkv[inp_idx]);
}

// kernel launchers
void repkv_forward2(floatX* out, const floatX* inp, int B, int T, int NH, int NH_KV, int d, int block_size) {
    int total_threads = B * T * (NH + 2 * NH_KV) * d;
    int num_blocks = ceil_div(total_threads, block_size);
    int replicate_factor = NH / NH_KV;
    repkv_forward_kernel2<<<num_blocks, block_size>>>(out, inp, B, T, NH, replicate_factor, d);
    cudaCheck(cudaGetLastError());
}

// kernel launchers
void repkv_forward1(floatX* out, const floatX* inp, int B, int T, int NH, int NH_KV, int d, int block_size) {
    int total_threads = B * T * (3 * NH) * d;
    int num_blocks = ceil_div(total_threads, block_size);
    int replicate_factor = NH / NH_KV;
    repkv_forward_kernel1<<<num_blocks, block_size>>>(out, inp, B, T, NH, replicate_factor, d);
    cudaCheck(cudaGetLastError());
}

// kernel dispatcher
void repkv_forward(int kernel_num,
                   floatX* out, const floatX* inp,
                   int B, int T, int NH, int NH_KV, int d,
                   int block_size) {
    switch (kernel_num) {
        case 1:
            repkv_forward1(out, inp, B, T, NH, NH_KV, d, block_size);
            break;
        case 2:
            repkv_forward2(out, inp, B, T, NH, NH_KV, d, block_size);
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

    int C = hd * (qh + kh + vh); // input channels
    int Cout = hd * (qh * 3); // output channels

    // allocate (and fill) CPU memory
    float* inp = make_random_float(B * T * C);
    float* out = (float*)malloc(B * T * Cout * sizeof(float));

    // allocate GPU memory
    float* d_inp;
    float* d_out;
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * Cout * sizeof(float)));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // CPU reference calculate
    repkv_forward_cpu(out, inp, B, T, C, hd, qh, kh, vh);

    // check the correctness of the kernel at all block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        repkv_forward(kernel_num, d_out, d_inp, B, T, qh, kh, hd, block_size);
        validate_result(d_out, out, "out", B * T * Cout, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    // now benchmark
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, repkv_forward, kernel_num,
                                            d_out, d_inp, B, T, qh, kh, hd, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(inp);
    free(out);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));
}

