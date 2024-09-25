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

See dev/cuda/repkv.cu for correctness and performance reference
block_size 128 seems fastest on H100
*/

#include "cuda_common.h"

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

void repkv_forward(floatX* out, const floatX* inp, int B, int T, int NH, int NH_KV, int HD) {
    // NH = number of query heads, NH_KV = number of key and value heads, HD = head dimension
    const int block_size = 128;
    int total_threads = B * T * (3 * NH) * HD; // one thread per output element
    int num_blocks = CEIL_DIV(total_threads, block_size);
    int replicate_factor = NH / NH_KV;
    if (replicate_factor > 1) {
        repkv_forward_kernel1<<<num_blocks, block_size>>>(out, inp, B, T, NH, replicate_factor, HD);
    } else {
        cudaMemcpy(out, inp, total_threads * sizeof(floatX), cudaMemcpyDeviceToDevice);
    }
    cudaCheck(cudaGetLastError());
}