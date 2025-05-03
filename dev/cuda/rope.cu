/*
CUDA kernels for RoPE.

Compile and run as:
make rope
./rope

The fastest block size is 128 on H100.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "common.h"

void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta, int use_scaled) {
    // same as precompute_freqs_cis_real in rope.py
    for (int i = 0; i < dim / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float freq = 1.0f / powf(theta, (float)(2 * i) / dim);
        if (use_scaled) {
            const int scale_factor = 8;
            const int low_freq_factor = 1;
            const int high_freq_factor = 4;
            const int old_context_len = 8192;  // original llama3 length
            const float low_freq_wavelen = (float)old_context_len / low_freq_factor;
            const float high_freq_wavelen = (float)old_context_len / high_freq_factor;
            float wavelen = 2.0f * M_PI / freq;
            if (wavelen < high_freq_wavelen) {
                // skip; keep freq as is
            } else if (wavelen > low_freq_wavelen) {
                // scale down by scale_factor
                freq /= scale_factor;
            } else {
                // smooth transition between scaled and unscaled
                float smooth = ((float)old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                freq = (1.0f - smooth) * freq / scale_factor + smooth * freq;
            }
        }

        // iterate over all time steps, calculate the angle, and store the cos/sin
        for (int t = 0; t < end; t++) {
            float angle = (float)t * freq;
            freqs_cis[t * dim + 2 * i] = cosf(angle);     // real part
            freqs_cis[t * dim + 2 * i + 1] = sinf(angle); // imaginary part
        }
    }
}

void apply_rotary_emb_forward(float *out, const float *inp, const float *freqs_cis, int B, int T, int n_head, int head_dim) {
    // same as apply_rotary_emb_real in rope.py
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx_bt = b * (T * 3*n_head * head_dim) + t * (3*n_head * head_dim);
            for (int h = 0; h < 3*n_head; h++) {
                // copy value head
                int idx_bth = idx_bt + h * head_dim;
                if(h >= 2*n_head) {
                    for (int d = 0; d < head_dim; d++) {
                        out[idx_bth + d] = inp[idx_bth + d];
                    }
                    continue;
                }

                // transform qk heads
                for (int d = 0; d < head_dim / 2; d++) {
                    // fetch a tuple of activations, which we imagine as a complex number
                    int idx = idx_bth + 2 * d;
                    float x_real = inp[idx];
                    float x_imag = inp[idx + 1];
                    // fetch the angle from freqs_cis
                    int freqs_idx = t * head_dim + 2 * d;
                    float freqs_cos = freqs_cis[freqs_idx];
                    float freqs_sin = freqs_cis[freqs_idx + 1];
                    // apply the rotation
                    out[idx] = x_real * freqs_cos - x_imag * freqs_sin;
                    out[idx + 1] = x_real * freqs_sin + x_imag * freqs_cos;
                }
            }
        }
    }
}

// kernel
__global__ void rope_forward_inplace_kernel1(floatX *inout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim_half = head_dim / 2;
    if (idx >= B * T * 3 * n_head * head_dim_half) return;
    // decode the qkv index early so we can early exit if it's a value index
    int qkv = (idx / (n_head * head_dim_half)) % 3;
    if (qkv == 2) return; // no-op for v
    // decode the individual indices and get the input index
    int b = idx / (T * 3 * n_head * head_dim_half);
    int t = (idx / (3 * n_head * head_dim_half)) % T;
    int h = (idx / head_dim_half) % n_head;
    int d = idx % head_dim_half;
    int idx_bt = b * (T * 3 * n_head * head_dim) + t * (3 * n_head * head_dim);
    int idx_bth = idx_bt + qkv * (n_head * head_dim) + h * head_dim;
    int idxi = idx_bth + 2 * d; // index in the input
    // fetch the freqs_cis
    int freqs_idx = t * head_dim + 2 * d;
    float freqs_cos = freqs_cis[freqs_idx];
    float freqs_sin = freqs_cis[freqs_idx + 1];
    // fetch the input
    float x_real = inout[idxi];
    float x_imag = inout[idxi + 1];
    // apply the rotation
    inout[idxi] = x_real * freqs_cos - x_imag * freqs_sin;
    inout[idxi + 1] = x_real * freqs_sin + x_imag * freqs_cos;
}

// launchers
void rope_forward_inplace1(floatX *inout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, int block_size) {
    // let's launch one thread per element of the output (but divide two!) because the work is in "tuples"
    int total_threads = B * T * 3 * n_head * head_dim / 2;
    int num_blocks = ceil_div(total_threads, block_size);
    rope_forward_inplace_kernel1<<<num_blocks, block_size>>>(inout, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaGetLastError());
}

void rope_forward_inplace(int kernel_num, floatX *inout, const floatX *freqs_cis,
                          int B, int T, int n_head, int head_dim,
                          int block_size) {
    switch (kernel_num) {
        case 1:
            rope_forward_inplace1(inout, freqs_cis, B, T, n_head, head_dim, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------
// while we're at it, let's also briefly validate our backward kernel here

void apply_rotary_emb_backward(float *dinp, const float *dout, const float *inp, const float *freqs_cis, int B, int T, int n_head, int head_dim) {
    // backward pass of the RoPE embedding
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int idx_bt = b * (T * 3*n_head * head_dim) + t * (3*n_head * head_dim);
            for (int h = 0; h < 3*n_head; h++) {
                int idx_bth = idx_bt + h * head_dim;
                // copy value head
                if(h >= 2*n_head) {
                    for (int d = 0; d < head_dim; d++) {
                        dinp[idx_bth + d] = dout[idx_bth + d];
                    }
                    continue;
                }

                for (int d = 0; d < head_dim / 2; d++) {
                    // fetch the angle from freqs_cis
                    int freqs_idx = t * head_dim + 2 * d;
                    float freqs_cos = freqs_cis[freqs_idx];
                    float freqs_sin = freqs_cis[freqs_idx + 1];
                    // and the input index we'll be updating
                    int idx = idx_bth + 2 * d;
                    // backward pass is simple because freqs_cis is just scaling by a constant
                    dinp[idx] += dout[idx] * freqs_cos + dout[idx + 1] * freqs_sin;
                    dinp[idx + 1] += -dout[idx] * freqs_sin + dout[idx + 1] * freqs_cos;
                }
            }
        }
    }
}

__global__ void rope_backward_inplace_kernel1(floatX *dinout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim_half = head_dim / 2;
    if (idx >= B * T * 3 * n_head * head_dim_half) return;
    // decode the qkv index early so we can early exit if it's a value index
    int qkv = (idx / (n_head * head_dim_half)) % 3;
    if (qkv == 2) return; // no-op for v
    // decode the individual indices and get the input index
    int b = idx / (T * 3 * n_head * head_dim_half);
    int t = (idx / (3 * n_head * head_dim_half)) % T;
    int h = (idx / head_dim_half) % n_head;
    int d = idx % head_dim_half;
    int idx_bt = b * (T * 3 * n_head * head_dim) + t * (3 * n_head * head_dim);
    int idx_bth = idx_bt + qkv * (n_head * head_dim) + h * head_dim;
    int idxi = idx_bth + 2 * d; // index in the input
    // fetch the freqs_cis
    int freqs_idx = t * head_dim + 2 * d;
    float freqs_cos = freqs_cis[freqs_idx];
    float freqs_sin = freqs_cis[freqs_idx + 1];
    // backward
    float dout_real = (float)dinout[idxi];
    float dout_imag = (float)dinout[idxi + 1];
    dinout[idxi] = dout_real * freqs_cos + dout_imag * freqs_sin;
    dinout[idxi + 1] = -dout_real * freqs_sin + dout_imag * freqs_cos;
}

void rope_backward_inplace(floatX *dinout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, cudaStream_t stream) {
    // backward pass of forward, mirrors the forward kernel in setup and indexing
    const int block_size = 128;
    int total_threads = B * T * 3 * n_head * head_dim / 2;
    int num_blocks = ceil_div(total_threads, block_size);
    rope_backward_inplace_kernel1<<<num_blocks, block_size, 0, stream>>>(dinout, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// tester
int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int n_head = 32;
    int head_dim = 128;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // do the CPU reference calculation
    float *inp = make_random_float(B * T * 3*n_head * head_dim);
    float *freqs_cis = (float *)malloc(T * head_dim * sizeof(float));
    precompute_freqs_cis(freqs_cis, head_dim, T, 10000, 1);
    float *out = (float *)malloc(B * T * 3*n_head * head_dim * sizeof(float));
    apply_rotary_emb_forward(out, inp, freqs_cis, B, T, n_head, head_dim);

    // allocate GPU memory
    float *d_inout;
    float *d_freqs_cis;
    cudaCheck(cudaMalloc(&d_inout, B * T * 3*n_head * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_cis, T * head_dim * sizeof(float)));

    // copy data to GPU
    cudaCheck(cudaMemcpy(d_freqs_cis, freqs_cis, T * head_dim * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // check the correctness of the kernel at all block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        // inplace kernel, need to restore input before every call
        cudaCheck(cudaMemcpy(d_inout, inp, B * T * 3*n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        rope_forward_inplace(kernel_num, d_inout, d_freqs_cis, B, T, n_head, head_dim, block_size);
        validate_result(d_inout, out, "out", B * T * 3 * n_head * head_dim, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    // now benchmark
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, rope_forward_inplace, kernel_num,
                                              d_inout, d_freqs_cis, B, T, n_head, head_dim, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // now also briefly validate the backward pass
    // first, the reference CPU calculation
    float *dinp = (float *)malloc(B * T * 3*n_head * head_dim * sizeof(float));
    memset(dinp, 0, B * T * 3*n_head * head_dim * sizeof(float)); // init at zero
    apply_rotary_emb_backward(dinp, out, inp, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaMemcpy(d_inout, out, B * T * 3*n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    // now the GPU calculation (note it is done in-place, as we wish it to be to save space)
    rope_backward_inplace(d_inout, d_freqs_cis, B, T, n_head, head_dim, 0);
    validate_result(d_inout, dinp, "dinp", B * T * 3*n_head * head_dim, 1e-5f);
    printf("Backward pass result matches.\n");

    // free memory
    free(inp);
    free(freqs_cis);
    free(out);
    cudaCheck(cudaFree(d_inout));
    cudaCheck(cudaFree(d_freqs_cis));
    return 0;
}
