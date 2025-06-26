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
            int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
            for (int h = 0; h < n_head; h++) {
                int idx_bth = idx_bt + h * head_dim;
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
__global__ void rope_forward_kernel1(floatX *out, const floatX *inp, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim_half = head_dim / 2;
    if (idx >= B * T * n_head * head_dim_half) return;
    // decode the individual indices
    int b = idx / (T * n_head * head_dim_half);
    int t = (idx / (n_head * head_dim_half)) % T;
    int h = (idx / head_dim_half) % n_head;
    int d = idx % head_dim_half;
    // calculate the index in the input
    int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
    int idx_bth = idx_bt + h * head_dim;
    int idxi = idx_bth + 2 * d; // index in the input
    // fetch the input
    float x_real = inp[idxi];
    float x_imag = inp[idxi + 1];
    // fetch the freqs_cis
    int freqs_idx = t * head_dim + 2 * d;
    float freqs_cos = freqs_cis[freqs_idx];
    float freqs_sin = freqs_cis[freqs_idx + 1];
    // apply the rotation
    out[idxi] = x_real * freqs_cos - x_imag * freqs_sin;
    out[idxi + 1] = x_real * freqs_sin + x_imag * freqs_cos;
}

// launchers
void rope_forward1(floatX *out, const floatX *inp, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, int block_size) {
    // let's launch one thread per element of the output (but divide two!) because the work is in "tuples"
    int total_threads = B * T * n_head * head_dim / 2;
    int num_blocks = ceil_div(total_threads, block_size);
    rope_forward_kernel1<<<num_blocks, block_size>>>(out, inp, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaGetLastError());
}

void rope_forward(int kernel_num, floatX *out, const floatX *inp, const floatX *freqs_cis,
                  int B, int T, int n_head, int head_dim,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            rope_forward1(out, inp, freqs_cis, B, T, n_head, head_dim, block_size);
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
            int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
            for (int h = 0; h < n_head; h++) {
                int idx_bth = idx_bt + h * head_dim;
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

__global__ void rope_backward_inplace_kernel1(floatX *dinp, const floatX *dout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim_half = head_dim / 2;
    if (idx >= B * T * n_head * head_dim_half) return;
    // decode the individual indices
    int b = idx / (T * n_head * head_dim_half);
    int t = (idx / (n_head * head_dim_half)) % T;
    int h = (idx / head_dim_half) % n_head;
    int d = idx % head_dim_half;
    // calculate the index in the input
    int idx_bt = b * (T * n_head * head_dim) + t * (n_head * head_dim);
    int idx_bth = idx_bt + h * head_dim;
    int idxi = idx_bth + 2 * d; // index in the input
    // fetch the freqs_cis
    int freqs_idx = t * head_dim + 2 * d;
    float freqs_cos = freqs_cis[freqs_idx];
    float freqs_sin = freqs_cis[freqs_idx + 1];
    // apply the rotation
    float dout_real = (float)dout[idxi];
    float dout_imag = (float)dout[idxi + 1];
    dinp[idxi] = dout_real * freqs_cos + dout_imag * freqs_sin;
    dinp[idxi + 1] = -dout_real * freqs_sin + dout_imag * freqs_cos;
}

void rope_backward_inplace(floatX *dinp, const floatX *dout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, cudaStream_t stream) {
    // backward pass of forward, mirrors the forward kernel in setup and indexing
    const int block_size = 128;
    int total_threads = B * T * 3 * n_head * head_dim / 2;
    int num_blocks = ceil_div(total_threads, block_size);
    rope_backward_inplace_kernel1<<<num_blocks, block_size, 0, stream>>>(dinp, dout, freqs_cis, B, T, n_head, head_dim);
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
    float *inp = make_random_float(B * T * n_head * head_dim);
    float *freqs_cis = (float *)malloc(T * head_dim * sizeof(float));
    precompute_freqs_cis(freqs_cis, head_dim, T, 10000, 1);
    float *out = (float *)malloc(B * T * n_head * head_dim * sizeof(float));
    apply_rotary_emb_forward(out, inp, freqs_cis, B, T, n_head, head_dim);

    // allocate GPU memory
    float *d_inp;
    float *d_freqs_cis;
    float *d_out;
    cudaCheck(cudaMalloc(&d_inp, B * T * n_head * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_cis, T * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * n_head * head_dim * sizeof(float)));

    // copy data to GPU
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * n_head * head_dim * sizeof(float), cudaMemcpyHostToDevice));
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
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        rope_forward(kernel_num, d_out, d_inp, d_freqs_cis, B, T, n_head, head_dim, block_size);
        validate_result(d_out, out, "out", B * T * n_head * head_dim, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    // now benchmark
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, rope_forward, kernel_num,
                                            d_out, d_inp, d_freqs_cis, B, T, n_head, head_dim, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // now also briefly validate the backward pass
    // first, the reference CPU calculation
    float *dinp = (float *)malloc(B * T * n_head * head_dim * sizeof(float));
    memset(dinp, 0, B * T * n_head * head_dim * sizeof(float)); // init at zero
    apply_rotary_emb_backward(dinp, out, inp, freqs_cis, B, T, n_head, head_dim);
    // now the GPU calculation (note it is done in-place, as we wish it to be to save space)
    rope_backward_inplace(d_out, d_out, d_freqs_cis, B, T, n_head, head_dim, 0);
    validate_result(d_out, dinp, "dinp", B * T * n_head * head_dim, 1e-5f);
    printf("Backward pass result matches.\n");

    // free memory
    free(inp);
    free(freqs_cis);
    free(out);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_freqs_cis));
    cudaCheck(cudaFree(d_out));
    return 0;
}


