/*
Implements the RoPE rotation for the attention mechanism.

See dev/cuda/rope.cu for correctness and performance reference
block_size 128 seems fastest on H100
*/

#include "cuda_common.h"

void precompute_freqs_cis(floatX *freqs_cis, int dim, int end, float theta, int use_scaled) {
    // helper function that (on the CPU!) precomputes the freqs_cis for the RoPE rotation
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

__global__ void rope_forward_kernel1(floatX *out, const floatX *inp, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
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
    float x_real = inp[idxi];
    float x_imag = inp[idxi + 1];
    // apply the rotation
    out[idxi] = x_real * freqs_cos - x_imag * freqs_sin;
    out[idxi + 1] = x_real * freqs_sin + x_imag * freqs_cos;
}

__global__ void rope_backward_inplace_kernel1(floatX *dinp, const floatX *dout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim) {
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
    float dout_real = (float)dout[idxi];
    float dout_imag = (float)dout[idxi + 1];
    dinp[idxi] = dout_real * freqs_cos + dout_imag * freqs_sin;
    dinp[idxi + 1] = -dout_real * freqs_sin + dout_imag * freqs_cos;
}

void rope_forward(floatX *out, const floatX *inp, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, cudaStream_t stream) {
    // the input and output to this kernel are (B, T, 3, NH, HD) where the 3 is q,k,v
    // we are going to launch exactly one thread per element of the output,
    // except divide by two because the work is in "tuples"
    // so this single kernel launch will do RoPE for both q and k, and the threads for v will be a no-op
    const int block_size = 128;
    int total_threads = B * T * 3 * n_head * head_dim / 2;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    rope_forward_kernel1<<<num_blocks, block_size, 0, stream>>>(out, inp, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaGetLastError());
}

void rope_backward_inplace(floatX *dinp, const floatX *dout, const floatX *freqs_cis, int B, int T, int n_head, int head_dim, cudaStream_t stream) {
    // backward pass of forward, mirrors the forward kernel in setup and indexing
    const int block_size = 128;
    int total_threads = B * T * 3 * n_head * head_dim / 2;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    rope_backward_inplace_kernel1<<<num_blocks, block_size, 0, stream>>>(dinp, dout, freqs_cis, B, T, n_head, head_dim);
    cudaCheck(cudaGetLastError());
}
