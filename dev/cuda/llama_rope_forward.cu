#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

void reshape_complex(float* inp, int B, int T, int C, float* out_real, float* out_imag) {
    #pragma unroll
    for (int b = 0; b < B; b++) {
        #pragma unroll 
        for (int t = 0; t < T; t++) {
            #pragma unroll
            for (int c = 0; c < C/2; c++) {
                int idx = b * T * C + t * C + 2 * c;
                out_real[b * T * (C/2) + t * (C/2) + c] = inp[idx];
                out_imag[b * T * (C/2) + t * (C/2) + c] = inp[idx + 1];
            }
        }
    }
}

// Llama3 Rotary Positional Embedding CPU forward pass
void apply_rotary_emb_forward_cpu(
    float* xq_inp,
    float* xk_inp,
    float* freqs_cos,
    float* freqs_sin,
    float* xq_out,
    float* xk_out,
    int B,
    int T,
    int C
) {
    float* xq_real = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xq_imag = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xk_real = (float*)malloc(B * T * (C/2) * sizeof(float));
    float* xk_imag = (float*)malloc(B * T * (C/2) * sizeof(float));

    reshape_complex(xq_inp, B, T, C, xq_real, xq_imag);
    reshape_complex(xk_inp, B, T, C, xk_real, xk_imag);

    #pragma unroll
    for (int b = 0; b < B; b++) {
        #pragma unroll
        for (int t = 0; t < T; t++) {
            #pragma unroll
            for (int c = 0; c < C/2; c++) {
                int idx = b * T * (C/2) + t * (C/2) + c;
                float xq_r_val = xq_real[idx];
                float xq_i_val = xq_imag[idx];
                float xk_r_val = xk_real[idx];
                float xk_i_val = xk_imag[idx];

                float cos_val = freqs_cos[c];
                float sin_val = freqs_sin[c];

                xq_out[idx * 2] = xq_r_val * cos_val - xq_i_val * sin_val;
                xq_out[idx * 2 + 1] = xq_r_val * sin_val + xq_i_val * cos_val;

                xk_out[idx * 2] = xk_r_val * cos_val - xk_i_val * sin_val;
                xk_out[idx * 2 + 1] = xk_r_val * sin_val + xk_i_val * cos_val;
            }
        }
    }

    free(xq_real);
    free(xq_imag);
    free(xk_real);
    free(xk_imag);
}

int main() {
    int B = 2;
    int T = 3;
    int C = 4;

    float* xq_inp = (float*)malloc(B * T * C * sizeof(float));
    float* xk_inp = (float*)malloc(B * T * C * sizeof(float));
    float* freqs_cos = (float*)malloc((C/2) * sizeof(float));
    float* freqs_sin = (float*)malloc((C/2) * sizeof(float));
    float* xq_out = (float*)malloc(B * T * C * sizeof(float));
    float* xk_out = (float*)malloc(B * T * C * sizeof(float));

    for (int i = 0; i < B * T * C; i++) {
        xq_inp[i] = i + 1;
        xk_inp[i] = i + 1;
    }
    for (int i = 0; i < C/2; i++) {
        freqs_cos[i] = cos(i);
        freqs_sin[i] = sin(i);
    }

    apply_rotary_emb_forward_cpu(xq_inp, xk_inp, freqs_cos, freqs_sin, xq_out, xk_out, B, T, C);

    for (int i = 0; i < B * T * C; i++) {
        printf("xq_out at index %d = %f\n", i+1, xq_out[i]);
        printf("xk_out at index %d = %f\n", i+1, xk_out[i]);
    }

    free(xq_inp);
    free(xk_inp);
    free(freqs_cos);
    free(freqs_sin);
    free(xq_out);
    free(xk_out);

    return 0;
}
