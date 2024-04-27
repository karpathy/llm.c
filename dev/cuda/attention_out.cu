// Output matrix multiplication if the attention mechanism. The implementation logic
// is very similar to `trimat_forward`, except this time, one of the factors (att)
// is triangular, and the result is dense.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"

static cublasHandle_t cublas_handle;
static float* d_qkvr;       // scratch for the cublas kernel
static float* d_vaccum;     // scratch for the cublas kernel


// taken from then attention forward pass

void attention_forward_cpu(float* out, float* preatt, float* att,
                           const float* inp,
                           int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// other kernels just needed for the reference cublas implementation

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

void att_out_cublas(float* out,
                    const float* att, const float* inp,
                    int B, int T, int C, int NH) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size
    constexpr const int block_size = 256;

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = d_qkvr + 0 * B * T * C;
    k = d_qkvr + 1 * B * T * C;
    v = d_qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;


    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          HS, T, T,
                                          &alpha,
                                          v, HS, T * HS,
                                          att, T, T * T,
                                          &beta,
                                          d_vaccum, HS, T * HS,
                                          B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(d_vaccum, out, B, T, NH, HS);

}

// ---------------------------------------------------------------------------------------------------------------------
//  Our kernels

template<auto matmul_tri>
__global__ void __launch_bounds__(256, 2) att_out_global(float* out, const float* att, const float* inp, int T, int C, int NH) {
    // att: (B, NH, T, T), inp: (B, T, 3C), out: (B, T, C)
    // set up indices
    int C3 = C*3;
    int hs = C / NH; // head size

    // we put the "batch x head" dimension into the z block index.
    int h = blockIdx.z % NH;
    int b = blockIdx.z / NH;

    // Get the base address for the current batch and head
    // (B, nh, T, hs)
    const float* a = att + b * NH * T * T + h * T * T;
    const float* v = inp + b * T * C3 + h * hs + 2*C;
    float* r = out + b*T*C + h * hs;
    //float* rend = out + B;

    // start the multiplication
    matmul_tri(r, C, a, T, v, C3, T);
}

template<auto matmul_tri>
void att_out_launcher(float* out, const float* att, const float* inp,
                      int B, int T, int C, int NH) {
    // we assume nice shapes here. Let's not make the code a mess by supporting weird shapes that you
    // wouldn't want to use anyway.
    assert(T % 128 == 0);
    assert(C % 128 == 0);
    // No need to ceil_div, if it's not a multiple of 128, we would get wrong results anyway.
    att_out_global<matmul_tri><<<dim3(C / NH / 64, T / 128, NH * B), dim3(16, 16)>>>(out, att, inp, T, C, NH);
    cudaCheck(cudaGetLastError());
}

// Simple version; Just get the indexing right
// Since j_base goes over attention heads, we have to chose a smaller area covered by each block in  j direction
__device__ void att_out_naive(float* o, int os, const float* a, int as, const float* v, int vs, int T) {
    // get coordinates of our block
    int i_base = 128 * blockIdx.y + 8 * threadIdx.y;
    int j_base = 64 * blockIdx.x + 4 * threadIdx.x;

    // adjust pointers to current block
    v += j_base;
    a += i_base * as;
    o += j_base + i_base * os;

    // Simple nested loop that calculates 8x4 results in one thread.
    for(int i = 0; i < 8; ++i) {
        for(int j = 0; j < 4; ++j) {
            float val = 0;
            // don't loop to the end; a is a triangular matrix
            for (int s = 0; s <= i_base + i; ++s) {
                val += a[i * as + s] * v[j + vs * s];
            }
            o[i * os + j] = val;
        }
    }
}

// Reuse data in registers.
__device__ void att_out_register(float* o, int os, const float* a, int as, const float* v, int vs, int T) {
    // get coordinates of our block
    int i_base = 128 * blockIdx.y + 8 * threadIdx.y;
    int j_base = 64 * blockIdx.x + 4 * threadIdx.x;
    // don't load the zero part of the attention matrix
    int i_max = i_base + 8;

    // adjust pointers to current block
    v += j_base;
    a += i_base * as;
    o += j_base + i_base * os;

    // calculate 8x4 results in one thread, loading all the required inputs just once
    float vals[8][4] = {};
    for (int s = 0; s < i_max ; ++s) {
        float ai[8];
        for (int i = 0; i < 8; ++i) {
            ai[i] = a[i * as + s];
        }
        for (int j = 0; j < 4; ++j) {
            float vj = v[j + vs * s];
            for (int io = 0; io < 8; ++io) {
                vals[io][j] += ai[io] * vj;
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            o[i * os + j] = vals[i][j];
        }
    }
}

__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

// Vectorized loads. Aligns very well with the stride of 4 in x-direction
__device__ void att_out_vec(float* o, int os, const float* a, int as, const float* v, int vs, int T) {
    // get coordinates of our block
    int i_base = 128 * blockIdx.y;
    int j_base = 64 * blockIdx.x;
    int i_max = i_base + 128;

    // adjust pointers to current block
    v += j_base;
    a += i_base * as;
    o += j_base + i_base * os;

    // calculate 8x4 results in one thread, loading all the required inputs just once
    float vals[8][4] = {};
    for (int so = 0; so < i_max ; so += 4) {
        float4 ai[8];
        for (int io = 0; io < 8; ++io) {
            int i = 8 * threadIdx.y + io;
            ai[io] = ld_vec(a + i * as + so);
        }
        for(int si = 0; si < 4; ++si) {
            float4 vj = ld_vec(v + 4 * threadIdx.x + vs * (so + si));
            for (int io = 0; io < 8; ++io) {
                vals[io][0] += vec_at(ai[io], si) * vj.x;
                vals[io][1] += vec_at(ai[io], si) * vj.y;
                vals[io][2] += vec_at(ai[io], si) * vj.z;
                vals[io][3] += vec_at(ai[io], si) * vj.w;
            }
        }
    }

    for (int io = 0; io < 8; ++io) {
        int i = 8 * threadIdx.y + io;
        float4 store = {vals[io][0], vals[io][1], vals[io][2], vals[io][3]};
        int j = 4 * threadIdx.x;
        st_vec(o + i * os + j, store);
    }
}

// shared memory
__device__ void att_out_shared(float* o, int os, const float* a, int as, const float* v, int vs, int T) {
    // get coordinates of our block
    int i_base = 128 * blockIdx.y;
    int j_base = 64 * blockIdx.x;
    int i_max = i_base + 128;


    __shared__ float4 v_buffer[16][16];
    __shared__ float4 a_buffer[4][16][8];

    // adjust pointers to current block
    v += j_base;
    a += i_base * as;
    o += j_base + i_base * os;

    // calculate 8x4 results in one thread, loading all the required inputs just once
    float vals[8][4] = {};
    for (int so = 0; so < i_max ; so += 16) {

        // fill buffers
        __syncthreads();
        int si = threadIdx.y;
        v_buffer[si][threadIdx.x] = ld_vec(v + vs * (so + si) + 4 * threadIdx.x);

        for(int mo = 0; mo < 2; ++mo) {
            int io = threadIdx.x % 8;
            int sm = 2*mo + threadIdx.x / 8;
            int i = 8 * threadIdx.y + io;
            a_buffer[sm][threadIdx.y][io] = ld_vec(a + i * as + so + 4 * sm);
        }
        __syncthreads();
        for(int sm = 0; sm < 4; ++sm) {
            float4 ai[8];
            for (int io = 0; io < 8; ++io) {
                ai[io] = a_buffer[sm][threadIdx.y][io];
            }
            for (int si = 0; si < 4; ++si) {
                float4 vj = v_buffer[4*sm + si][threadIdx.x];
                for (int io = 0; io < 8; ++io) {
                    vals[io][0] += vec_at(ai[io], si) * vj.x;
                    vals[io][1] += vec_at(ai[io], si) * vj.y;
                    vals[io][2] += vec_at(ai[io], si) * vj.z;
                    vals[io][3] += vec_at(ai[io], si) * vj.w;
                }
            }
        }
    }

    for (int io = 0; io < 8; ++io) {
        int i = 8 * threadIdx.y + io;
        float4 store = {vals[io][0], vals[io][1], vals[io][2], vals[io][3]};
        int j = 4 * threadIdx.x;
        st_vec(o + i * os + j, store);
    }
}

// -------------------------------------------------------------------------------------------------------------
//  dispatch

void att_out_gpu(int kernel_num, float* out, const float* att, const float* inp, int B, int T, int C, int NH) {
    switch (kernel_num) {
        case 0:
            att_out_cublas(out, att, inp, B, T, C, NH);
            break;
        case 1:
            att_out_launcher<att_out_naive>(out, att, inp, B, T, C, NH);
            break;
        case 2:
            att_out_launcher<att_out_register>(out, att, inp, B, T, C, NH);
            break;
        case 3:
            att_out_launcher<att_out_vec>(out, att, inp, B, T, C, NH);
            break;
        case 4:
            att_out_launcher<att_out_shared>(out, att, inp, B, T, C, NH);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}



int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cublasCreate(&cublas_handle);
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    cudaCheck(cudaMemcpy(d_att, att, B * NH * T * T * sizeof(float), cudaMemcpyHostToDevice));
    att_out_gpu(kernel_num, d_out, d_att, d_inp, B, T, C, NH);
    validate_result(d_out, out, "out", B * T * C, 1e-4f);

    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, att_out_gpu,
                                          kernel_num, d_out, d_att, d_inp,
                                          B, T, C, NH);

    float cublas_time = benchmark_kernel(repeat_times, att_out_gpu,
                                         0, d_out, d_att, d_inp,
                                         B, T, C, NH);

    printf("time %.2f ms vs %.2f ms for CuBLAS\n", elapsed_time, cublas_time);

    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

    return 0;
}
