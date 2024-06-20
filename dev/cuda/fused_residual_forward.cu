/*
Kernels for residual forward pass fused with layernorm

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt fused_residual_forward.cu -o fused_residual_forward

version 1 is naive port from CPU code to kernel
./fused_residual_forward 1
version 2 packs input into 128 bit memory reads
./fused_residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference lol

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
__global__ void layernorm_forward_kernel1(floatX* out, floatX* mean, floatX* rstd,
                                          const floatX* inp, const floatX* weight, const floatX* bias,
                                          int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const floatX* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += (float)x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = (float)x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[idx,:]
        floatX* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * ((float)x[i] - m)); // normalized output
            float o = n * (float)weight[i] + (float)bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// naive fusion; uncoalesced access pattern leads to terrible performance
__global__ void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;

    float m = 0.0f;
    for(int c = 0; c < C; ++c) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = (floatX)out;
    }

    m = m / C;
    float v = 0.0f;
    for (int c = 0; c < C; c++) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for (int c = 0; c < C; c++) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = (floatX)o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = (floatX)m;
    rstd[idx] = (floatX)s;
}

// handle one token per warp for coalesced access
__global__ void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                        const floatX* inp1, const floatX* inp2,
                                        const floatX* weight, const floatX* bias,
                                        int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    float eps = 1e-5f;
    float m = 0.0f;
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float out = (float)inp1[c] + (float)inp2[c];
        m += out;
        residual[c] = out;
    }

    m = warpReduceSum(m);

    m = m / C;
    float v = 0.0f;
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float xshift = (float)residual[c] - m;
        v += xshift * xshift;
    }

    v = warpReduceSum(v);
    v = v / C;

    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    for(int c = threadIdx.x; c < C; c += WarpSize) {
        float n = (s * ((float)residual[c] - m)); // normalized output
        float o = n * (float)weight[c] + (float)bias[c]; // scale and shift it
        normed[c] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// vectorized loading, single pass stats, streaming access and zigzag loop
__global__ void fused_residual_forward_kernel4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    using x128 = Packed128<floatX>;
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int c = threadIdx.x * x128::size;
    for(; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
            sum_sq += (float)out[k] * (float)out[k];
        }
        store128(residual + c, out);
    }

    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    float m = sum / C;
    float v = sum_sq / C - m * m;
    float s = rsqrtf(v + eps);

    c -= WarpSize * x128::size;
    for(; c >= 0; c -= WarpSize * x128::size) {
        const x128 res = load128cs(residual + c);
        const x128 w = load128(weight + c);
        const x128 b = load128(bias + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}

// what do you want in shared memory? EVERYTHING!
// thus, we no longer require zigzag loops and can do the numerically more stable variance estimation
// needs special attention in the kernel launcher to ensure we have enough smem.
__global__ void fused_residual_forward_kernel5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_res = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WarpSize * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WarpSize * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // adjust pointers to current token
    residual += C * idx;
    normed += C * idx;
    inp1 += C * idx;
    inp2 += C * idx;

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 in1 = load128cs(inp1 + c);
        const x128 in2 = load128cs(inp2 + c);
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            out[k] = (floatX)((float)in1[k] + (float)in2[k]);
            sum += (float)out[k];
        }
        store128cs(residual + c, out);
        s_res[c / x128::size] = out;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for(int c = threadIdx.x * x128::size; c < C; c += WarpSize * x128::size) {
        const x128 res = s_res[c / x128::size];
        const x128 w = s_weight[c / x128::size];
        const x128 b = s_bias[c / x128::size];
        x128 out;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * (float)w[k] + (float)b[k]; // scale and shift it
            out[k] = o;
        }

        store128cs(normed + c, out);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        mean[idx] = m;
        rstd[idx] = s;
    }
}


// using multiple warps per token, and keep threads persistent, so we never have to reload weights and biases
// if we had one warp per token, though, this would require us to use a huge amount of shared memory. Therefore,
// we use multiple warps per token; but generally we cannot use the entire block, because that would give too
// little work per warp to be effective (each warp processes 256 bfloat16 elements, so for C=768 more than 3 warps
// will just mean idle). Therefore, we add a z dimension, where warps with different z handle different tokens.
// all this makes the launcher logic more complicated :(
__global__ void fused_residual_forward_kernel6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                                               const floatX* inp1, const floatX* inp2,
                                               const floatX* weight, const floatX* bias,
                                               int N, int C) {
    constexpr const int WarpSize = 32;
    assert(blockDim.x == WarpSize);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    // weights and biases are  shared among all tokens
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params + C * sizeof(floatX));
    // residual output (input to layernorm) is independent for each sub-block indicates by threadIdx.z
    x128* s_res = reinterpret_cast<x128*>(params + (2 + threadIdx.z) * C * sizeof(floatX));
    // similarly, each sub-block needs its own reduction buffers
    float* s_mean = reinterpret_cast<float*>(params + (2 + blockDim.z) * C * sizeof(floatX) + threadIdx.z * 32 * sizeof(float));
    float* s_var = reinterpret_cast<float*>(params + (2 + blockDim.z) * C * sizeof(floatX) + 32 * sizeof(float) * (blockDim.z + threadIdx.z));

    int cidx = (threadIdx.x + WarpSize * threadIdx.y) * x128::size;
    int step = blockDim.y * WarpSize * x128::size;

    for(int c = cidx; c < C; c += step) {
        s_weight[c / x128::size] = load128(weight + c);
        s_bias[c / x128::size] = load128(bias + c);
    }

    // the block-level reductions will cause sync before the first time we read these
    // => no syncthreads needed here

    // loop over all tokens
    for(int tidx = blockIdx.x * blockDim.z + threadIdx.z; tidx < N; tidx += gridDim.x * blockDim.z) {
        // adjust pointers to current token
        floatX* residual_bt = residual + C * tidx;
        floatX* normed_bt = normed + C * tidx;
        const floatX* inp1_bt = inp1 + C * tidx;
        const floatX* inp2_bt = inp2 + C * tidx;

        const float eps = 1e-5f;
        float sum = 0.0f;
        for (int c = cidx; c < C; c += step) {
            const x128 in1 = load128cs(inp1_bt + c);
            const x128 in2 = load128cs(inp2_bt + c);
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                out[k] = (float) in1[k] + (float) in2[k];
                sum += (float) out[k];
            }
            store128cs(residual_bt + c, out);
            s_res[c / x128::size] = out;
        }
        sum = warpReduceSum(sum);
        if(threadIdx.x == 0) {
            s_mean[threadIdx.y] = sum;
        }
        __syncthreads();
        float m = warpReduceSum(threadIdx.x < blockDim.y ? s_mean[threadIdx.x] : 0.f) / C;
        // normally, we'd syncthread here to make sure that no warp is already at the next
        // iteration of the loop, messing with s_mean. The fact that we interleave s_mean and s_var means
        // we don't need these additional syncs.
        float v = 0.f;

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            for (int k = 0; k < x128::size; ++k) {
                v += ((float) res[k] - m) * ((float) res[k] - m);
            }
        }

        v = warpReduceSum(v);
        if(threadIdx.x == 0) {
            s_var[threadIdx.y] = v;
        }
        __syncthreads();
        v = warpReduceSum(threadIdx.x < blockDim.y ? s_var[threadIdx.x] : 0.f) / C;
        float s = rsqrtf(v + eps);

        for (int c = cidx; c < C; c += step) {
            const x128 res = s_res[c / x128::size];
            const x128 w = s_weight[c / x128::size];
            const x128 b = s_bias[c / x128::size];
            x128 out;
            for (int k = 0; k < x128::size; ++k) {
                float n = s * ((float) res[k] - m); // normalized output
                float o = n * (float) w[k] + (float) b[k]; // scale and shift it
                out[k] = o;
            }

            store128(normed_bt + c, out);
        }
        // cache the mean and rstd for the backward pass later
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            mean[tidx] = m;
            rstd[tidx] = s;
        }
    }
}



// ----------------------------------------------------------------------------
// kernel launcher

void fused_residual_forward1(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size_resid = ceil_div(N * C, block_size);
    residual_forward_kernel1<<<grid_size_resid, block_size>>>(residual, inp1, inp2, N*C);
    cudaCheck(cudaGetLastError());
    const int grid_size_ln = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size_ln, block_size>>>(normed, mean, rstd, residual, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward2(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size));
    fused_residual_forward2<<<grid_size, block_size>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward3(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_residual_forward3<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward4(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    fused_residual_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward5(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int block_y = block_size / 32;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if(status == cudaSuccess) {
        fused_residual_forward_kernel5<<<grid_size, dim3(32, block_y), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                               weight, bias, N, C);
    } else {
        fused_residual_forward_kernel4<<<grid_size, dim3(32, block_y)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                         weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

void fused_residual_forward6(floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                             const floatX* inp1, const floatX* inp2,
                             const floatX* weight, const floatX* bias,
                             int N, int C, const int block_size) {
    int warps_per_token = max(1, C / Packed128<floatX>::size / 32);
    int total_warps = block_size / 32;
    int block_z = max(1, total_warps / warps_per_token);
    int block_y = max(1, total_warps / block_z);
    size_t smem = (2 + block_z) * C * sizeof(floatX) + 64 * sizeof(float) * block_z;

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(fused_residual_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if(status == cudaSuccess) {
        const int num_blocks = max(1, cuda_threads_per_SM * cuda_num_SMs / block_size);
        fused_residual_forward_kernel6<<<num_blocks, dim3(32, block_y, block_z), smem>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                               weight, bias, N, C);
    } else {
        const int grid_size = ceil_div(N, total_warps);
        fused_residual_forward_kernel4<<<grid_size, dim3(32, total_warps)>>>(residual, normed, mean, rstd, inp1, inp2,
                                                                         weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void fused_residual_forward(int kernel_num, floatX* residual, floatX* normed, floatX* mean, floatX* rstd,
                            const floatX* inp1, const floatX* inp2,
                            const floatX* weight, const floatX* bias,
                            int N, int C, const int block_size) {
    switch (kernel_num) {
        case 1:
            fused_residual_forward1(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 2:
            fused_residual_forward2(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 3:
            fused_residual_forward3(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 4:
            fused_residual_forward4(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 5:
            fused_residual_forward5(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        case 6:
            fused_residual_forward6(residual, normed, mean, rstd, inp1, inp2, weight, bias, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create host memory of random numbers
    float* residual = (float*)malloc(B * T * C * sizeof(float));
    float* normed = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    
    // move to GPU
    floatX* d_residual;
    floatX* d_normed;
    floatX* d_inp1;
    floatX* d_inp2;
    floatX* d_mean;
    floatX* d_rstd;
    floatX* d_weight;
    floatX* d_bias;
    cudaCheck(cudaMalloc(&d_residual, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_normed, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(memcpy_convert(d_inp1, inp1, B * T * C));
    cudaCheck(memcpy_convert(d_inp2, inp2, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_bias, bias, C));

    // first check the correctness of the kernel
    residual_forward_cpu(residual, inp1, inp2, B * T * C);
    layernorm_forward_cpu(normed, mean, rstd, residual, weight, bias, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        cudaCheck(cudaMemset(d_residual, 0, B * T * C * sizeof(floatX)));
        fused_residual_forward(kernel_num, d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                               B*T, C, block_size);
        float tol = std::is_same_v<floatX, float> ? 1e-5 : 5e-2;
        validate_result(d_residual, residual, "residual", B * T * C, tol);
        validate_result(d_mean, mean, "mean", B * T, tol);
        validate_result(d_rstd, rstd, "rstd", B * T, tol);
        validate_result(d_normed, normed, "normed", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_residual_forward, kernel_num,
                                              d_residual, d_normed, d_mean, d_rstd, d_inp1, d_inp2, d_weight, d_bias,
                                              B*T, C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 reads and 2 writes, plus 2 BT writes for mean/rstd
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * (C * 4 + 2) * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        float toks_per_msec = B * T / elapsed_time / 1e3;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s | elements: %.2f ktok/ms\n",
               block_size, elapsed_time, memory_bandwidth, toks_per_msec);
    }

    // free memory
    free(residual);
    free(normed);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
    free(inp1);
    free(inp2);
    cudaCheck(cudaFree(d_residual));
    cudaCheck(cudaFree(d_normed));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(d_inp1));
    cudaCheck(cudaFree(d_inp2));

    return 0;
}
