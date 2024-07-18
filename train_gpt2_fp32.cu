/*
GPT-2 Transformer Neural Net trained in raw CUDA
Non-trivial notes to be aware of:

We are being clever in the backward pass to conserve memory.
In particular, all parameters use a += in the backward pass, so we
can later do gradient accumulation. But all activations have = instead of +=
because these are faster (just read, no write). This is okay for all activations
except for those in the residual stream, where the gradients have to add. We make
sure that those parts work out ok and that we do a += as necessary. E.g.,
the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx> // for TMA on H100+

// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"
// ----------------------------------------------------------------------------
// CUDA utils
constexpr int WARP_SIZE = 32;

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

int custom_matmul_kernel = 0; // 0=cuBLAS TF32, 1=TF32, 2=cuBLAS FP32, 3=FP32
cublasHandle_t cublas_handle;
cudaDeviceProp deviceProp;

// helpers for float4
__device__ inline float4 add_float4(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}
__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

// ----------------------------------------------------------------------------
// defines: matmul_forward (requires CUDA utils defined above)
#include "llmc/matmul_fp32.cuh"

// ----------------------------------------------------------------------------
// functions for warp/block/row reductions (used in softmax, layernorm, etc.)
// row reductions are a very useful primitive but limited to axis=0 with 1 row per block
// they could be generalised and used across more kernels but this goes beyond our current scope

// these structures are passed to the row/block/warp reduction functions
// row_reduction uses operator() while warp_reduce/block_reduce only use merge()
// e.g. for softmax, row_reduction() applies the complex op with expf once per element
// then calls block_reduction() (which calls warp_reduction()) to merge all the results
struct MaxOp {
    float identity = -FLT_MAX;
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ static float merge(float a, float b) { return fmaxf(a, b); }
};
struct SumOp {
    float identity = 0.0f;
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ static float merge(float a, float b) { return a + b; }
};
struct VarianceOp : public SumOp {
    float mean;
    __device__ VarianceOp(float m) : mean(m) {}
    __device__ float operator()(float a, float b) const { return a + (b - mean) * (b - mean); }
};
struct SoftmaxOp : public SumOp {
    float max, inv_t;
    __device__ SoftmaxOp(float mval, float inv_temperature) : max(mval), inv_t(inv_temperature) {}
    __device__ float operator()(float a, float b) const { return a + expf(inv_t * (b - max)); }
};

// warp reduction for exactly 32 threads (all must be active!) with result available to all threads
// e.g. warp_reduce<SumOp>(val) is equivalent to cg::reduce(warp, sum, cg::plus<float>{})
// for maximum flexibility, this could be extended to shuffling a structure rather than a float
template<typename Op>
__device__ inline float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = Op::merge(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// reduction for up to 1024 threads (1D grid & 1D block only) with result available to all threads
// (1) intra-warp (shuffle) (2) inter-warp (shared memory) (3) intra-warp (shuffle) (4) broadcast
// requires all threads in the threadgroup to be active(!) but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 132 bytes
// block_reduce<SumOp>(val) is similar to cub::BlockReduce<float,block_size>(tmp_shared).Sum(val)
template<typename Op>
__device__ inline float block_reduce(float val, float identity=0.0f) {
    __shared__ float shared_val[WARP_SIZE];
    __shared__ float broadcast_val;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduce<Op>(val); // (1)a
    if (lane_id == 0) { shared_val[warp_id] = warp_val; } // (2)a
    __syncthreads();
    if (warp_id == 0) {
        warp_val = (lane_id < num_warps) ? shared_val[lane_id] : identity; // (2)b
        broadcast_val = warp_reduce<Op>(warp_val); // (3) + (4)a
    }
    // we could avoid the final sync most of the time by making all warps do the final reduction
    // but we'd need to sync inside loops anyway since it'd reuse shared memory across iterations
    // this is definitely more power efficient + probably faster with enough parallel threadgroups
    __syncthreads();
    return broadcast_val; // (4)b
}

// reduction for any number of elements which are stored consecutively in global or shared memory
// requires 16B alignment due to float4 but supports a non-multiple-of-4 number of elements
// (we could handle non-aligned pointers with preprocessing, but it's not needed for llm.c)
// for additional flexibility, we'd ideally want to support any axis and >1 row per block
template<bool reverse_order = false, typename OpFunctor>
__device__ inline float row_reduce(const float* data, int elements, const OpFunctor& op) {
    assert((reinterpret_cast<uintptr_t>(data) % 16) == 0); // input pointer must be 16B aligned
    const float4* data4 = reinterpret_cast<const float4*>(data);
    float result = op.identity;

    // Main loop: Use float4 for aligned access (optionally in reverse to maximize cache hits)
    int loop_start = reverse_order ? (elements/4 + threadIdx.x - blockDim.x) : threadIdx.x;
    int loop_stride = reverse_order ? -blockDim.x : blockDim.x;
    for (int i = loop_start; reverse_order ? (i >= 0) : (i < elements/4); i += loop_stride) {
        float4 current4 = data4[i];
        for (int k = 0; k < 4; k++) {
            result = op(result, ((float*)&current4)[k]);
        }
    }
    // Handle remaining elements (not multiple of 4)
    int idx = threadIdx.x + (elements/4) * 4;
    if (idx < elements) { result = op(result, data[idx]); }
    // Return the result of the block-wide reduction
    return block_reduce<OpFunctor>(result, op.identity);
}

// copy N consecutive elements from global memory to shared memory (using TMA on H100+)
__device__ void copy_to_shared(float* out, const float* in, int elements, bool padding=true, float padding_value=0.0f) {
    assert((uintptr_t)in % 16 == 0); // input pointer must be 16B aligned
    assert((uintptr_t)out % 16 == 0); // output pointer must be 16B aligned
    size_t elements_rounded = elements - (elements % 4); // TMA/LDGSTS/float4 need 16B chunks

    // handle non-multiple-of-4 elements (up to 3 elements in total)
    if (threadIdx.x < 4 && elements % 4) {
        int i = elements_rounded + threadIdx.x;
        if (i < elements) { out[i] = in[i]; }
        else if (padding) { out[i] = padding_value; }
    }
    if (elements_rounded == 0) {
        __syncthreads();
        return;
    } // nothing left to do if there are less than 4 elements

    #if __CUDA_ARCH__ >= 900 // Hopper: use TMA in 1D mode
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar; // todo: reuse barrier if calling function multiple times per kernel
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        cuda::memcpy_async(out, in, cuda::aligned_size_t<16>(elements_rounded * sizeof(float)), bar);
    }
    __syncthreads(); // needed because of barrier initialisation (I *think* it's safe here?)
    bar.wait(std::move(bar.arrive())); // todo: only wait for the final call of copy_to_shared()

    #else // Older GPUs: just do a manual copy using float4 (128-bit load/stores)
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x; i < elements / 4; i += blockDim.x) {
        out4[i] = reinterpret_cast<const float4*>(in)[i];
    }
    __syncthreads();
    #endif
}

// ----------------------------------------------------------------------------
// all the kernels

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
// note that there is no native addition for float4 so we use our own add_float4
// as a more flexible alternative, we could use the kernel_float library on GitHub
// for the full version of llm.c, we created our on x128 class for maximum performance
__global__ void encoder_forward_kernel3(float4* out,
                               const int* inp, const float4* wte, const float4* wpe,
                               int B, int T, int C) {
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if (idx < N) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        // very good memory coalescing as "address = ... + c4" and "c4 = (... + threadIdx.x) % C4"
        // i.e. as long as C is a multiple of 128, every warp will read 2x 512 consecutive bytes
        // and write 512 consecutive bytes, which is the best case scenario really
        // (+a tiny scalar read of inp[] which is very low bandwidth)
        out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
    }
}

// naive non-deterministic kernel with atomicAdd
__global__ void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        // very good cacheline coalescing as "address = ... + c" and "c = (... + threadIdx.x) % C"
        // NVIDIA atomics are pretty fast when coalesced *and* most warps have different addresses
        // but FP32 atomics are still non-deterministic, see full llm.c for deterministic kernel
        atomicAdd(dwte_ix, *dout_btc);
        atomicAdd(dwpe_tc, *dout_btc);
    }
}

// layernorms require going over the same data multiple times, so we optimise
__global__ void layernorm_forward_kernel(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int C) {
    // the row of input that this group of threads is responsible for
    int row = blockIdx.x;
    const float* x = inp + row * C;

    // calculate the mean, variance, and reciprocal standard deviation of the row
    // we could also use the "Variance = E[x^2] - E[x]^2" formula to only go over the data once
    // but this is easier with row_reduce() and roughly the same performance (+better numerics?)
    float m = row_reduce<false>(x, C, SumOp()) / C;
    float v = row_reduce<true>(x, C, VarianceOp(m)) / C; // reverse order to maximise cache hits
    float s = rsqrtf(v + 1e-5f); // add a small epsilon to avoid divisions by zero

    // final normalization and scaling by weight/bias
    float* o = out + row * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x + c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }

    // store the mean and rstd for the backward pass later
    if(threadIdx.x == 0 && mean != nullptr) { mean[row] = m; }
    if(threadIdx.x == 0 && rstd != nullptr) { rstd[row] = s; }
}

// all permute & unpermute kernels (used for attention) could be optimised further using float4
// with PyTorch or cuDNN, this is handled automatically, with *some* matmul kernels able to fuse it
// but we can't fuse this with cuBLAS and different parts need different layouts, so do it manually
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, attention forward wants Q,K,V to all be of shape (B, NH, N, d)
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
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = __ldcs(&inp[inp_idx]);
        k[idx] = __ldcs(&inp[inp_idx + NH * d]);
        v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
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
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
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
        out[other_idx] = __ldcs(&inp[idx]);
    }
}

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__global__ void softmax_forward_kernel(float* out, float inv_temperature, const float* inp, int N, int T) {
    int idx = gridDim.x - blockIdx.x - 1; // backward order
    const float* x = inp + idx * T; // up to T elements per row (with B*T columns)
    int own_pos_plus1 = (idx % T) + 1; // causal masking: only look at previous elements

    extern __shared__ float shared_row[];
    copy_to_shared(shared_row, x, own_pos_plus1, true, -FLT_MAX); // bulk copy via TMA if available

    // 2 passes (not using online softmax): we calculate the maximum for the row, then the softmax
    // could be merged into a single pass with online softmax to avoid loading shared memory twice
    // but this would be much more complicated to handle in row_reduce, and only slightly faster
    // expf() is 16 threads/clk and 4-byte loads are 32 threads/clk, so we are ~limited by MUFU
    auto maxval = row_reduce(shared_row, own_pos_plus1, MaxOp());
    auto sumval = row_reduce(shared_row, own_pos_plus1, SoftmaxOp(maxval, inv_temperature));
    float norm = 1.0f / sumval;

    // note: this might go up to 3 elements beyond own_pos+1, this is OK because we have padding
    // on both input side (initialised to -FLT_MAX) and output side (where we will write 0)
    // this would not be correct if we loaded directly from global memory rather than shared_row
    for (int i = threadIdx.x * 4; i < own_pos_plus1; i += blockDim.x * 4) {
        float evn[4];
        for (int k = 0; k < 4; k++) {
            evn[k] = expf(inv_temperature * (shared_row[i+k] - maxval)) * norm;
        }
        *reinterpret_cast<float4*>(out + idx * T + i) = *reinterpret_cast<float4*>(evn);
    }
}

__global__ void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    // fully optimal float4 kernel (the only way to make it faster is to fuse it with layernorm)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        const float4 inp1_float4 = __ldcs(reinterpret_cast<const float4*>(inp1 + idx));
        const float4 inp2_float4 = __ldcs(reinterpret_cast<const float4*>(inp2 + idx));
        st_vec(out + idx, add_float4(inp1_float4, inp2_float4));
    }
}

// could be optimised using float4
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

// could be optimised using float4
__global__ void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// cannot use row_reduce() as it reduces along column elements which are not consecutive in memory
// ideally we'd have fused "transpose + row_reduce" kernels for column reduction but it's tricky
//
// the solution is to employ one block to reduce along several columns in parallel,
// where each block has a width of blockIdx.x columns to ensure coalesced access.
// and we also process blockIdx.y rows in parallel to maximise GPU utilization,
// at the end we combine the reductions of the rows with shared memory.
template <int block_dim_x=8, int block_dim_y=64> // specified at compile time to allow unrolling
__global__ void matmul_backward_bias_kernel5(float* dbias, const float* dout, int B, int T, int OC) {
    // blockDim.x ==> 8 columns (total of OC columns) ==> 32 bytes per load (NVIDIA L1 sector size)
    // blockDim.y ==> 64 rows (total of B*T rows)
    // we want the sum of all the elements of each column individually (one output per column)
    constexpr int block_size = block_dim_x * block_dim_y;
    __shared__ float smem[block_size];

    // let each thread in a sub-group of blockDim.x threads process its part of the column
    float dout_sum = 0.0f;
    int column = blockIdx.x * block_dim_x + threadIdx.x;
    int smem_idx = threadIdx.x + threadIdx.y * block_dim_x; // idx for this specific thread
    for (int row = threadIdx.y; row < B * T; row += block_dim_y) {
        // maybe faster using float4 but hard trade-off (4x columns in parallel per block_dim_x!)
        dout_sum += dout[column + row * OC];
    }
    smem[smem_idx] = dout_sum; // write partial sums to shared memory

    // blockDim.y threads are processing the same column, so we need to reduce their sum
    // i.e. we calculate blockDim.x sums in parallel (one per column)
    for (int stride = block_size/2; stride >= block_dim_x; stride /= 2) {
        __syncthreads();
        if (threadIdx.y * block_dim_x < stride) {
            smem[smem_idx] = smem[smem_idx] + smem[smem_idx + stride];
        }
    }

    // accumulate blockDim.x sums in global memory (one per column)
    if (threadIdx.y == 0) {
        dbias[column] += smem[threadIdx.x]; // bias parameter gradient per column/OC
    }
}

// this kernel handles multiple rows per thread block (one per warp)
// that is to reduce the number of global memory atomics which would kill performance otherwise,
// which is why we do not use row_reduce (it would work but require a global memory atomic per row)
// alternatively, row_reduce could be generalised to do multiple rows in parallel for 2D blocks
// to be deterministic, the implementation in train_gpt2.cu is completely different and atomic-free
template <int block_size = 512>
__global__ void layernorm_backward_kernel(float* dinp, float* dweight, float* dbias,
                                           const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                                           int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;
	for(int i = threadIdx.x; i < C*2; i+= block_size){
       shared[i] = 0.0f; // init shared memory to zero (before the out-of-bounds check)
    }

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = block_size / 32;
    int idx = blockIdx.x * warps_per_block + warp_id;
    int N = B * T;
    if(idx >= N) { return; } // out-of-bounds check after initialising shared memory

    int b = idx / T;
    int t = idx % T;
    int offset_bt = (b * T + t);
    float* dinp_bt = dinp + offset_bt * C;
    const float* dout_bt = dout + offset_bt * C;
    const float* inp_bt = inp + offset_bt * C;
    const float mean_bt = mean[offset_bt];
    const float rstd_bt = rstd[offset_bt];

    // first: two reduce operations (dnorm_mean and dnorm_norm_mean)
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    // all reductions are at warp granularity (1 row per warp, multiple rows in parallel per block)
    dnorm_mean = warp_reduce<SumOp>(dnorm_mean) / C;
    dnorm_norm_mean = warp_reduce<SumOp>(dnorm_norm_mean) / C;
    __syncthreads(); // todo - for the shared memory initialisation, ideally would use arrive/wait

    // now iterate again and and accumulate dbias/dweight in shared memory
    // note that FP32 shared memory atomics are actually emulated on (all?) NVIDIA GPUs using CAS
    // so they are even more sensitive to conflicts when memory addresses are the same, but OK here
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias (all rows contribute to the same bias which is per-C)
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // add everything together in global memory (non-deterministic due to floating point atomics)
    // if we have blocks of 512 threads with every 32 threads handling one BT element, that means
    // we only need to do 1/16th as many global memory atomics (but many shared memory atomics)
	for(int i = threadIdx.x; i < C; i+= block_size){
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
	}
}

// todo: could be optimized with float4 but needs to handle unaligned pointers & non-multiple-of-4 elements
__global__ void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale) {
    // go through blocks in reverse order, so the slowest block starts first
    // blocks need to process between 1 and T elements, so their runtime varies a lot
    // this kernel is faster without shared memory because we'd need to reserve for the worst case
    int t = T - 1 - blockIdx.x;

    size_t bth_offset = (blockIdx.y * T * T) + (t * T);
    const float* att_bth = att + bth_offset;
    const float* datt_bth = datt + bth_offset;
    float* dpreatt_bth = dpreatt + bth_offset;

    float local_sum = 0;
    for (int t2 = threadIdx.x; t2 <= t; t2 += blockDim.x) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }
    local_sum = block_reduce<SumOp>(local_sum);

    for (int t3 = threadIdx.x; t3 <= t; t3 += blockDim.x) {
        // hopefully att/datt will still be in the L1 cache from the t2 loop above
        // they're not needed after so load/store with .cs hint to avoid thrashing L1/L2 caches
        float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
        __stcs(dpreatt_bth + t3, scale * acc);
    }
}

// Implements linear interpolation using only two floating-point FMA operations
// as opposed to three operations in naive "(1-t)*a + t*b" (FADD+FMUL+FMA)
// same performance but more precise than "a + t*(b-a)" (FADD+FMA)
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float fast_lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= num_parameters) return;
   float grad = grads_memory[i];
   // update the first moment (momentum)
   float m = fast_lerp(grad, m_memory[i], beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   float v = fast_lerp(grad * grad, v_memory[i], beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

// this kernel will _update_ logits to logit gradients
template <bool write_probs=false>
__global__ void __launch_bounds__(1024, 2) // warning for GPUs with fewer threads can be ignored
fused_classifier_kernel(float* logits, float* losses, float* probs,
                        const float* dlosses, const int* targets,
                        int B, int T, int V, int P) {
    int idx = blockIdx.x;
    int ix = targets[idx];

    // softmax: reading B * T * V, same logits read 3 times, 2x by row_reduce and 1x in loop below
    // V is ~50K for GPT2 which is 200KiB in FP32 and too big to fit in shared memory on most GPUs
    // but H100 has 50MiB L2 = ~384KiB per SM => roughly enough for 2 full threadblocks in parallel
    const float* x = logits + idx * P;
    auto maxval = row_reduce(x, V, MaxOp());
    auto sumval = row_reduce<true /*reversed order*/>(x, V, SoftmaxOp(maxval, 1.0f));
    float softmax_scale = 1.0f / sumval;

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - maxval) * softmax_scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the final read of logits after the ones in row_reduce
        // this data will never be needed again, but it will be overwritten => .cs for the *store*
        float v = logits_vec[i];
        float prob = expf(v - maxval) * softmax_scale;
        if constexpr (write_probs) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        __stcs(&logits[idx * P + i], (prob - indicator) * dloss); // .cs to avoid thrashing L1/L2
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(float* out,
                     const int* inp, const float* wte, const float* wpe,
                     int B, int T, int C) {
    assert(C % 4 == 0);
    const int block_size = 512;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N / 4, block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4*) out, inp, (float4*) wte, (float4*) wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = min(C / 4, 1024);
    const int grid_size = B * T;
    layernorm_forward_kernel<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, C);
    cudaCheck(cudaGetLastError());
}

void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 128;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS
    // todo - could use compute_tf32gemm_async_copy but tricky to get the layouts right etc...
    // + would need to add support for strided batched (easy given it's using persistent threads?)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);

    // we want 100% of the L1 to be used for (dynamic) shared memory to maximise occupancy
    cudaFuncSetAttribute(softmax_forward_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    size_t smem = (T + 4) * sizeof(float); // round up T (should already be a multiple of 4)
    softmax_forward_kernel<<<B * NH * T, softmax_block_size, smem>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    assert((N % 4) == 0); // check we have a multiple of 4 elements for float4
    assert(uintptr_t(out) % 16 == 0 && uintptr_t(inp1) % 16 == 0 && uintptr_t(inp2) % 16 == 0);
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size * 4); // 4 floats per thread
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // todo - could use compute_tf32gemm_async_copy but tricky to get the layouts right etc...
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        constexpr dim3 block_size = dim3(8, 64); // block_size.y threads reduce one column of the bias
        const dim3 grid_size = dim3(OC / block_size.x); // block_size.x columns are processed per block
        assert(OC % block_size.x == 0); // should always be true for sensible values of block_size.x
        matmul_backward_bias_kernel5<block_size.x, block_size.y><<<grid_size, block_size>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T; // B*T rows
    const int grid_size = CEIL_DIV(32*N, block_size); // each warp is responsible for one row
    size_t shared_mem_size = 2 * C * sizeof(float); // 2*C shared memory for dbias and dweight
    layernorm_backward_kernel<block_size><<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size (64 for GPT2 and 128/256 for many newer models)
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T, B * NH), 128>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// replaces logits with logit gradients
void fused_classifier(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int grid_size = B * T;
    cudaFuncSetAttribute(fused_classifier_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    fused_classifier_kernel<false><<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int on_device) {
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    float* params_memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(float)));
    } else {
        params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    }
    // assign all the tensors their place in the array
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 21
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* atty; // (L, B, T, C)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)

    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    float* output;
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * C; // atty
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = L * B * T; // ln2_mean
    act_sizes[10] = L * B * T; // ln2_rstd
    act_sizes[11] = L * B * T * 4*C; // fch
    act_sizes[12] = L * B * T * 4*C; // fch_gelu
    act_sizes[13] = L * B * T * C; // fcproj
    act_sizes[14] = L * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = B * T; // lnf_mean
    act_sizes[17] = B * T; // lnf_rstd
    act_sizes[18] = B * T; // losses
    act_sizes[19] = L * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * max(3*C, max(NH*T, Vp)); // output / scratch
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#define NUM_BACKWARD_TENSORS 3
typedef struct {
    float* bt4c; // (B, T, 4*C)
    float* preatt; // (B, NH, T, T)
    float* residual3; // (B, T, C)
} GradActTensors;


void fill_in_grad_act_sizes(size_t* act_sizes, int B, int T, GPT2Config config) {
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C; // bt4c
    act_sizes[1] = B * NH * T * T; // preatt
    act_sizes[2] = B * T * C; // residual3
}


float* malloc_and_point(float** targets[], const size_t* act_sizes, int n) {
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float* malloc_and_point_activations(ActivationTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float* malloc_and_point_backward(GradActTensors* acts, const size_t* act_sizes) {
    float** ptrs[] = {
        &acts->bt4c, &acts->preatt, &acts->residual3
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL) {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float* scratch = acts.output;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->cpu_losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;

    } else {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if (model->grads_acts_memory != NULL) { cudaCheck(cudaMemset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float))); }
    if (model->grads_memory != NULL) { cudaCheck(cudaMemset(model->grads_memory, 0, model->num_parameters * sizeof(float))); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        // allocate buffers for weight gradients
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        GPT2Config cfg = model->config;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++) {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        // init gradients of parameters and activations to zero
        gpt2_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    // backward the final layernorm
    float* residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    float* dresidual = grads_acts.residual3; // the main buffer holding the gradient in the backward pass
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // now backward all the layers
    for (int l = L-1; l >= 0; l--) {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        float* dl_btc = acts.lnf;
        float* dl_bt4c = grads_acts.bt4c;
        float* dl_preatt = grads_acts.preatt;

        // re-use scratch buffer of the forward pass
        float* scratch = acts.output;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_bt4c, l_fch, dl_bt4c, B*T*4*C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        float* buffer_a = l_atty;
        float* buffer_b = l_fch;        // this is B x T x 4C, so even larger than what we need

        attention_backward(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, dresidual, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        cudaCheck(cudaMalloc((void**)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<num_blocks, block_size>>>(model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                                              model->num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct {
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename) {
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL) { logger->logfile = fopenCheck(filename, "w"); }
}

void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss) {
    if (logger->logfile != NULL) {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0) { fflush(logger->logfile); }
    }
}

void logger_free(Logger *logger) {
    if (logger->logfile != NULL) { fclose(logger->logfile); }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2fp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -c <int>    custom forward matmul (0=default cuBLAS TF32, 1=TF32, 2=cuBLAS FP32, 3=FP32\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {

    // read in the (optional) command line arguments
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* output_log_file = NULL;
    int B = 4; // batch size
    int T = 1024; // sequence length max
    float learning_rate = 3e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_file = argv[i+1]; }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); }
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'c') { custom_matmul_kernel = atoi(argv[i+1]); }
        else { error_usage(); }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("| custom matmul         | %-50d |\n", custom_matmul_kernel);
    printf("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = (deviceProp.major >= 8 ? 1 : 0) && (custom_matmul_kernel <= 1);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    printf("| device                | %-50s |\n", deviceProp.name);
    printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
    printf("+-----------------------+----------------------------------------------------+\n");

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf("| channels C            | %-50d |\n", model.config.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = train_loader.num_tokens / (B*T); // let's do 1 epoch by default for now
    int val_num_batches = val_loader.num_tokens / (B*T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");

    // print model parameter allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf("allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));

    // set up the Logger
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    float* cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++) {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % sample_every == 0 || last_step) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                float* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits, logits, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_sum_iteration_time_s += time_elapsed_s;
        int tokens_per_second = (B * T) / time_elapsed_s;
        printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(cpu_logits);
    free(gen_tokens);
    cublasCheck(cublasDestroy(cublas_handle));
    logger_free(&logger);

    return 0;
}
#endif