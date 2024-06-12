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

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <dpct/blas_utils.hpp>

// GPU / CUDA related
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"
#include <dpct/lib_common_utils.hpp>

// ----------------------------------------------------------------------------
// CUDA utils

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// CUDA error checking
void cudaCheck(dpct::err0 error, const char *file, int line) {

};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(int status, const char *file, int line)
{
    if (status != 0) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static dpct::library_data_t cublas_compute_type;
dpct::blas::descriptor_ptr cublas_handle;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// all the kernels

inline sycl::float4 add_float4(const sycl::float4 &a, const sycl::float4 &b) {
    return sycl::float4(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(),
                        a.w() + b.w());
}

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
void encoder_forward_kernel3(sycl::float4 *out, const int *inp,
                             const sycl::float4 *wte, const sycl::float4 *wpe,
                             int B, int T, int C,
                             const sycl::nd_item<3> &item_ct1) {
    int C4 = C / 4;
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int N = B * T * C4;
    if (idx < N) {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        out[b * T * C4 + t * C4 + c4] = add_float4(wte[ix * C4 + c4], wpe[t * C4 + c4]);
    }
}

// really bad naive kernel with atomicAdd
void encoder_backward_kernel(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C,
                                        const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            dwte_ix, *dout_btc);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            dwpe_tc, *dout_btc);
    }
}

SYCL_EXTERNAL void layernorm_forward_kernel3(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C,
    const sycl::nd_item<3> &item_ct1) {
    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    /*
    DPCT1007:254: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = item_ct1.get_group(2) * warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        sum += x[i];
    }
    sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sum,
                                  sycl::plus<float>{});
    float m = sum / C;
    if (item_ct1.get_sub_group().get_local_linear_id() == 0 &&
        mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sum,
                                  sycl::plus<float>{});
    float s = sycl::rsqrt(sum / C + 1e-5f);
    if (item_ct1.get_sub_group().get_local_linear_id() == 0 &&
        rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = item_ct1.get_sub_group().get_local_linear_id(); c < C;
         c += item_ct1.get_sub_group().get_local_linear_range()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

SYCL_EXTERNAL void permute_kernel(float *q, float *k, float *v,
                                  const float *inp, int B, int N, int NH, int d,
                                  const sycl::nd_item<3> &item_ct1) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

SYCL_EXTERNAL void permute_kernel_backward(float *dinp, const float *dq,
                                           const float *dk, const float *dv,
                                           int B, int N, int NH, int d,
                                           const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d,
                      const sycl::nd_item<3> &item_ct1) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

SYCL_EXTERNAL void unpermute_kernel_backward(float *dinp, const float *dout,
                                             int B, int N, int NH, int d,
                                             const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
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

SYCL_EXTERNAL float &vec_at(sycl::float4 &vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

SYCL_EXTERNAL float vec_at(const sycl::float4 &vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

SYCL_EXTERNAL void softmax_forward_kernel5(float *out, float inv_temperature,
                                           const float *inp, int N, int T,
                                           const sycl::nd_item<3> &item_ct1) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(0);
    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    /*
    DPCT1007:255: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) *
                  warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id(); // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const sycl::float4 *x_vec = reinterpret_cast<const sycl::float4 *>(x);
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < pos_by_4;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        sycl::float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, vec_at(v, k));
        }
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval +=
                sycl::native::exp(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if (4 * pos_by_4 + item_ct1.get_sub_group().get_local_linear_id() <=
        own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(
            maxval, (float)(x[4 * pos_by_4 +
                              item_ct1.get_sub_group().get_local_linear_id()]));
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::native::exp(
            inv_temperature *
            (x[4 * pos_by_4 + item_ct1.get_sub_group().get_local_linear_id()] -
             maxval));
    }

    float global_maxval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), maxval, sycl::maximum<float>{});
    sumval *= sycl::native::exp(inv_temperature * (maxval - global_maxval));

    float sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sumval,
                                        sycl::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i <= own_pos;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

void residual_forward_kernel(float* out, float* inp1, float* inp2, int N,
                             const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < N) {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}

#define GELU_SCALING_FACTOR sycl::sqrt((float)(2.0f / M_PI))
void gelu_forward_kernel(float* out, const float* inp, int N,
                         const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] =
            0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

void gelu_backward_kernel(float* dinp, const float* inp, const float* dout, const int N,
                          const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
void matmul_backward_bias_kernel4(float* dbias, const float* dout, int B, int T, int OC,
                                  const sycl::nd_item<3> &item_ct1,
                                  uint8_t *dpct_local) {
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    auto smem = (float *)dpct_local; // of size block_size (128)
    const int warp_id = item_ct1.get_local_id(2) /
                        item_ct1.get_sub_group().get_local_range().get(
                            0); // warp index in the block, 0,1,2,3
    const int lane_id = item_ct1.get_local_id(2) %
                        item_ct1.get_sub_group().get_local_range().get(
                            0); // thread index in the warp, 0,1,2,...,31
    const int tl = item_ct1.get_group(2) *
                   item_ct1.get_sub_group().get_local_range().get(
                       0); // pointer to the start column for this block
    const int vstep = item_ct1.get_local_range(2) /
                      item_ct1.get_sub_group().get_local_range().get(
                          0); // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float* dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep) {
        dout_sum += dout_col[row * OC];
    }
    smem[lane_id + warp_id * item_ct1.get_sub_group().get_local_range().get(
                                 0)] = dout_sum;
    /*
    DPCT1065:256: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0) {
        for (int j = 0; j < vstep; j++) {
            dout_sum +=
                smem[lane_id +
                     j * item_ct1.get_sub_group().get_local_range().get(0)];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

// uses shared memory instead for the reduces
/*
DPCT1110:19: The total declared local variable size in device function
layernorm_backward_kernel2 exceeds 128 bytes and may cause high register
pressure. Consult with your hardware vendor to find the total register size
available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void layernorm_backward_kernel2(float *dinp, float *dweight, float *dbias,
                                const float *dout, const float *inp,
                                const float *weight, const float *mean,
                                const float *rstd, int B, int T, int C,
                                const sycl::nd_item<3> &item_ct1,
                                uint8_t *dpct_local) {
    auto shared = (float *)dpct_local; // size = 2 * C

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    /*
    DPCT1007:259: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = item_ct1.get_group(2) * warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id();
    int N = B * T;
    if(idx >= N) { return; } // thread guards

    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll
        for (int i = item_ct1.get_local_id(2); i < C;
             i += item_ct1.get_local_range(2)) {
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    /*
    DPCT1065:257: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = sycl::reduce_over_group(item_ct1.get_sub_group(), dnorm_mean,
                                         sycl::plus<float>{});
    dnorm_norm_mean = sycl::reduce_over_group(
        item_ct1.get_sub_group(), dnorm_norm_mean, sycl::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i; // term 1
        dval -= dnorm_mean; // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt; // final scale
        dinp_bt[i] += dval;
    }
    /*
    DPCT1065:258: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write to global memory
        for (int i = item_ct1.get_local_id(2); i < C;
             i += item_ct1.get_local_range(2)) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dbias[i], dbias_shared[i]);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &dweight[i], dweight_shared[i]);
        }
}

void softmax_autoregressive_backward_kernel(float* dpreatt, const float* datt, const float* att,
                                                       int B, int T, int C, float scale,
                                                       const sycl::nd_item<3> &item_ct1,
                                                       float *block_acc) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    int idx = item_ct1.get_group(1);
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block * item_ct1.get_group(2);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (item_ct1.get_sub_group().get_group_linear_id() == 0) {
        block_acc[item_ct1.get_sub_group().get_local_linear_id()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = item_ct1.get_local_linear_id(); t2 <= t;
             t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[item_ct1.get_sub_group().get_group_linear_id()] =
            sycl::reduce_over_group(item_ct1.get_sub_group(), local_sum,
                                    sycl::plus<float>{});
        /*
        DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        local_sum = sycl::reduce_over_group(
            item_ct1.get_sub_group(),
            block_acc[item_ct1.get_sub_group().get_local_linear_id()],
            sycl::plus<float>{});

        for (int t3 = item_ct1.get_local_linear_id(); t3 <= t;
             t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
SYCL_EXTERNAL inline float lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

void adamw_kernel2(float* params_memory, float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              const sycl::nd_item<3> &item_ct1) {
   int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
           item_ct1.get_local_id(2);
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sycl::sqrt(v) + eps) +
                                        weight_decay * params_memory[i]);
}

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams
prepare_softmax_blockwide_nofloat4(sycl::sub_group &warp, int idx,
                                   const float *inp, int V, int P,
                                   const sycl::nd_item<3> &item_ct1,
                                   float *shared_maxval, float *shared_sumval) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + item_ct1.get_local_id(2) - item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = sycl::fmax(thread_maxval, v);
        thread_sumval *= sycl::native::exp((old_maxval - thread_maxval));
        thread_sumval += sycl::native::exp(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce

    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;

    // reduce maxval within each warp
    float warp_maxval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), thread_maxval, sycl::maximum<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    /*
    DPCT1065:260: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), warp_maxval, sycl::maximum<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), thread_sumval, sycl::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    /*
    DPCT1065:261: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), warp_sumval, sycl::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
void fused_classifier_kernel3(float* logits, float* losses, float* probs,
                                         const float* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_maxval,
                                         float *shared_sumval) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    int idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(
        warp, idx, logits, V, P, item_ct1, shared_maxval, shared_sumval);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < V;
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = sycl::native::exp(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
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
    /*
    DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            encoder_forward_kernel3((sycl::float4 *)out, inp,
                                    (sycl::float4 *)wte, (sycl::float4 *)wpe, B,
                                    T, C, item_ct1);
        });
    /*
    DPCT1010:262: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            encoder_backward_kernel(dwte, dwpe, dout, inp, B, T, C, item_ct1);
        });
    /*
    DPCT1010:263: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    /*
    DPCT1049:22: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel3(out, mean, rstd, inp, weight, bias, N, C,
                                      item_ct1);
        });
    /*
    DPCT1010:264: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// uses cuBLASLt to fuse the bias and gelu. does not work with OC = 50257 (last layer)
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward_cublaslt(float* out,
                     float* inp, float* weight, float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    oneapi::mkl::transpose opNoTranspose = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose opTranspose = oneapi::mkl::transpose::trans;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    /*
    DPCT1007:265: Migration of cublasLtMatmulDescCreate is not supported.
    */
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type,
                                         dpct::library_data_t::real_float));
    /*
    DPCT1007:266: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose,
        sizeof(opTranspose)));
    /*
    DPCT1007:267: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose,
        sizeof(opNoTranspose)));
    if(has_bias) {
        /*
        DPCT1007:268: Migration of cublasLtMatmulDescSetAttribute is not
        supported.
        */
        cublasCheck(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias,
            sizeof(epilogueBias)));
    }
    /*
    DPCT1007:269: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    /*
    DPCT1007:270: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &weightLayout, dpct::library_data_t::real_float, C, OC, C));
    /*
    DPCT1007:271: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &inputLayout, dpct::library_data_t::real_float, C, B * T, C));
    /*
    DPCT1007:272: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &outputLayout, dpct::library_data_t::real_float, OC, B * T, OC));
    /*
    DPCT1007:273: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &biasLayout, dpct::library_data_t::real_float, OC, 1, OC));

    // create a preference handle with specified max workspace
    /*
    DPCT1007:274: Migration of cublasLtMatmulPreferenceCreate is not supported.
    */
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    /*
    DPCT1007:275: Migration of cublasLtMatmulPreferenceSetAttribute is not
    supported.
    */
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    /*
    DPCT1007:276: Migration of cublasLtMatmulAlgoGetHeuristic is not supported.
    */
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, weightLayout, inputLayout, outputLayout,
        outputLayout, preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d\n", B, T, C, OC, has_bias);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    /*
    DPCT1007:277: Migration of cublasLtMatmul is not supported.
    */
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, weight,
                               weightLayout, inp, inputLayout, &beta, out,
                               outputLayout, out, outputLayout, &heuristic.algo,
                               cublaslt_workspace, cublaslt_workspace_size,
                               &dpct::get_in_order_queue()));

    // cleanups
    /*
    DPCT1007:278: Migration of cublasLtMatmulPreferenceDestroy is not supported.
    */
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    /*
    DPCT1007:279: Migration of cublasLtMatmulDescDestroy is not supported.
    */
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    /*
    DPCT1007:280: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    /*
    DPCT1007:281: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    /*
    DPCT1007:282: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    /*
    DPCT1007:283: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

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
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, T, NH, HS, item_ct1);
        });
    /*
    DPCT1010:284: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha, k, HS, T * HS, q, HS,
        T * HS, beta, preatt, T, T * T, B * NH)));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        int B_NH_ct3 = B * NH;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, softmax_block_size),
                              sycl::range<3>(1, 1, softmax_block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel5(att, scale, preatt, B_NH_ct3, T,
                                        item_ct1);
            });
    });
    /*
    DPCT1010:285: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);

    // new approach: first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha, v, HS, T * HS, att,
        T, T * T, beta, vaccum, HS, T * HS, B * NH)));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(vaccum, out, B, T, NH, HS, item_ct1);
        });
    /*
    DPCT1010:286: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            residual_forward_kernel(out, inp1, inp2, N, item_ct1);
        });
    /*
    DPCT1010:287: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_forward_kernel(out, inp, N, item_ct1);
        });
    /*
    DPCT1010:288: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_backward_kernel(dinp, inp, dout, N, item_ct1);
        });
    /*
    DPCT1010:289: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, C, B * T, OC,
        dpct::get_value(&one, cublas_handle->get_queue()), weight, C, dout, OC,
        dpct::get_value(&zero, cublas_handle->get_queue()), dinp, C)));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, C, OC, B * T,
        dpct::get_value(&one, cublas_handle->get_queue()), inp, C, dout, OC,
        dpct::get_value(&one, cublas_handle->get_queue()), dweight, C)));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 1024;
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        /*
        DPCT1049:23: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1083:543: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(block_size * sizeof(float)), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                      sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    matmul_backward_bias_kernel4(
                        dbias, dout, B, T, OC, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
        /*
        DPCT1010:290: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        cudaCheck(0);
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(32*N, block_size);
    /*
    DPCT1083:25: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = 2 * C * sizeof(float);
    /*
    DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                layernorm_backward_kernel2(
                    dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T,
                    C, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:291: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
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
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel_backward(scratch, dout, B, T, NH, HS, item_ct1);
        });
    /*
    DPCT1010:292: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
    // backward into datt
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, one, v, HS, T * HS, scratch,
        HS, T * HS, zero, datt, T, T * T, B * NH)));
    // backward into dv
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, one, scratch, HS, T * HS, att,
        T, T * T, zero, dv, HS, T * HS, B * NH)));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> block_acc_acc_ct1(sycl::range<1>(32),
                                                         cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, B * NH, T / 4) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_autoregressive_backward_kernel(
                    dpreatt, datt, att, B, T, C, scale, item_ct1,
                    block_acc_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:293: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
    // backward into q
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, one, k, HS, T * HS, dpreatt,
        T, T * T, zero, dq, HS, T * HS, B * NH)));
    // backward into k
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, one, q, HS, T * HS, dpreatt, T,
        T * T, zero, dk, HS, T * HS, B * NH)));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel_backward(dinp, dq, dk, dv, B, T, NH, HS, item_ct1);
        });
    /*
    DPCT1010:294: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// replaces logits with logit gradients
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_maxval_acc_ct1(sycl::range<1>(32),
                                                             cgh);
        sycl::local_accessor<float, 1> shared_sumval_acc_ct1(sycl::range<1>(32),
                                                             cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                fused_classifier_kernel3(
                    logits, losses, NULL, dlosses, targets, B, T, V, P,
                    item_ct1,
                    shared_maxval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sumval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:295: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct dpct_type_981750 {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct dpct_type_137057 {
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
        /*
        DPCT1064:529: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(
            DPCT_CHECK_ERROR(params_memory = sycl::malloc_device<float>(
                                 num_parameters, dpct::get_in_order_queue())));
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
typedef struct dpct_type_171021 {
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
    act_sizes[20] =
        B * T * dpct::max(3 * C, std::max(NH * T, Vp)); // output / scratch
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#define NUM_BACKWARD_TENSORS 3
typedef struct dpct_type_369313 {
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
    /*
    DPCT1064:530: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    cudaCheck(
        DPCT_CHECK_ERROR(acts_memory = sycl::malloc_device<float>(
                             num_activations, dpct::get_in_order_queue())));
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

typedef struct dpct_type_166660 {
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
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(model->params_memory, params_memory_cpu,
                                     num_parameters * sizeof(float))
                             .wait()));
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
        /*
        DPCT1064:531: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(DPCT_CHECK_ERROR(model->inputs = sycl::malloc_device<int>(
                                       B * T, dpct::get_in_order_queue())));
        /*
        DPCT1064:532: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(DPCT_CHECK_ERROR(model->targets = sycl::malloc_device<int>(
                                       B * T, dpct::get_in_order_queue())));
        /*
        DPCT1064:533: Migrated cudaMallocHost call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(DPCT_CHECK_ERROR(model->cpu_losses = sycl::malloc_host<float>(
                                       B * T, dpct::get_in_order_queue())));
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(model->inputs, inputs, B * T * sizeof(int))
                             .wait()));
    if (targets != NULL) {
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(model->targets, targets, B * T * sizeof(int))
                .wait()));
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
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward_cublaslt(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward_cublaslt(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward_cublaslt(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memcpy(model->cpu_losses, acts.losses, B * T * sizeof(float))
                .wait()));
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
    if (model->grads_acts_memory != NULL) {
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(model->grads_acts_memory, 0,
                                         model->num_grad_acts * sizeof(float))
                                 .wait()));
    }
    if (model->grads_memory != NULL) {
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(model->grads_memory, 0,
                                         model->num_parameters * sizeof(float))
                                 .wait()));
    }
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
        /*
        DPCT1064:534: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(DPCT_CHECK_ERROR(
            model->m_memory = sycl::malloc_device<float>(
                model->num_parameters, dpct::get_in_order_queue())));
        /*
        DPCT1064:535: Migrated cudaMalloc call is used in a macro/template
        definition and may not be valid for all macro/template uses. Adjust the
        code.
        */
        cudaCheck(DPCT_CHECK_ERROR(
            model->v_memory = sycl::malloc_device<float>(
                model->num_parameters, dpct::get_in_order_queue())));
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(model->m_memory, 0,
                                         model->num_parameters * sizeof(float))
                                 .wait()));
        cudaCheck(
            DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memset(model->v_memory, 0,
                                         model->num_parameters * sizeof(float))
                                 .wait()));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    /*
    DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        float *model_params_memory_ct0 = model->params_memory;
        float *model_grads_memory_ct1 = model->grads_memory;
        float *model_m_memory_ct2 = model->m_memory;
        float *model_v_memory_ct3 = model->v_memory;
        long model_num_parameters_ct4 = model->num_parameters;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                               sycl::range<3>(1, 1, block_size),
                                           sycl::range<3>(1, 1, block_size)),
                         [=](sycl::nd_item<3> item_ct1) {
                             adamw_kernel2(
                                 model_params_memory_ct0,
                                 model_grads_memory_ct1, model_m_memory_ct2,
                                 model_v_memory_ct3, model_num_parameters_ct4,
                                 learning_rate, beta1, beta2, beta1_correction,
                                 beta2_correction, eps, weight_decay, item_ct1);
                         });
    });
    /*
    DPCT1010:296: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void gpt2_free(GPT2 *model) {
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->params_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->grads_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->m_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->v_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->acts_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->grads_acts_memory, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->inputs, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(model->targets, dpct::get_in_order_queue())));
    sycl::free(model->cpu_losses, dpct::get_in_order_queue());
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

typedef struct dpct_type_114836 {
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
    printf("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    int deviceIdx = 0;
    /*
    DPCT1093:297: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    dpct::get_device_info(deviceProp,
                          dpct::dev_mgr::instance().get_device(deviceIdx));
    // setup cuBLAS and cuBLASLt
    cublasCheck(DPCT_CHECK_ERROR(cublas_handle = new dpct::blas::descriptor()));
    /*
    DPCT1007:298: Migration of cublasLtCreate is not supported.
    */
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    /*
    DPCT1005:299: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    int enable_tf32 = deviceProp.get_major_version() >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    int cublas_math_mode =
        enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    /*
    DPCT1027:300: The call to cublasSetMathMode was replaced with 0 because this
    functionality is redundant in SYCL.
    */
    cublasCheck(0);
    cudaCheck(DPCT_CHECK_ERROR(
        cublaslt_workspace = (void *)sycl::malloc_device(
            cublaslt_workspace_size, dpct::get_in_order_queue())));
    printf("| device                | %-50s |\n", deviceProp.get_name());
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
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1);
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
                cudaCheck(DPCT_CHECK_ERROR(
                    dpct::get_in_order_queue()
                        .memcpy(cpu_logits, logits,
                                model.config.vocab_size * sizeof(float))
                        .wait()));
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
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_current_device()
                .queues_wait_and_throw())); // finish all CUDA work to get
                                            // correct precise timings
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
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(cublaslt_workspace, dpct::get_in_order_queue())));
    cublasCheck(DPCT_CHECK_ERROR(delete (cublas_handle)));
    /*
    DPCT1007:301: Migration of cublasLtDestroy is not supported.
    */
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    logger_free(&logger);

    return 0;
}
#endif