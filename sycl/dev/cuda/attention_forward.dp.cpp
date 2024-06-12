/*
Kernels for attention forward pass.

If you do not have CUDNN, you can remove ENABLE_CUDNN to run the other kernels

See the README for cuDNN install instructions

Compile example with cuDNN:
nvcc -I/PATH/TO/cudnn-frontend/include -DENABLE_CUDNN -O3 --use_fast_math --lcublas -lcublasLt -lcudnn attention_forward.cu -o attention_forward

Compile example without cuDNN:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward

version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
./attention_forward 1

version 2 is a naive implementation of flash attention, taken, adapted from
https://github.com/tspeterkim/flash-attention-minimal
and with help from
https://github.com/leloykun/flash-hyperbolic-attention-minimal
sadly, this flash attention version seems about 3X slower than the naive version
./attention_forward 2

version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
cuBLAS is used both to calculate the QK^T and the final weighted sum
the softmax is calculated using a custom, efficient kernel as well
this turns out to be ~20X faster than (1) nice
./attention_forward 3

version 4 is a further optimized kernel that fuses the scale operation,
uses a directly autoregressive softmax, and uses the online softmax algorithm.
./attention_forward 4

version 5 is a FP16 version of kernel 4
./attention_forward 5

version 6 is kernel 5 skipping (un)permute (unrealistic but useful comparison point)

version 10 is using cuDNN Flash Attention using FP16 or BF16, see:
https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md
./attention_forward 10

version 11 is kernel 10 skipping FP16/FP32 conversions (full FP16/BF16 network)
./attention_forward 11
*/
//#define ENABLE_CUDNN // can be enabled via nvcc "-DENABLE_CUDNN"
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <dpct/blas_utils.hpp>

#define ENABLE_BF16
#include "common.h"
#include <cmath>

#include <dpct/lib_common_utils.hpp>

// ----------------------------------------------------------------------------
// CUDA & cuDNN setup
static bool first_run_validation = true; // always run e.g. permute on 1st run

#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#else
#define CUDNN_16BIT fe::DataType_t::HALF
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;

#define checkCudaErr(err) assert((int)err == 0);
#define checkCudnnErr(err) assert((int)err == 0);
#endif // ENABLE_CUDNN
// ----------------------------------------------------------------------------
// CPU code reference

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

// ----------------------------------------------------------------------------
// GPU kernels

void attention_query_key_kernel1(float* preatt, const float* inp,
                                           int B, int T, int C, int NH,
                                           const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        int t2 = idx % T;
        int t = (idx / T) % T;
        if (t2 > t) {
            // autoregressive mask
            preatt[idx] = -INFINITY;
            return;
        }
        int h = (idx / (T * T)) % NH;
        int b = idx / (NH * T * T);

        int C3 = C*3;
        int hs = C / NH; // head size
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0 / sycl::sqrt((float)hs);

        preatt[idx] = val;
    }
}

void attention_softmax_kernel1(float* att, const float* preatt,
                                         int B, int T, int NH,
                                         const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // find maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= t; t2++) {
            if (preatt_bth[t2] > maxval) {
                maxval = preatt_bth[t2];
            }
        }

        // calculate the exp and keep track of sum
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = sycl::native::exp(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f;
            }
        }
    }
}

// warp-level reduction for finding the maximum value
SYCL_EXTERNAL float warpReduceMax(float val, const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        /*
        DPCT1121:88: Make sure that the "val" which is used in the SYCL group
        function/algorithm is initialized.
        */
        /*
        DPCT1096:548: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::shift_sub_group_left" may return an unexpected result on the CPU
        device. Modify the size of the work-group to ensure that the value of
        the right-most dimension is a multiple of "32".
        */
        val = sycl::fmax(val, dpct::shift_sub_group_left(
                                  item_ct1.get_sub_group(), val, offset));
    }
    return val;
}

SYCL_EXTERNAL void softmax_forward_kernel4(float *out, const float *inp, int N,
                                           int C,
                                           const sycl::nd_item<3> &item_ct1,
                                           uint8_t *dpct_local) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);
    int tid = item_ct1.get_local_id(2);
    int warpId = item_ct1.get_local_id(2) / 32; // warp index within a block
    int laneId = item_ct1.get_local_id(2) % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = item_ct1.get_local_range(2) / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        maxval = sycl::fmax(maxval, (float)(x[i]));
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval, item_ct1);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = sycl::fmax(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        // subtract max for numerical stability
        out[idx * C + i] = sycl::native::exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval, item_ct1);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    /*
    DPCT1065:431: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    /*
    DPCT1065:432: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += item_ct1.get_local_range(2)) {
        out[idx * C + i] = x[i] / sum;
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
    /*
    DPCT1007:433: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = item_ct1.get_group(2) * warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id();
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


void attention_value_kernel1(float* out, const float* att, const float* inp,
                                       int B, int T, int C, int NH,
                                       const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_bth = out + b * T * C + t * C + h * hs;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
           const  float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}


void attention_forward_kernel2(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O
,
    const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local) {
    int tx = item_ct1.get_local_id(2);
    int bx = item_ct1.get_group(2);
        int by = item_ct1.get_group(1); // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * item_ct1.get_group_range(1) * N * d) +
                     (by * N * d); // gridDim.y = nh
    int lm_offset =
        (bx * item_ct1.get_group_range(1) * N) + (by * N); // offset for l and m

    // Define SRAM for Q,K,V,S
    auto sram = (float *)dpct_local;
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        /*
        DPCT1118:89: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:434: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1
            .barrier(); // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= N) {
                break;
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // implement softmax with causal masking
            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] =
                        sycl::native::exp(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = sycl::max(row_m_prev, row_m);
            float row_l_new =
                (sycl::native::exp(row_m_prev - row_m_new) * row_l_prev) +
                (sycl::native::exp(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= N) {
                        break;
                    }
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] =
                    (1 / row_l_new) *
                    ((row_l_prev * sycl::native::exp(row_m_prev - row_m_new) *
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) +
                     (sycl::native::exp(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        /*
        DPCT1118:90: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:435: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // otherwise, thread can use the wrong Kj, Vj in
                            // inner loop
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

SYCL_EXTERNAL void unpermute_kernel(const float *inp, float *out, int B, int N,
                                    int NH, int d,
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
        out[other_idx] = inp[idx];
    }
}

void scale_kernel(float* inp, float scale, int B, int NH, int T,
                  const sycl::nd_item<3> &item_ct1) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

// direct translation of the CPU kernel. Each warp handles ont (b, h, t) combination.
// The important changes compared to the CPU version:
//  - each inner loop is handled by a warp
//  - don't write non-autoregressive parts
//  - reordered the last loops so that we can do all writing in the outer loop.
/*
DPCT1110:91: The total declared local variable size in device function
attention_forward_fused1 exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void attention_forward_fused1(float *out, float *preatt, float *att,
                              const float *inp, int B, int T, int C, int NH,
                              const sycl::nd_item<3> &item_ct1) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sycl::sqrt((float)hs);

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    /*
    DPCT1007:436: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int t = item_ct1.get_group(2) * warp.meta_group_size() +
            item_ct1.get_sub_group().get_group_linear_id();
    int h = item_ct1.get_group(1);
    int b = item_ct1.get_group(0);

    if(t >= T) return;

    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // pass 1: calculate query dot key and maxval
    float maxval = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < hs;
             i += item_ct1.get_sub_group().get_local_linear_range()) {
            val += query_t[i] * key_t2[i];
        }
        val = sycl::reduce_over_group(item_ct1.get_sub_group(), val,
                                      sycl::plus<float>{});
        val *= scale;
        maxval = sycl::max(maxval, val);
        if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
            preatt_bth[t2] = val;
        }
    }

    // pass 2: calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = item_ct1.get_sub_group().get_local_linear_id(); t2 <= t;
         t2 += item_ct1.get_sub_group().get_local_linear_range()) {
        float expv = sycl::native::exp(preatt_bth[t2] - maxval);
        expsum += expv;
    }

    expsum = sycl::reduce_over_group(item_ct1.get_sub_group(), expsum,
                                     sycl::plus<float>{});

    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // pass 3: normalize to get the softmax is combined with the next loop to reduce memory round-trips
    for (int t2 = item_ct1.get_sub_group().get_local_linear_id(); t2 <= t;
         t2 += item_ct1.get_sub_group().get_local_linear_range()) {
        att_bth[t2] = sycl::native::exp(preatt_bth[t2] - maxval) * expsum_inv;
    }

    // pass 4: accumulate weighted values into the output of attention
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < hs;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float o = 0.f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            o += att_btht2 * value_t2[i];
        }
        out_bth[i] = o;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:92: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            attention_query_key_kernel1(preatt, inp, B, T, C, NH, item_ct1);
        });
    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:93: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            attention_softmax_kernel1(att, preatt, B, T, NH, item_ct1);
        });
    /*
    DPCT1049:94: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            attention_value_kernel1(out, att, inp, B, T, C, NH, item_ct1);
        });
}


void attention_forward2(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // TODO there should be no mallocs inside any of these functions!
    // not fixing this because we don't intend to use attention_forward2,
    // it seems to be way too slow as is

    // these are hardcoded to 32 for now
    const int Bc = 32;
    const int Br = 32;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    const int d = C / NH;
    // more
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);
    // create some temporary memory
    float* l;
    float* m;
    cudaCheck(DPCT_CHECK_ERROR(l = sycl::malloc_device<float>(
                                   B * nh * N, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(m = sycl::malloc_device<float>(
                                   B * nh * N, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memset(l, 0, B * nh * N * sizeof(float))
                                   .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memset(m, -10000.0f, B * nh * N * sizeof(float))
                             .wait()));

    // calculate SRAM size needed per block, ensure we have enough shared memory
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        /*
        DPCT1083:96: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        (2 * col_tile_size * sizeof(float)) // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))   // SRAM size for Qi
        + (Bc * Br * sizeof(float));        // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
        printf("SRAM size exceeds maximum shared memory per block\n");
        printf("Try decreasing col_tile_size or row_tile_size further\n");
        exit(1);
    }

    // grid and block dims
    sycl::range<3> grid_dim(1, nh, B);  // batch_size x num_heads
    sycl::range<3> block_dim(1, 1, Br); // Br threads per block

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    cudaCheck(DPCT_CHECK_ERROR(
        q = sycl::malloc_device<float>(B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        k = sycl::malloc_device<float>(B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        v = sycl::malloc_device<float>(B * T * C, dpct::get_in_order_queue())));
    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:95: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, N, nh, d, item_ct1);
        });

    // now actually call the flash attention kernel
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(sram_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid_dim * block_dim, block_dim),
            [=](sycl::nd_item<3> item_ct1) {
                attention_forward_kernel2(
                    q, k, v, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, out,
                    item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    /*
    DPCT1049:97: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(out, q, B, N, nh, d, item_ct1);
        });
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(out, q, B * T * C * sizeof(float))));

    // free memory
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_free(l, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_free(m, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_free(q, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_free(k, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_free(v, dpct::get_in_order_queue())));
}

void attention_forward3(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
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
    int num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:98: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, T, NH, HS, item_ct1);
        });

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha, k, HS, T * HS, q, HS,
        T * HS, beta, preatt, T, T * T, B * NH)));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:99: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_kernel(preatt, scale, B, NH, T, item_ct1);
        });

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    /*
    DPCT1083:101: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the migrated
    code is correct.
    */
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    /*
    DPCT1049:100: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shared_mem_size), cgh);

        int B_NH_T_ct2 = B * NH * T;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, softmax_block_size),
                              sycl::range<3>(1, 1, softmax_block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel4(
                    att, preatt, B_NH_T_ct2, T, item_ct1,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha, v, HS, T * HS, att,
        T, T * T, beta, vaccum, HS, T * HS, B * NH)));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    /*
    DPCT1049:102: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(vaccum, out, B, T, NH, HS, item_ct1);
        });
}

void attention_forward4(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
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
    int num_blocks = ceil_div(total_threads, block_size);
    /*
    DPCT1049:103: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, T, NH, HS, item_ct1);
        });

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha, k, HS, T * HS, q, HS,
        T * HS, beta, preatt, T, T * T, B * NH)));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    /*
    DPCT1049:104: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
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

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha, v, HS, T * HS, att,
        T, T * T, beta, vaccum, HS, T * HS, B * NH)));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    /*
    DPCT1049:105: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(vaccum, out, B, T, NH, HS, item_ct1);
        });
}


void softmax_forward_kernel5_lowp(floatX* out, float inv_temperature,
                                             const floatX* inp, int N, int T,
                                             const sycl::nd_item<3> &item_ct1) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(0);

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    /*
    DPCT1007:437: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = item_ct1.get_group(2) * warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // Same thing but without float4, one at a time
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < pos_by_4;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, (float)x[4 * i + k]);
        }
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += sycl::native::exp(inv_temperature *
                                        ((float)x[4 * i + k] - maxval));
        }
    }

    if (4 * pos_by_4 + item_ct1.get_sub_group().get_local_linear_id() <=
        own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(
            maxval, (float)x[4 * pos_by_4 +
                             item_ct1.get_sub_group().get_local_linear_id()]);
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::native::exp(
            inv_temperature *
            ((float)x[4 * pos_by_4 +
                      item_ct1.get_sub_group().get_local_linear_id()] -
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
        /*
        DPCT1098:439: The '*' expression is used instead of the __ldcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        float ev = sycl::native::exp(inv_temperature *
                                     ((float)*(x + i) - global_maxval));
        /*
        DPCT1098:438: The '=' expression is used instead of the __stcs call.
        These two expressions do not provide the exact same functionality. Check
        the generated code for potential precision and/or performance issues.
        */
        *(out + idx * T + i) = (floatX)(ev * norm);
    }
}

void permute_kernel_lowp(floatX* q, floatX* k, floatX* v,
                                    const float* inp,
                                    int B, int N, int NH, int d,
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

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = (floatX)inp[inp_idx];
        k[idx] = (floatX)inp[inp_idx + NH * d];
        v[idx] = (floatX)inp[inp_idx + 2 * (NH * d)];
    }
}

void unpermute_kernel_lowp(const floatX* inp, float *out, int B, int N, int NH, int d,
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
        out[other_idx] = (float)inp[idx];
    }
}

void attention_forward5(float* out, floatX* vaccum, floatX* qkvr, floatX* preatt, floatX* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size, bool skip_permute=false) {
    // FP16 version of kernel 4 (with permute/unpermute doing FP32<->FP16)
    // That permute can be skipped on perf runs to analyse its performance impact
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    int HS = C / NH; // head size
    floatX *q = qkvr + 0 * B * T * C;
    floatX *k = qkvr + 1 * B * T * C;
    floatX* v = qkvr + 2 * B * T * C;

    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    if (!skip_permute || first_run_validation) {
        /*
        DPCT1049:107: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                permute_kernel_lowp(q, k, v, inp, B, T, NH, HS, item_ct1);
            });
    }

    // IMPORTANT: alpha/beta are FP32 for CUBLAS_COMPUTE_32F even if FP16 inputs/outputs
    // But need FP16 scale for CUBLAS_COMPUTE_16F (no errors otherwise, just garbage results *sigh*)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const floatX alpha_lowp = (floatX)alpha;
    const floatX beta_lowp = (floatX)beta;
    void* alpha_ptr = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&alpha_lowp : (void*)&alpha;
    void* beta_ptr = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&beta_lowp : (void*)&beta;

    // batched matrix multiply with cuBLAS
    cublasCheck(DPCT_CHECK_ERROR(dpct::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, alpha_ptr, k, CUBLAS_LOWP,
        HS, T * HS, q, CUBLAS_LOWP, HS, T * HS, beta_ptr, preatt, CUBLAS_LOWP,
        T, T * T, B * NH, CUBLAS_LOWP_COMPUTE)));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    /*
    DPCT1049:106: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        int B_NH_ct3 = B * NH;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, softmax_block_size),
                              sycl::range<3>(1, 1, softmax_block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_forward_kernel5_lowp(att, scale, preatt, B_NH_ct3, T,
                                             item_ct1);
            });
    });

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(DPCT_CHECK_ERROR(dpct::gemm_batch(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, alpha_ptr, v, CUBLAS_LOWP,
        HS, T * HS, att, CUBLAS_LOWP, T, T * T, beta_ptr, vaccum, CUBLAS_LOWP,
        HS, T * HS, B * NH, CUBLAS_LOWP_COMPUTE)));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    if(!skip_permute || first_run_validation) {
        /*
        DPCT1049:108: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                unpermute_kernel_lowp(vaccum, out, B, T, NH, HS, item_ct1);
            });
    }
}

#ifdef ENABLE_CUDNN
using graph_tensors_fwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                                     std::shared_ptr<fe::graph::Tensor_attributes>>; // Stats

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;

// Loosely based on cuDNN frontend samples functions and massively simplified
template <typename... Args>
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("attn_scale")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1});

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1});
    }

    assert(graph->validate().is_good());
    auto key = graph->key();
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    assert(graph->build_operation_graph(cudnn_handle).is_good());
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    assert(graph->check_support(cudnn_handle).is_good());
    assert(graph->build_plans(cudnn_handle).is_good());

    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}

// Used on first run only so we can validate against the CPU results
__global__ void fp32_to_lowp_kernel(floatX* out, const float* inp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (floatX)inp[idx];
}

__global__ void lowp_to_fp32_kernel(const floatX* inp, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (float)inp[idx];
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             float* in_fp32,  // fp32 input
                             float* out_fp32, // fp32 output for validation
                             int B, int T, int C, int NH) {
    static bool first_run_validation = true;
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    // Convert from FP32 to FP16/BF16 on 1st run to get correct results
    const int block_size = 64; // smallest full occupancy block size on modern GPUs
    if (first_run_validation) {
        int total_threads = B * T * C * 3;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        fp32_to_lowp_kernel<<<num_blocks, block_size>>>(inp, in_fp32);
    }

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace).is_good());
    cudaCheck(cudaGetLastError());

    // Optionally convert back from FP16/BF16 to FP32
    if (first_run_validation) {
        int total_threads = B * T * C;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        lowp_to_fp32_kernel<<<num_blocks, block_size>>>(out, out_fp32);
    }
    cudaCheck(cudaGetLastError());
    first_run_validation = false;
}

#endif // ENABLE_CUDNN

// kernel version dispatch
void attention_forward(int kernel_num,
                       float* out, float* stats, float* vaccum,
                       float* qkvr, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_forward1(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 2:
            attention_forward2(out, inp, B, T, C, NH, block_size);
            break;
        case 3:
            attention_forward3(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 4:
            attention_forward4(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 5:
            attention_forward5(out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, false);
            break;
        case 6: // skip permutes for perf passes (to analyse perf as if in/out were truly 16-bit)
            attention_forward5(out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, true);
            break;
        #ifdef ENABLE_CUDNN
        case 10:
            // note: validation only cares about out, which is out_fp32 of the function
            // inp is hackily converted to FP16 into qkvr only on the first run
            // similarly, vaccum is converted to FP32 into out only on the first run
            attention_forward_cudnn((floatX*)vaccum, stats, (floatX*)qkvr, inp, out, B, T, C, NH);
            break;
        #endif
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
    /*
    DPCT1093:440: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    dpct::get_device_info(deviceProp,
                          dpct::dev_mgr::instance().get_device(deviceIdx));

    // setup cuBLAS (and cuDNN if needed)
    cublas_handle = new dpct::blas::descriptor();
    /*
    DPCT1005:441: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    int enable_tf32 = deviceProp.get_major_version() >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    int cublas_math_mode =
        enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    /*
    DPCT1027:442: The call to cublasSetMathMode was replaced with 0 because this
    functionality is redundant in SYCL.
    */
    cublasCheck(0);

#ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
    #endif

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_stats; // for cuDNN
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_stats = sycl::malloc_device<float>(
                                   B * NH * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_vaccum = sycl::malloc_device<float>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_qkvr = sycl::malloc_device<float>(
                                   B * T * 3 * C, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(d_preatt = sycl::malloc_device<float>(
                             B * NH * T * T, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(d_att = sycl::malloc_device<float>(
                             B * NH * T * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                                   B * T * 3 * C, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, B * T * 3 * C * sizeof(float))
                             .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = (kernel_num <= 4) ? 1e-3f : 1e-2f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward(kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        // todo - make accuracy threshold dynamic and depend on FP16 vs FP32?
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        // but as for preatt and att, things get a bit more complicated:
        if (kernel_num != 2 && kernel_num < 5) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        }
        if (kernel_num != 2 && kernel_num < 4) {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");
    first_run_validation = false;

    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                              kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att,
                                              d_inp, B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_vaccum, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_qkvr, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_preatt, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_att, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
    delete (cublas_handle);

#ifdef ENABLE_CUDNN
    cudnnDestroy(cudnn_handle);
    if (cudnn_workspace_size > 0) {
        cudaCheck(cudaFree(cudnn_workspace));
    }
    #endif

    return 0;
}