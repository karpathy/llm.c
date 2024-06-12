/*
Attention, as a fallback when we do not use the Flash Attention from cuDNN
*/
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
//#include "cublas_common.h"
#include <dpct/lib_common_utils.hpp>
#include <dpct/blas_utils.hpp>

#include <cmath>

typedef float floatX;

//typedef sycl::ext::oneapi::bfloat16 floatX;

// ----------------------------------------------------------------------------
// CUDA kernels

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d,
                               const sycl::nd_item<3> &item_ct1) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;

    q[idx] = inp[inp_idx];
    
    k[idx] = inp[inp_idx + NH * d];
    
    v[idx] = inp[inp_idx + 2 * (NH * d)];
}


void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d,
                                        const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx >= B * NH * N * d) { return; }

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

void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d,
                      const sycl::nd_item<3> &item_ct1) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2));
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    
    out[other_idx] = inp[idx];
}

void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d,
                               const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T,
                             const sycl::nd_item<3> &item_ct1) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(0);
    int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
    int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    int num_warps = item_ct1.get_local_range(2) / WARP_SIZE;

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx =
        (item_ct1.get_group_range(2) - item_ct1.get_group(2) - 1) * num_warps +
        warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = sycl::fmax(maxval, regarray[k]);
        }
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval +=
                sycl::native::exp(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = sycl::fmax(maxval, (float)x[4 * pos_by_4 + lane_id]);
        sumval *= sycl::native::exp(inv_temperature * (old_maxval - maxval));
        sumval += sycl::native::exp(
            inv_temperature * ((float)x[4 * pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval, item_ct1);
    sumval *= sycl::native::exp(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(sumval, item_ct1);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // recalculation is faster than doing the round-trip through memory.
        
        float ev = sycl::native::exp(inv_temperature *
                                     ((float)*(x + i) - global_maxval));

        *(out + idx * T + i) = (floatX)(ev * norm);
    }
}

void softmax_autoregressive_backward_kernel(floatX* dpreatt, const floatX* datt, const floatX* att,
                                                       int B, int T, int C, float scale,
                                                       const sycl::nd_item<3> &item_ct1,
                                                       float *shared_val) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block * item_ct1.get_group(2);
    int idx = item_ct1.get_group(1);

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = item_ct1.get_local_id(2); t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = blockReduce<warpReduceSum>(local_sum, item_ct1, shared_val);

        for (int t3 = item_ct1.get_local_id(2); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            
            float acc =
                (float)*(att_bth + t3) * ((float)*(datt_bth + t3) - local_sum);
            
            *(dpreatt_bth + t3) = (floatX)(scale * acc);
        }
    }
}
// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, sycl::queue &q_ct1) {
    
try{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const float alpha = 1.0f, beta = 0.0f;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    q_ct1.submit(
		[&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel(q, k, v, inp, B, T, NH, HS, item_ct1);
        });
      });

    floatX* preatt = inp;
    
    dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, &alpha, k, CUBLAS_LOWP, HS,
        T * HS, q, CUBLAS_LOWP, HS, T * HS, &beta, preatt, CUBLAS_LOWP, T,
        T * T, B * NH, cublas_compute);
     
    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    q_ct1.submit([&](sycl::handler &cgh) {
        int B_NH_ct3 = B * NH;
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                               sycl::range<3>(1, 1, block_size),
                                           sycl::range<3>(1, 1, block_size)),
                         [=](sycl::nd_item<3> item_ct1)
                             [[intel::reqd_sub_group_size(32)]] {
                                 softmax_forward_kernel5(att, scale, preatt,
                                                         B_NH_ct3, T, item_ct1);
                             });
    });

    // new approach: first BLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    
    dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, &alpha, v, CUBLAS_LOWP, HS,
        T * HS, att, CUBLAS_LOWP, T, T * T, &beta, vaccum, CUBLAS_LOWP, HS,
        T * HS, B * NH, cublas_compute);
     
    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    q_ct1.submit(
		[&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel(vaccum, out, B, T, NH, HS, item_ct1);
        });
      });
    }
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
    

}


// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* dpreatt, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, sycl::queue &q_ct1) {
try{
    const int block_size = 256;
    int HS = C / NH; // head size
    const float alpha = 1.0f, beta = 0.0f;

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    q_ct1.submit(
		[&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            unpermute_kernel_backward(scratch, dout, B, T, NH, HS, item_ct1);
        });
    });
    // backward into datt
    
    dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, T, T, HS, &alpha, v, CUBLAS_LOWP, HS,
        T * HS, scratch, CUBLAS_LOWP, HS, T * HS, &beta, datt, CUBLAS_LOWP, T,
        T * T, B * NH, cublas_compute);
    // backward into dv
    dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, &alpha, scratch, CUBLAS_LOWP,
        HS, T * HS, att, CUBLAS_LOWP, T, T * T, &beta, dv, CUBLAS_LOWP, HS,
        T * HS, B * NH, cublas_compute);
    
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, B * NH, T / 4) *
                                  sycl::range<3>(1, 1, 256),
                              sycl::range<3>(1, 1, 256)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_autoregressive_backward_kernel(
                    dpreatt, datt, att, B, T, C, scale, item_ct1,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    
    // backward into q
    
   dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, HS, T, T, &alpha, k, CUBLAS_LOWP, HS,
        T * HS, dpreatt, CUBLAS_LOWP, T, T * T, &beta, dq, CUBLAS_LOWP, HS,
        T * HS, B * NH, cublas_compute);
    // backward into k
    dpct::gemm_batch(
        q_ct1, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, HS, T, T, &alpha, q, CUBLAS_LOWP, HS,
        T * HS, dpreatt, CUBLAS_LOWP, T, T * T, &beta, dk, CUBLAS_LOWP, HS,
        T * HS, B * NH, cublas_compute);
    // backward into inp
    
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            permute_kernel_backward(dinp, dq, dk, dv, B, T, NH, HS, item_ct1);
        });
      });
    }
    
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}
int main(){

    
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    sycl::context ctx = q_ct1.get_context();
    
    int B = 8;
    int T = 1024;
    int C = 768;
    size_t L = 2;
    size_t NH = 4;
    //int OC = 768 * 4; // expansion of 4, e.g., in the MLP

    // set up the device
    std::cout << "Device: " << q_ct1.get_device().get_info<sycl::info::device::name>() << std::endl;

    // create host memory of random numbers
    float* dinp = make_zeros_floatX(B * T * 3*C);
    float* qkvr = make_zeros_floatX(B * T * 3*C);
    float* preatt = make_zeros_floatX(B * NH * T *T);
    float* att = make_zeros_floatX(B * NH * T *T);
    float* dout = make_random_floatX(B * T * C);
    float* scratch = make_zeros_floatX(B * T * C);
    
    // move to GPU
    float* d_dinp = sycl::malloc_device<float>(B * T * 3*C, q_ct1);
    float* d_qkvr = sycl::malloc_device<float>(B * T * 3*C, q_ct1);
    float* d_preatt = sycl::malloc_device<float>(B * NH * T * T, q_ct1);
    float* d_att = sycl::malloc_device<float>(B * NH * T * T, q_ct1);
    float* d_dout = sycl::malloc_device<float>(B * T * C, q_ct1);
    float* d_scratch = sycl::malloc_device<float>(B * T * C, q_ct1);
    
    
    q_ct1.memcpy(d_dinp, dinp, B * T * 3*C * sizeof(float)).wait();
    q_ct1.memcpy(d_qkvr, qkvr, B * T * 3*C * sizeof(float)).wait();
    q_ct1.memcpy(d_preatt, preatt, B * NH * T * T * sizeof(float)).wait();
    q_ct1.memcpy(d_att, att, B * NH * T * T * sizeof(float)).wait();
    q_ct1.memcpy(d_dout, dout, B * T * C * sizeof(float)).wait();
    q_ct1.memcpy(d_scratch, scratch, B * T * C * sizeof(float)).wait();
  
    attention_forward(d_dout, d_qkvr, d_att,
                       d_dinp,
                       B, T, C, NH, q_ct1);
    attention_backward(d_dinp, d_qkvr, d_preatt, d_att, d_scratch,
                         d_dout,
                        qkvr, att,
                        B, T, C, NH, q_ct1);

    validate_result(d_dinp, dinp, "master_params_memory", B * T * 3*C * sizeof(float));
    validate_result(d_qkvr, qkvr, "params_memory", B * T * 3*C * sizeof(float));
    validate_result(d_preatt, att, "grads_memory", B * NH * T * T * sizeof(float));
    validate_result(d_att, att, "grads_memory", B * NH * T * T * sizeof(float));
    validate_result(d_dout, dout, "m_memory", B * T * C * sizeof(float));
    validate_result(d_scratch, scratch, "v_memory", B * T * C * sizeof(float));
    
    // Free device memory
    sycl::free(d_dinp, q_ct1);
    sycl::free(d_qkvr, q_ct1);
    sycl::free(d_preatt, q_ct1);
    sycl::free(d_att, q_ct1);
    sycl::free(d_dout, q_ct1);
    sycl::free(d_scratch, q_ct1);

    // cleanup
    

  return 0;

}