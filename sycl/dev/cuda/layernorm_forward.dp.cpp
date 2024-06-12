/*
Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
          (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
./layernorm_forward 5
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include <cmath>

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
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

// naive drag and drop implementation into kernel, parallelize over B,T, loop over C
void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 const float* inp, const float* weight, const float* bias,
                                 int N, int C, const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sycl::sqrt(v + eps);
        // seek to the output position in out[idx,:]
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[idx] = m;
        rstd[idx] = s;
    }
}

void mean_kernel(float* mean, const float* inp, int N, int C, int block_size,
                 const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);    // range [0, B*T)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    const float* x = inp + idx * C;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:174: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size,
                 const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
    auto shared = (float *)dpct_local;
    int idx = item_ct1.get_group(2);    // range [0, B*T)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:175: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sycl::sqrt(shared[0] / C + 1e-5f);
    }
}

void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C,
                                     const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

SYCL_EXTERNAL void layernorm_forward_kernel3(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C,
    const sycl::nd_item<3> &item_ct1) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    /*
    DPCT1007:519: Migration of
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

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C,
                                    const sycl::nd_item<3> &item_ct1) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    /*
    DPCT1007:520: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int idx = item_ct1.get_group(2) * warp.meta_group_size() +
              item_ct1.get_sub_group().get_group_linear_id();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // thread coarsening through the row, reduce the sum in series
    float sum = 0.0; // stores sum(x)
    float sum2 = 0.0; // stores sum(x**2)
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < C;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;
    }
    // warp-level reduction at the end
    sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sum,
                                  sycl::plus<float>{}); // sum(x)
    sum2 = sycl::reduce_over_group(item_ct1.get_sub_group(), sum2,
                                   sycl::plus<float>{}); // sum(x**2)
    sum /= C; // mean(x)
    sum2 /= C; // mean(x**2)

    // mean, var, rstd
    float m = sum;
    float var = sum2 - sum * sum;
    float s = sycl::rsqrt(var + 1e-5f);

    // store the mean, no need to cache it
    if (item_ct1.get_sub_group().get_local_linear_id() == 0 &&
        mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if (item_ct1.get_sub_group().get_local_linear_id() == 0 &&
        rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = item_ct1.get_sub_group().get_local_linear_id(); c < C;
         c += item_ct1.get_sub_group().get_local_linear_range()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single warp
void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C,
                                    const sycl::nd_item<3> &item_ct1,
                                    float *shared_sum, float *shared_sum2) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
     // block_size max is 1024 = 32 * 32 warps
     // warps will be writing into shared memeory after warp-reduce
    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;
    int idx = item_ct1.get_group(2); // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum = 0.0; // stores sum(x)
    float thread_sum2 = 0.0; // stores sum(x**2)
    // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum = sycl::reduce_over_group(
        item_ct1.get_sub_group(), thread_sum, sycl::plus<float>{}); // sum(x)
    float warp_sum2 =
        sycl::reduce_over_group(item_ct1.get_sub_group(), thread_sum2,
                                sycl::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum = sycl::reduce_over_group(
        item_ct1.get_sub_group(), warp_sum, sycl::plus<float>{}); // sum(x)
    float block_sum2 = sycl::reduce_over_group(
        item_ct1.get_sub_group(), warp_sum2, sycl::plus<float>{}); // sum(x**2)
    // mean, var, rstd
    block_sum /= C; // mean(x)
    block_sum2 /= C; // mean(x**2)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = sycl::rsqrt(var + 1e-5f);
    // store the mean, no need to cache it
    if (item_ct1.get_local_id(2) == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if (item_ct1.get_local_id(2) == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = item_ct1.get_local_id(2); i < C;
         i += item_ct1.get_local_range(2)) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:176: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            layernorm_forward_kernel1(out, mean, rstd, inp, weight, bias, N, C,
                                      item_ct1);
        });
    /*
    DPCT1010:521: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    /*
    DPCT1049:177: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:552: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(block_size * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, B * T) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                mean_kernel(mean, inp, N, C, block_size, item_ct1,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    /*
    DPCT1010:522: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
    /*
    DPCT1049:178: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:553: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(block_size * sizeof(float)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, B * T) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                rstd_kernel(rstd, inp, mean, N, C, block_size, item_ct1,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
            });
    });
    /*
    DPCT1010:523: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size2),
                          sycl::range<3>(1, 1, block_size2)),
        [=](sycl::nd_item<3> item_ct1) {
            normalization_kernel(out, inp, mean, rstd, weight, bias, B, T, C,
                                 item_ct1);
        });
    /*
    DPCT1010:524: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward3(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:179: The work-group size passed to the SYCL kernel may exceed the
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
    DPCT1010:525: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    /*
    DPCT1049:180: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            layernorm_forward_kernel4(out, mean, rstd, inp, weight, bias, N, C,
                                      item_ct1);
        });
    /*
    DPCT1010:526: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:181: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_sum_acc_ct1(sycl::range<1>(32),
                                                          cgh);
        sycl::local_accessor<float, 1> shared_sum2_acc_ct1(sycl::range<1>(32),
                                                           cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                layernorm_forward_kernel5(
                    out, mean, rstd, inp, weight, bias, N, C, item_ct1,
                    shared_sum_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sum2_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:527: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    /*
    DPCT1093:528: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_mean = sycl::malloc_device<float>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_rstd = sycl::malloc_device<float>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<float>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        d_weight = sycl::malloc_device<float>(C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        d_bias = sycl::malloc_device<float>(C, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, B * T * C * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_weight, weight, C * sizeof(float))
                                   .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_bias, bias, C * sizeof(float))
                                   .wait()));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    float* out_gpu = (float*)malloc(B * T * C * sizeof(float));
    float* mean_gpu = (float*)malloc(B * T * sizeof(float));
    float* rstd_gpu = (float*)malloc(B * T * sizeof(float));

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                              kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_mean, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_rstd, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_weight, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_bias, dpct::get_in_order_queue())));

    return 0;
}