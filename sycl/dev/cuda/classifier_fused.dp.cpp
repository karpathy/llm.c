/*  Kernels for fused forward/backward classifier part
This fuses softmax, crossentropy, and logit gradients into a single pass, so we don't have to write unnecessary
(B, T, V) tensors. Such an operation is only possible if `dloss` can be known beforehand, which doesn't seem like
much of a restriction: In pretraining, it is just a constant 1/batch_size tensor, for fine-tuning we might zero
out the input prompt, but that is known in advance.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt classifier_fused.cu -o classifier_fused

./classifier_fused 1
./classifier_fused 2
./classifier_fused 3
./classifier_fused 4
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "common.h"
#include <cmath>

// todo - this file does not properly support anything but FP32
// kernel 5 can be run in fp16/bf16 to test performance, but the outputs will be wrong
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
#elif defined(ENABLE_FP16)
typedef half floatX;
#else
typedef float floatX;
#endif
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int64_t i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}


void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int64_t bt = 0; bt < B * T; bt++) {
        // loss = -log(probs[target])
        const float* probs_bt = probs + bt * V;
        int ix = targets[bt];
        losses[bt] = -logf(probs_bt[ix]);
    }
}

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int64_t bt = 0; bt < B * T; bt++) {
        float* dlogits_bt = dlogits + bt * V;
        const float* probs_bt = probs + bt * V;
        float dloss = dlosses[bt];
        int ix = targets[bt];
        for (int i = 0; i < V; i++) {
            float p = probs_bt[i];
            float indicator = i == ix ? 1.0f : 0.0f;
            dlogits_bt[i] = (p - indicator) * dloss;
        }
    }
}

// ----------------------------------------------------
// Kernel Utils

// warp-level reduction for finding the maximum value
SYCL_EXTERNAL float warpReduceMax(float val, const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        /*
        DPCT1096:547: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        val = sycl::fmax(val, dpct::permute_sub_group_by_xor(
                                  item_ct1.get_sub_group(), val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

SoftmaxParams prepare_softmax(sycl::sub_group &warp, int64_t idx,
                              const float *inp, int V, int P,
                              const sycl::nd_item<3> &item_ct1) {
    // this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
    // note that inp is actually (B * T, P) but we only use the first V elements
    // this function tehen calculates:
    // 1) the max value to subtract for numerical stability and
    // 2) the sum normalization factor
    const float* x = inp + idx * P;
    // thread coarsening loop, where the 32 threads serially process all V elements
    // thread_rank() is in [0, 31], warp.size() is 32
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < V;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float v = x[i];
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = sycl::fmax(maxval, v);
        sumval *= sycl::native::exp((old_maxval - maxval));
        sumval += sycl::native::exp(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = sycl::reduce_over_group(
        item_ct1.get_sub_group(), maxval, sycl::maximum<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= sycl::native::exp((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                  sumval, sycl::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

void fused_classifier_kernel1(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P,
                             const sycl::nd_item<3> &item_ct1) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    // example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
    // each block of 4 warps is in charge of 4 rows of the input, one warp per row
    // meta_group_size is the number of warps per block (e.g. 4)
    // meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
    /*
    DPCT1007:335: Migration of
    cooperative_groups::thread_block_tile::meta_group_size is not supported.
    */
    int64_t idx = item_ct1.get_group(2) * warp.meta_group_size() +
                  item_ct1.get_sub_group().get_group_linear_id();
    if (idx >= B * T) { // there are B * T rows in the input
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P, item_ct1);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
        int ix = targets[b * T + t];
        float prob =
            sycl::native::exp(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -sycl::log(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    for (int i = item_ct1.get_sub_group().get_local_linear_id(); i < V;
         i += item_ct1.get_sub_group().get_local_linear_range()) {
        float prob =
            sycl::native::exp(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}

SYCL_EXTERNAL float vec_at(const sycl::float4 &vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

SoftmaxParams prepare_softmax_blockwide(sycl::sub_group &warp, int64_t idx,
                                        const float *inp, int V, int P,
                                        const sycl::nd_item<3> &item_ct1,
                                        float *shared_maxval,
                                        float *shared_sumval) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    // float4 to get 128-bit loads and memory level parallelism
    const sycl::float4 *x_vec4 =
        reinterpret_cast<const sycl::float4 *>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, 4) + item_ct1.get_local_id(2) -
                 item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        sycl::float4 v4 = x_vec4[i];
#pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // bounds checking against real V
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, vec_at(v4, k));
            thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::native::exp(vec_at(v4, k) - thread_maxval);
        }
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
    DPCT1065:336: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:337: Consider replacing sycl::nd_item::barrier() with
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

// Fused forward and backward pass for classifier including softmax, and logit gradients
// Writes to both probs (only for debugging) and dlogits (only for training) are optional
// N.B.: We may want to reuse the logits memory for dlogits, so they should *not* be __restrict__!
void fused_classifier_kernel2(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_maxval,
                                         float *shared_sumval) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide(
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
    const sycl::float4 *logits_vec4 =
        reinterpret_cast<const sycl::float4 *>(logits + idx * P);
    for (int i = item_ct1.get_local_id(2); i < ceil_div(V, 4);
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        sycl::float4 v4 = __ldcs(&logits_vec4[i]);

#pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob =
                sycl::native::exp(vec_at(v4, k) - sp.Offset) * sp.Scale;
            prob = (element < V) ? prob : 0.0f; // bounds checking against real V

            // this kernel is DRAM limited so cost of inner branch is ~zero
            if (probs != NULL) {
                probs[idx * P + element] = prob;
            }
            if (dlogits != NULL) {
                float indicator = element == ix ? 1.0f : 0.0f;
                dlogits[idx * P + element] = (prob - indicator) * dloss;
            }
        }
    }
}

SoftmaxParams
prepare_softmax_blockwide_nofloat4(sycl::sub_group &warp, int64_t idx,
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
        thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
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
    DPCT1065:338: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:339: Consider replacing sycl::nd_item::barrier() with
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

// same as 2 but not using float4
void fused_classifier_kernel3(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_maxval,
                                         float *shared_sumval) {

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();
    int64_t idx = item_ct1.get_group(2);
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
        if (dlogits != NULL) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

SoftmaxParams prepare_softmax_blockwide2(int64_t idx, const floatX* inp, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, x128::size) + item_ct1.get_local_id(2) -
                 item_ct1.get_local_range(2);
         i >= 0; i -= item_ct1.get_local_range(2)) {
        x128 packed_x = load128cs(x + i * x128::size); // load and do not keep in cache
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {  // bounds checking against real V
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp(old_maxval - thread_maxval);
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
    }
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce

    int num_warps = item_ct1.get_local_range(2) / 32;
    int warp_id = item_ct1.get_local_id(2) / 32;
    int lane_id = item_ct1.get_local_id(2) % 32;

    // reduce maxval within each warp
    float warp_maxval = warpReduceMax(thread_maxval, item_ct1);
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    /*
    DPCT1065:340: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = warpReduceMax(warp_maxval, item_ct1);
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = warpReduceSum(
        thread_sumval,
        item_ct1); // cg::reduce(warp, thread_sumval, cg::plus<float>{});

    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    /*
    DPCT1065:341: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = warpReduceSum(
        warp_sumval,
        item_ct1); // cg::reduce(warp, thread_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but using x128
void fused_classifier_kernel4(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_maxval,
                                         float *shared_sumval) {
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide2(idx, logits, V, P, item_ct1,
                                                  shared_maxval, shared_sumval);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp((float)logits[idx * P + ix] - sp.Offset) *
            sp.Scale;
        losses[idx] = -sycl::log(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < ceil_div(V, x128::size);
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        x128 packed_logits_vec = load128cs(logits_vec + i * x128::size); // load and do not keep in cache
        x128 packed_probs;
        x128 packed_dlogits;
        for(int k = 0; k < packed_logits_vec.size; ++k) {
            int element = i*packed_logits_vec.size + k;
            if (element >= V) {  // bounds checking against real V
                continue;
            }
            float v = packed_logits_vec[k];
            float prob = sycl::native::exp(v - sp.Offset) * sp.Scale;
            packed_probs[k] = prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_dlogits[k] = (prob - indicator) * dloss;
        }
        // Note: missing .cs hint hurts our performance due to cache thrashing, fixed in kernel5
        store128(dlogits + idx * P + i * packed_logits_vec.size, packed_dlogits);
        if (probs != NULL) {
            store128(probs + idx * P + i * packed_logits_vec.size, packed_probs);
        }
    }
}

// todo - move to common.h - or ideally somewhere it's not duplicated between train & common?
// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float);
template <reduction_func_t warp_reduction>
SYCL_EXTERNAL float blockReduce(float val, const sycl::nd_item<3> &item_ct1,
                                float *shared_val, bool final_sync = false,
                                float out_of_bounds = 0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)

    const int lane_id = item_ct1.get_local_id(2) % 32;
    const int warp_id = item_ct1.get_local_id(2) / 32;
    const int num_warps = item_ct1.get_local_range(2) / 32;

    float warp_val = warp_reduction(val, item_ct1);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    /*
    DPCT1065:342: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val, item_ct1);

    if (final_sync) {
        /*
        DPCT1065:343: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // only needed in loops when effectively reusing
                            // shared memory etc.
    }
    return block_val;
}

SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_val) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V + x128::size - 1) / x128::size + item_ct1.get_local_id(2) -
            item_ct1.get_local_range(2);

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
        i -= item_ct1.get_local_range(2);
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= item_ct1.get_local_range(2)) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = sycl::fmax(thread_maxval, v);
            thread_sumval *= sycl::native::exp((old_maxval - thread_maxval));
            thread_sumval += sycl::native::exp(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(
        thread_maxval, item_ct1, shared_val, false, -FLT_MAX);
    thread_sumval *= sycl::native::exp(thread_maxval - block_maxval);
    float block_sumval =
        blockReduce<warpReduceSum>(thread_sumval, item_ct1, shared_val);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
void 
                fused_classifier_kernel5(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P,
                                         const sycl::nd_item<3> &item_ct1,
                                         float *shared_val) {
    int64_t idx = item_ct1.get_group(2);
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp =
        prepare_softmax_blockwide3(idx, logits, V, P, item_ct1, shared_val);

    // calculate the probability needed for the loss and update (single-threaded)
    if (item_ct1.get_local_id(2) == 0) {
        float prob =
            sycl::native::exp((float)logits[idx * P + ix] - sp.Offset) *
            sp.Scale;
        losses[idx] = (floatX)(-sycl::log(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = item_ct1.get_local_id(2); i < V / x128::size;
         i += item_ct1.get_local_range(2)) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob =
                sycl::native::exp((float)packed_logits_vec[k] - sp.Offset) *
                sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(dlogits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = item_ct1.get_local_id(2) + unaligned_start; i < V; i++) {
        float prob =
            sycl::native::exp((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteLogits){
            __stcs(dlogits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void fused_classifier1(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T; // total number of rows in the input
    // how many rows of the input can each block of threads process?
    // e.g. in block_size=128, 4 rows get handled by 4 warps (of 32 threads each)
    const int rows_per_block = block_size / 32;
    const int grid_size = N / rows_per_block; // total number of blocks needed
    /*
    DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            fused_classifier_kernel1(dlogits, losses, logits, dlosses, targets,
                                     B, T, V, P, item_ct1);
        });
    /*
    DPCT1010:344: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier2(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
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
                fused_classifier_kernel2(
                    dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P,
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
    DPCT1010:345: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier3(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
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
                    dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P,
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
    DPCT1010:346: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier4(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:35: The work-group size passed to the SYCL kernel may exceed the
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
                fused_classifier_kernel4(
                    (floatX *)dlogits, (floatX *)losses, NULL, (floatX *)logits,
                    (floatX *)dlosses, targets, B, T, V, P, item_ct1,
                    shared_maxval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    shared_sumval_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:347: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier5(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    /*
    DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:570: 'WARP_SIZE' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                fused_classifier_kernel5<true, false>(
                    (floatX *)dlogits, (floatX *)losses, NULL, (floatX *)logits,
                    (floatX *)dlosses, targets, B, T, V, P, item_ct1,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:348: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void fused_classifier(int kernel_num, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 2:
            fused_classifier2(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 3:
            fused_classifier3(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 4:
            fused_classifier4(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        case 5:
            fused_classifier5(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int64_t B = 8;              // batch size
    int64_t T = 1024;           // sequence length
    int64_t V = 50257;          // vocab size
    int64_t P = (V + 63) & ~63; // padded vocab size, up to nearest multiple of 64

    int deviceIdx = 0;
    /*
    DPCT1093:349: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));

    // create host memory of random numbers
    float* logits = make_random_float_01(B * T * V);
    float* probs = (float*)malloc(B * T * V * sizeof(float));
    float* dlogits = (float*)malloc(B * T * V * sizeof(float));
    float* losses = (float*)malloc(B * T * sizeof(float));
    float* dlosses = make_random_float(B * T);
    int* targets = make_random_int(B * T, V);
    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            logits[j * V +  outliers[j*3 + k]] *= 20;
        }
    }

    // move to GPU
    int *d_targets;
    float *d_logits, *d_losses;
    float *d_dlogits, *d_dlosses, *d_dlogits_no_pad;
    cudaCheck(DPCT_CHECK_ERROR(d_dlogits = sycl::malloc_device<float>(
                                   B * T * P, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_logits = sycl::malloc_device<float>(
                                   B * T * P, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dlogits_no_pad = sycl::malloc_device<float>(
                                   B * T * V, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_targets = sycl::malloc_device<int>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_losses = sycl::malloc_device<float>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dlosses = sycl::malloc_device<float>(
                                   B * T, dpct::get_in_order_queue())));

    // move to GPU
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memset(d_logits, 0xff, B * T * P * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_memcpy(
        d_logits, P * sizeof(float), logits, V * sizeof(float),
        V * sizeof(float), B * T, dpct::host_to_device)));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dlosses, dlosses, B * T * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_targets, targets, B * T * sizeof(int))
                             .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // define block sizes we'll use in correctness and timing
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    // first check the correctness of the kernel
    softmax_forward_cpu(probs, logits, B * T, V);
    crossentropy_forward_cpu(losses, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
    if (kernel_num < 4) // kernel 4/5 + BF16 is only for testing performance, it doesn't do the format conversions yet etc...
#endif
    {
        // time the kernel at different block sizes
        for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
            int block_size = block_sizes[j];
            printf("Checking block size %d.\n", block_size);
            fused_classifier(kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, P, block_size);
            validate_result(d_losses, losses, "losses", B * T, 1e-4f);
            // undo the padding before we can check for correctness
            cudaCheck(DPCT_CHECK_ERROR(dpct::dpct_memcpy(
                d_dlogits_no_pad, V * sizeof(float), d_dlogits,
                P * sizeof(float), V * sizeof(float), B * T,
                dpct::device_to_device)));
            validate_result(d_dlogits_no_pad, dlogits, "dlogits", B * T * V, 1e-4f);
        }
        printf("All results match. Starting benchmarks.\n\n");
    }

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_classifier,
                                              kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets,
                                              B, T, V, P, block_size);
        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(logits);
    free(probs);
    free(dlogits);
    free(losses);
    free(dlosses);
    free(targets);
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_dlogits, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_losses, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_logits, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_dlosses, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_targets, dpct::get_in_order_queue())));

    return 0;
}