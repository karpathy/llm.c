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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

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
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};
namespace cg = cooperative_groups;
__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp,
                                         int64_t idx, const float* inp, int V, int P) {
    // this warp (of 32) threads processes one row of inp, i.e. inp[idx, :] of shape (V,)
    // note that inp is actually (B * T, P) but we only use the first V elements
    // this function then calculates:
    // 1) the max value to subtract for numerical stability and
    // 2) the sum normalization factor
    const float* x = inp + idx * P;
    // thread coarsening loop, where the 32 threads serially process all V elements
    // thread_rank() is in [0, 31], warp.size() is 32
    float maxval = -INFINITY;
    float sumval = 0.0f;
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float v = x[i];
        float old_maxval = maxval;
        // online softmax recurrence from "Online normalizer calculation for softmax" paper
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }
    // warp-level reduction to get the maxval across the 32 threads
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    // all 32 threads do a final shift of the sum considering the global max in this row
    sumval *= expf((maxval - global_maxval));
    // warp-level reduction to get the sumval across the 32 threads
    float global_sumval = cg::reduce(warp, sumval, cg::plus<float>{});
    // the final normalization factor
    float norm = 1.0f / global_sumval;
    return SoftmaxParams{norm, global_maxval};
}

__global__ void fused_classifier_kernel1(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // example: B = 4, T = 1024, block_size = 128 => we'd have grid_size = 1024
    // each block of 4 warps is in charge of 4 rows of the input, one warp per row
    // meta_group_size is the number of warps per block (e.g. 4)
    // meta_group_rank is the index of the warp in the block (e.g. 0, 1, 2, 3)
    int64_t idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) { // there are B * T rows in the input
        return;
    }
    int b = idx / T;
    int t = idx % T;

    // calculate the offset (maxval) and scale (sumval) for the softmax
    SoftmaxParams sp = prepare_softmax(warp, idx, logits, V, P);

    // in each row (handled by one warp), thread 0 calculates the loss
    // calculate the probability needed for the loss and update losses
    if(warp.thread_rank() == 0) {
        int ix = targets[b * T + t];
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -logf(prob);
    }

    // finally all threads calculate the gradients
    // prob is only materialized here temporarily and in registers, never
    // as a full tensor that gets written to global memory
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float prob = expf(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (prob - indicator) * dloss;
    }
}


__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__device__ SoftmaxParams prepare_softmax_blockwide(cg::thread_block_tile<32>& warp,
                                                   int64_t idx, const float* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    // float4 to get 128-bit loads and memory level parallelism
    const float4* x_vec4 = reinterpret_cast<const float4*>(inp + idx * P);

    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, 4) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float4 v4 = x_vec4[i];
        #pragma unroll
        for(int k = 0; k < 4; k++) {
            if (i*4+k >= V) {  // bounds checking against real V
                continue;
            }
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, vec_at(v4, k));
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(vec_at(v4, k) - thread_maxval);
        }
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// Fused forward and backward pass for classifier including softmax, and logit gradients
// Writes to both probs (only for debugging) and dlogits (only for training) are optional
// N.B.: We may want to reuse the logits memory for dlogits, so they should *not* be __restrict__!
__global__ void fused_classifier_kernel2(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float4* logits_vec4 = reinterpret_cast<const float4*>(logits + idx * P);
    for (int i = threadIdx.x; i < ceil_div(V, 4); i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float4 v4 = __ldcs(&logits_vec4[i]);

        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            int element = i*4 + k;
            float prob = expf(vec_at(v4, k) - sp.Offset) * sp.Scale;
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

__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32>& warp,
                                                            int64_t idx, const float* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf(old_maxval - thread_maxval);
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4
__global__ void fused_classifier_kernel3(float* dlogits, float* losses, float* probs,
                                         const float* logits, const float* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL) {
            probs[idx * P + i] = prob;
        }
        if (dlogits != NULL) {
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[idx * P + i] = (prob - indicator) * dloss;
        }
    }
}

__device__ SoftmaxParams prepare_softmax_blockwide2(int64_t idx, const floatX* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = ceil_div(V, x128::size) + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128cs(x + i * x128::size); // load and do not keep in cache
        for(int k = 0; k < packed_x.size; ++k) {
            if (i*x128::size+k >= V) {  // bounds checking against real V
                continue;
            }
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf(old_maxval - thread_maxval);
            thread_sumval += expf(v - thread_maxval);
        }
    }
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = warpReduceMax(thread_maxval);
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0) { shared_maxval[warp_id] = warp_maxval; }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = warpReduceMax(warp_maxval);
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = warpReduceSum(thread_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});

    if (lane_id == 0) { shared_sumval[warp_id] = warp_sumval; }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = warpReduceSum(warp_sumval); //cg::reduce(warp, thread_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but using x128
__global__ void fused_classifier_kernel4(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide2(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < ceil_div(V , x128::size); i += blockDim.x) {
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
            float prob = expf(v - sp.Offset) * sp.Scale;
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

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -FLT_MAX);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
                fused_classifier_kernel5(floatX* dlogits, floatX* losses, floatX* probs,
                                         const floatX* logits, const floatX* dlosses, const int* targets,
                                         int B, int T, int V, int P) {
    int64_t idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = (floatX)(-logf(prob));
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = (dlosses != NULL) ? (float)dlosses[idx] : 1.0f / (B*T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
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
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
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
    fused_classifier_kernel1<<<grid_size, block_size>>>(dlogits, losses, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier2(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel2<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier3(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(dlogits, losses, NULL, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier4(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel4<<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier5(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel5<true,false><<<grid_size, block_size>>>((floatX*)dlogits, (floatX*)losses, NULL, (floatX*)logits, (floatX*)dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
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
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* logits = make_random_float(B * T * V);
    float* probs = make_random_float_01(B * T * V);
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
    cudaCheck(cudaMalloc(&d_dlogits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_logits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits_no_pad, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_losses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));

    // move to GPU
    cudaCheck(cudaMemset(d_logits, 0xff, B * T * P * sizeof(float)));
    cudaCheck(cudaMemcpy2D(d_logits, P * sizeof(float), logits, V * sizeof(float), V * sizeof(float), B * T, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

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
            cudaCheck(cudaMemcpy2D(d_dlogits_no_pad, V * sizeof(float), d_dlogits, P * sizeof(float), V * sizeof(float), B * T, cudaMemcpyDeviceToDevice));
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
    free(outliers);
    cudaCheck(cudaFree(d_dlogits));
    cudaCheck(cudaFree(d_losses));
    cudaCheck(cudaFree(d_logits));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_targets));

    return 0;
}