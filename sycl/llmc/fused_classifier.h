/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/
// llmc internal imports
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

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
        thread_maxval, item_ct1, shared_val, false, -INFINITY);
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
    fused_classifier_kernel5(floatX* logits, floatX* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P,
                                const sycl::nd_item<3> &item_ct1,
                                float *shared_val) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = item_ct1.get_group_range(2) -
                  (item_ct1.get_group(2) +
                   1); // reverse order for cache hits on matmul data
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
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
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
            
            *(logits + idx * P + i) = (floatX)dlogit;
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type>
void fused_classifier(Type* logits, Type* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, sycl::queue &q) {
    
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    
    q.submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                fused_classifier_kernel5(
                    logits, losses, (floatX *)NULL, dloss, targets, B, T, V, P,
                    item_ct1,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    
}
