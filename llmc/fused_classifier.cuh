/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, tensorX inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    int elements = inp.num_per_128();
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+elements-1)/elements + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*elements > V) {
        for(int k = 0; k < elements; ++k) {
            if (i*elements+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = inp.get_scalar(idx * P + i * elements + k);
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        auto inp128 = load_tensor128(inp, idx * P + i * elements);
        for(int k = 0; k < elements; ++k) {
            float v = inp128.get(k);
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(tensorX dlogits, tensorX logits, float* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int V, int P, std::bool_constant<WriteDLogits>) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it size_t here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int elements = logits.num_per_128();
    size_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf(logits.get_scalar(idx * P + ix) - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    // since the "logits" are overwritten to be in the [-1, 1] range and sp.Offset is sometimes smaller than -90
    // we errouneously end up computing exp^(90+) which gives us infinities in the loss! this is the fix.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    tensor128<floatX> dlogits128 = new_tensor128<WriteDLogits>(dlogits, true);
    for (int i = threadIdx.x; i < V/elements; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        auto logits128 = load_tensor128(logits, idx * P + i * elements, false, true);
        x128 packed_probs; // todo - unused but might be read on CPU in the future so not scaling (???)
        for(int k = 0; k < elements; ++k) {
            int element = i*elements + k;
            float prob = expf(logits128.get(k) - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            dlogits128.set(k, (prob - indicator) * dloss);
        }
        if constexpr (WriteDLogits) {
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            dlogits128.store(idx * P + i * elements, true);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * elements, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of the number of elements
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(elements - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf(logits.get_scalar(idx * P + i) - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            floatX dlogitX = dlogits.set_scalar(idx * P + i, dlogit); // write to memory
            dlogits128.add_value_stats(dlogit, dlogitX); // add to absmax stats etc.
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
    if constexpr (WriteDLogits) {
        dlogits128.update_absmax(threadIdx.x, blockDim.x, true);
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <bool WriteDLogits>
void fused_classifier(tensorX dlogits, tensorX logits, tensorFP32 losses,
                      const float dloss, const int* targets,
                      int BT, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;
    const int grid_size = BT;
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(dlogits, logits, losses, (floatX*)NULL, dloss, targets, V, P, write_dlogits);
    cudaCheck(cudaGetLastError());
}
