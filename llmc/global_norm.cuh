// TODO - BUGGED - just committing my WIP, not sure why grad norm is zero, probably something silly!

/*
Global norm, used in gralldient clipping
*/
#include <assert.h>
#include <stddef.h>
#include <cuda_runtime_api.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// currently assumes all gradients are the same type (simplified adamw_update_everything)
// ZeRO 1 should use shard_idx, while DPP and ZeRO 2/3 should simply set it to 0
template <typename Tgrad=floatX>
__global__ void __launch_bounds__(256, MAX_WARPS/8) global_norm_tensors_kernel(float* out, int num_params_tensors, unsigned int shard_idx) {
    float grad_norm_accumulator = 0.f;

    constexpr size_t block_size = 256;
    constexpr size_t iteration_size = Packed128<Tgrad>::size;
    size_t idx = (blockIdx.x * block_size * iteration_size) + (threadIdx.x * iteration_size);
    unsigned int stride = gridDim.x * blockDim.x * iteration_size;

    int spec_id = 0;
    TensorSpec* grad_specs   = tensor_specs_ptr + num_params_tensors;
    TensorSpec* opt_v_specs  = tensor_specs_ptr + 3 * num_params_tensors;

    TensorSpec opt_v_spec = opt_v_specs[spec_id];
    size_t current_start = opt_v_spec.element_start_end.x;
    size_t current_end = opt_v_spec.element_start_end.y;

    while (true) {
        while (idx >= current_end) {
            // todo - check performance, misses probably okay if they reduce the tail effect
            // (fastest block/SM "prefetches" for the slower ones)
            // but tiny tensors back-to-back might be inefficient
            spec_id++;
            if (spec_id >= num_params_tensors) {
                break;
            }
            opt_v_spec = opt_v_specs[spec_id];
            current_start = opt_v_spec.element_start_end.x;
            current_end = opt_v_spec.element_start_end.y;
        }
        if (spec_id >= num_params_tensors) {
            break; // goto would avoid this but I don't want to go to hell
        }

        // offset is 32-bit (checked <=4B elements in add_tensor_spec)
        unsigned int offset = (idx - current_start) + (shard_idx * opt_v_spec.num_elements);
        TensorGPU<floatX> grad_tensor  = grad_specs[spec_id];

        __syncthreads(); // todo - hopefully improves memory locality
        while (idx < current_end) {
            auto grad128 = load_tensor128(grad_tensor, offset, false, true);
            for (int k = 0; k < grad_tensor.num_per_128(); k++) {
                float grad = grad128.get(k);
                grad_norm_accumulator += grad * grad;
            }
            idx += stride;
            offset += stride;
        }
    }
    float output = blockReduce<warpReduceSum>(grad_norm_accumulator);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = output;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

template<typename Tgrad=floatX>
void global_norm_tensors(float* out, int gpu_process_rank, cudaStream_t stream=main_stream) {
    const int block_size = 256;
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;

    int num_params_tensors = tensors_start[PARAMETER+1];
    int num_shards_opt = tensor_specs[tensors_start[PARAMETER_OPT_M]].num_shards;
    int num_shards_grad = tensor_specs[tensors_start[PARAMETER_GRAD]].num_shards;
    int num_shards = num_shards_opt / num_shards_grad; // should work for both DPP and ZeRO 1/2/3
    int shard_idx = gpu_process_rank % num_shards;

    global_norm_tensors_kernel<Tgrad><<<grid_size, block_size, 0, stream>>>(out, num_params_tensors, shard_idx);
    global_sum_deterministic(out, out, grid_size, stream);
}