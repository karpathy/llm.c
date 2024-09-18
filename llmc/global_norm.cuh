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
__global__ void __launch_bounds__(256, MAX_THREADS/256) global_norm_tensors_kernel(float* out, int num_params_tensors, unsigned int shard_idx) {
    float grad_norm_accumulator = 0.f;

    constexpr size_t block_size = 256;
    constexpr size_t iteration_size = Packed128<Tgrad>::size;
    size_t idx = (blockIdx.x * block_size + threadIdx.x) * iteration_size;
    unsigned int stride = gridDim.x * blockDim.x * iteration_size;

    int opt_m_spec_id = 2 * num_params_tensors; // opt_m is sharded with ZeRO 1/2/3
    int last_opt_m_id = 3 * num_params_tensors - 1;

    size_t current_end = tensor_end_element_ptr[opt_m_spec_id];

    while (true) {
        while (idx >= current_end) {
            opt_m_spec_id++;
            if (opt_m_spec_id > last_opt_m_id) break;

            #if __CUDA_ARCH__ < 800
            current_end = tensor_end_element_ptr[opt_m_spec_id];
            #else
            // on A100+ we can prefetch 256B (32 end values) into the L2
            asm("ld.global.L1::evict_last.L2::256B.u64 {%0}, [%1];" : "=l"(current_end) : "l"(tensor_end_element_ptr + opt_m_spec_id));
            #endif
        }
        if (opt_m_spec_id > last_opt_m_id) break;

        // offset is 32-bit (checked <=4B elements in add_tensor_spec)
        size_t current_start = tensor_specs_ptr[opt_m_spec_id].start_element;
        unsigned int offset = (idx - current_start) + (shard_idx * tensor_specs_ptr[opt_m_spec_id].num_elements);

        int grad_spec_id = opt_m_spec_id - num_params_tensors;
        TensorGPU<floatX> grad_tensor  = tensor_specs_ptr[grad_spec_id];

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
    cudaCheck(cudaGetLastError());
    global_sum_deterministic(out, out, grid_size, stream);
    cudaCheck(cudaGetLastError());
}