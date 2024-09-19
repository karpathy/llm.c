/*
AdamW kernel
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <bool use_master_weights=true, typename Tparam=floatX, typename Tgrad=floatX, typename Tm=float, typename Tv=float, typename Tmaster=float>
__device__ size_t adamw_update_part(TensorGPU<Tparam> param_tensor,
                                    size_t idx, size_t current_start, size_t current_end, size_t stride, unsigned int seed, unsigned int shard_idx,
                                    TensorGPU<Tgrad> grad_tensor, TensorGPU<Tmaster> master_tensor, TensorGPU<Tm> opt_m_tensor, TensorGPU<Tv> opt_v_tensor,
                                    float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                                    float eps, float wd, float grad_scale, int t) {
    auto out_master128 = new_tensor128<use_master_weights>(master_tensor, true);
    auto out_opt_m128 = new_tensor128(opt_m_tensor, true);
    auto out_opt_v128 = new_tensor128(opt_v_tensor, true);
    auto out_param128 = new_tensor128(param_tensor);

    __syncthreads(); // todo - this should improve memory locality
    while (idx < current_end) {
        unsigned int random = get_random_noise(seed, idx);

        tensor128<Tparam> param128;
        tensor128<Tgrad> grad128;
        tensor128<Tm> opt_m128;
        tensor128<Tv> opt_v128;
        tensor128<Tmaster> master128;
        int next_idx[TT::NUM_TYPES_PARAM] = {0};
        int current_idx[TT::NUM_TYPES_PARAM] = {0};

        // todo - assuming either DPP or ZeRO 1 now (sharded optimizer/master, unsharded gradients/parameters)
        // offset is 32-bit (checked <= elements in add_tensor_spec)
        unsigned int offset = idx - current_start;
        unsigned int unsharded_offset = offset + shard_idx * opt_v_tensor.num_elements;

        // this implementation has a stride causing sparse reads/writes and bank conflicts for non-FP8
        // todo - compare performance with a version that uses 128-bit for FP32, 64-bit for BF16, 32-bit for FP8 (probably much faster)
        #pragma unroll
        for (int i = 0; i < 16; i += 4, offset += 4, unsharded_offset += 4) {
            if (current_idx[PARAMETER] == 0) param128 = load_tensor128(param_tensor, unsharded_offset);
            if (current_idx[PARAMETER_GRAD] == 0) grad128 = load_tensor128(grad_tensor, unsharded_offset, false, true);
            if (current_idx[PARAMETER_OPT_M] == 0) opt_m128 = load_tensor128(opt_m_tensor, offset, false,true);
            if (current_idx[PARAMETER_OPT_V] == 0) opt_v128 = load_tensor128(opt_v_tensor, offset, false, true);
            if (current_idx[PARAMETER_MASTER] == 0 && use_master_weights) master128 = load_tensor128(master_tensor, offset, false, true);

            for (int k = 0; k < 4; k++) {
                float grad = grad128.get(current_idx[PARAMETER_GRAD] + k);
                float m = opt_m128.get(current_idx[PARAMETER_OPT_M] + k);
                float v = opt_v128.get(current_idx[PARAMETER_OPT_V] + k);

                m = lerp(grad, m, beta1);
                v = lerp(grad * grad, v, beta2);
                out_opt_m128.set(current_idx[PARAMETER_OPT_M] + k, m);
                out_opt_v128.set(current_idx[PARAMETER_OPT_V] + k, v);
                m /= beta1_correction;
                v /= beta2_correction;

                float old_param;
                if constexpr (use_master_weights) {
                    old_param = master128.get(current_idx[PARAMETER_MASTER] + k);
                } else {
                    old_param = param128.get(current_idx[PARAMETER] + k);
                }

                float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + wd * old_param));
                if constexpr (use_master_weights) {
                    out_master128.set(current_idx[PARAMETER_MASTER] + k, param);
                }
                out_param128.set_stochastic(current_idx[PARAMETER] + k, param, random);
            }
            next_idx[PARAMETER] = (i + 4) % (16 / sizeof(Tparam));
            next_idx[PARAMETER_GRAD] = (i + 4) % (16 / sizeof(Tgrad));
            next_idx[PARAMETER_OPT_M] = (i + 4) % (16 / sizeof(Tm));
            next_idx[PARAMETER_OPT_V] = (i + 4) % (16 / sizeof(Tv));
            next_idx[PARAMETER_MASTER] = (i + 4) % (16 / sizeof(Tmaster));

            if (next_idx[PARAMETER] == 0) out_param128.store(unsharded_offset - current_idx[PARAMETER]);
            if (next_idx[PARAMETER_OPT_M] == 0) out_opt_m128.store(offset - current_idx[PARAMETER_OPT_M]);
            if (next_idx[PARAMETER_OPT_V] == 0) out_opt_v128.store(offset - current_idx[PARAMETER_OPT_V]);
            if constexpr (use_master_weights) {
                if (next_idx[PARAMETER_MASTER] == 0) out_master128.store(offset - current_idx[PARAMETER_MASTER]);
            }

            for (int n = 0; n < TT::NUM_TYPES_PARAM; n++) {
                current_idx[n] = next_idx[n];
            }
        }
        idx += stride;
    }
    out_param128.update_absmax(1);
    return idx;
}

template <bool use_master_weights=true>
__global__ void adamw_update_everything(int num_params_tensors, int start_tensor, int last_tensor, unsigned int seed , unsigned int shard_idx,
                                        float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                                        float eps, float weight_decay, float grad_scale, int t) {
    // ...
    constexpr size_t block_size = 64;
    constexpr size_t iteration_size = 16; // todo - this causes sparsit and bank clashes for FP32/BF16 loads/stores
    size_t idx = (blockIdx.x * block_size * iteration_size) + (threadIdx.x * iteration_size);
    unsigned int stride = gridDim.x * blockDim.x * iteration_size;

    int opt_m_spec_id = 2 * num_params_tensors;
    int last_opt_m_id = opt_m_spec_id + last_tensor; // opt_m is sharded with ZeRO 1 so use it as reference
    opt_m_spec_id += start_tensor - 1; // -1 to compensate for the increment at the start of the loop below

    while (true) {
        size_t current_end;
        do {
            opt_m_spec_id++;
            if (opt_m_spec_id > last_opt_m_id) return; // done!

            // on A100+ we can prefetch 256B (32 values) into the L2, on older GPUs just use a regular load
            #if __CUDA_ARCH__ < 800
            current_end = tensor_end_element_ptr[opt_m_spec_id];
            #else
            asm("ld.global.L2::256B.u64 {%0}, [%1];" : "=l"(current_end) : "l"(tensor_end_element_ptr + opt_m_spec_id));
            #endif
        } while (idx >= current_end);

        int spec_id = opt_m_spec_id - 2 * num_params_tensors;
        size_t current_start = tensor_specs_ptr[opt_m_spec_id].start_element;

        TensorSpec param_spec = tensor_specs_ptr[spec_id];
        TensorGPU<floatX> grad_tensor  = tensor_specs_ptr[spec_id + 1*num_params_tensors];
        TensorGPU<float> opt_m_tensor  = tensor_specs_ptr[spec_id + 2*num_params_tensors];
        TensorGPU<float> opt_v_tensor  = tensor_specs_ptr[spec_id + 3*num_params_tensors];
        TensorGPU<float> master_tensor = use_master_weights ? tensor_specs_ptr[spec_id + 4*num_params_tensors] : opt_m_tensor;

        float wd = (param_spec.flags & TENSOR_2D) ? weight_decay : 0.0f;

        if (param_spec.data_type == DType::FP32) {
            idx = adamw_update_part<use_master_weights>((TensorGPU<float>)param_spec,
                                    idx, current_start, current_end, stride, seed, shard_idx,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else if (param_spec.data_type == DType::BF16) {
            idx = adamw_update_part<use_master_weights>((TensorGPU<__nv_bfloat16>)param_spec,
                                    idx, current_start, current_end, stride, seed, shard_idx,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else if (param_spec.data_type == DType::FP8E4M3) {
            idx = adamw_update_part<use_master_weights>((TensorGPU<__nv_fp8_e4m3>)param_spec,
                                    idx, current_start, current_end, stride, seed, shard_idx,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else {
            assert(false); // TODO (no FP16 to avoid compile time increase but trivial to add here)
        }
    }
}
