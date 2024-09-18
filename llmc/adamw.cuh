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

// always sizeof(param) <= sizeof(grad) <= sizeof(opt/master) <= sizeof(float)
template <bool use_master_weights=true, typename Tparam=floatX, typename Tgrad=floatX, typename Tm=float, typename Tv=float, typename Tmaster=float>
__device__ size_t adamw_update_part(TensorGPU<Tparam> param_tensor, size_t idx, int spec_id, size_t current_start, size_t current_end, size_t stride,
                                    TensorGPU<Tgrad> grad_tensor, TensorGPU<Tmaster> master_tensor, TensorGPU<Tm> opt_m_tensor, TensorGPU<Tv> opt_v_tensor,
                                    unsigned int seed, int num_params_tensors, size_t num_parameters, size_t num_opt_parameters,
                                    float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                                    float eps, float wd, float grad_scale, int t) {
    auto out_master128 = new_tensor128(master_tensor, true);
    auto out_opt_m128 = new_tensor128(opt_m_tensor, true);
    auto out_opt_v128 = new_tensor128(opt_v_tensor, true);
    auto out_param128 = new_tensor128(param_tensor);

    __syncthreads(); // todo - hopefully results in better memory access patterns => TBC
    while (idx < current_end) {
        unsigned int random = get_random_noise(seed, idx);

        tensor128<Tparam> param128;
        tensor128<Tgrad> grad128;
        tensor128<Tm> opt_m128;
        tensor128<Tv> opt_v128;
        tensor128<Tmaster> master128;

        size_t offset = idx - current_start;
        int next_idx[TT::NUM_TYPES_PARAM] = {0};
        int current_idx[TT::NUM_TYPES_PARAM] = {0};

        // this implementation has a stride causing sparse reads/writes and bank conflicts for non-FP8
        // todo - compare performance with a version that uses 128-bit for FP32, 64-bit for BF16, 32-bit for FP8
        #pragma unroll
        for (int i = 0; i < 16; i += 4, offset += 4) {
            if (current_idx[PARAMETER] == 0) param128 = load_tensor128(param_tensor, offset);
            if (current_idx[PARAMETER_GRAD] == 0) grad128 = load_tensor128(grad_tensor, offset, false, true);
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
                out_param128.set_stochastic(current_idx[PARAMETER] + k, param, random);
                float new_param = out_param128.get(current_idx[PARAMETER] + k);
                if constexpr (use_master_weights) {
                    out_master128.set(current_idx[PARAMETER_MASTER] + k, param);
                }
            }
            next_idx[PARAMETER] = (i + 4) % (16 / sizeof(Tparam));
            next_idx[PARAMETER_GRAD] = (i + 4) % (16 / sizeof(Tgrad));
            next_idx[PARAMETER_OPT_M] = (i + 4) % (16 / sizeof(Tm));
            next_idx[PARAMETER_OPT_V] = (i + 4) % (16 / sizeof(Tv));
            next_idx[PARAMETER_MASTER] = (i + 4) % (16 / sizeof(Tmaster));

            if (next_idx[PARAMETER] == 0) out_param128.store(offset - current_idx[PARAMETER]);
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
__global__ void adamw_full_update(TensorSpec* specs, unsigned int seed,
                                  int num_params_tensors, size_t num_parameters, size_t num_opt_parameters,
                                  float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                                  float eps, float weight_decay, float grad_scale, int t) {
    // ...
    constexpr size_t block_size = 64; // 64 ==> 4KiB chunks with iteration_size=16 for FP32 opt/master
    constexpr size_t iteration_size = 16;
    size_t idx = (blockIdx.x * block_size * iteration_size) + (threadIdx.x * iteration_size);
    size_t stride = gridDim.x * blockDim.x * iteration_size;

    int spec_id = 0;
    TensorSpec* grad_specs   = specs + num_params_tensors;
    TensorSpec* opt_m_specs  = specs + 2 * num_params_tensors;
    TensorSpec* opt_v_specs  = specs + 3 * num_params_tensors;
    TensorSpec* master_specs = use_master_weights ? specs + 4 * num_params_tensors : opt_m_specs;

    TensorSpec opt_spec = opt_v_specs[spec_id];
    size_t current_start = opt_spec.offset / sizeof(float);
    size_t current_end = current_start + opt_spec.num_elements;

    while (true) {
        // todo - performance analysis/optimisation! (impact of using step 0?)
        while (idx >= current_end) {
            spec_id++;
            if (spec_id >= num_params_tensors) {
                return;
            }
            opt_spec = opt_v_specs[spec_id];
            current_start = opt_spec.offset / sizeof(float);
            current_end = current_start + opt_spec.num_elements;

            while (idx < current_start) {
                idx += stride;
            }
        }

        opt_spec = opt_v_specs[spec_id];
        current_start = opt_spec.offset / sizeof(float);
        current_end = current_start + opt_spec.num_elements;
        float wd = (opt_spec.flags & TENSOR_2D) ? weight_decay : 0.0f;

        TensorGPU<floatX> grad_tensor = grad_specs[spec_id];
        TensorGPU<float> master_tensor = master_specs[spec_id];
        TensorGPU<float> opt_m_tensor = opt_m_specs[spec_id];
        TensorGPU<float> opt_v_tensor = opt_spec;

        if (specs[spec_id].data_type == DType::FP32) {
            TensorGPU<float> param_tensor = specs[spec_id];
            idx = adamw_update_part<use_master_weights, float>(
                                    param_tensor, idx, spec_id, current_start, current_end, stride,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    seed, num_params_tensors, num_parameters, num_opt_parameters,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else if (specs[spec_id].data_type == DType::BF16) {
            TensorGPU<__nv_bfloat16> param_tensor = specs[spec_id];
            idx = adamw_update_part<use_master_weights, __nv_bfloat16>(
                                    param_tensor, idx, spec_id, current_start, current_end, stride,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    seed, num_params_tensors, num_parameters, num_opt_parameters,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else if (specs[spec_id].data_type == DType::FP8E4M3) {
            TensorGPU<__nv_fp8_e4m3> param_tensor = specs[spec_id];
            idx = adamw_update_part<use_master_weights, __nv_fp8_e4m3>(
                                    param_tensor, idx, spec_id, current_start, current_end, stride,
                                    grad_tensor, master_tensor, opt_m_tensor, opt_v_tensor,
                                    seed, num_params_tensors, num_parameters, num_opt_parameters,
                                    learning_rate, beta1, beta2, beta1_correction, beta2_correction,
                                    eps, wd, grad_scale, t);
        } else {
            assert(false); // TODO (no FP16 to avoid compile time increase but it'd be trivial to add)
        }
    }
}
