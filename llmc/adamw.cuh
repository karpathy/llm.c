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

template <bool use_master_weights=true, bool master_init_modes=false>
__global__ void adamw_full_update(TensorSpec* specs, unsigned int seed,
                                  int num_params_tensors, size_t num_parameters, size_t num_opt_parameters,
                                  float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction,
                                  float eps, float weight_decay, float grad_scale, int t, bool init_from_master_only=false) {
    // ...
    constexpr size_t block_size = 64; // 64 ==> 4KiB chunks with iteration_size=16 for FP32 opt/master
    size_t iteration_size = 16;
    assert(iteration_size <= 16);
    size_t idx_blk = blockIdx.x * block_size * iteration_size;
    size_t idx = idx_blk + (threadIdx.x * iteration_size);
    size_t stride = gridDim.x * blockDim.x * iteration_size;

    int spec_id = 0;

    TensorSpec* grad_specs   = specs + num_params_tensors;
    TensorSpec* opt_m_specs  = specs + 2 * num_params_tensors;
    TensorSpec* opt_v_specs  = specs + 3 * num_params_tensors;
    TensorSpec* master_specs = use_master_weights ? specs + 4 * num_params_tensors : opt_m_specs;

    TensorSpec opt_spec = opt_v_specs[spec_id];
    size_t current_start = opt_spec.offset / sizeof(float);
    size_t current_end = current_start + opt_spec.num_elements;

    while (idx < num_opt_parameters) {
        // todo - do this part on thread 0 only?
        while (idx >= current_end) {
            spec_id++;
            if (spec_id >= num_params_tensors) {
                return;
            }
            opt_spec = opt_v_specs[spec_id];
            current_start = opt_spec.offset / sizeof(float);
            current_end = current_start + opt_spec.num_elements;
        }

        TensorGPU<floatX> grad_tensor = grad_specs[spec_id];
        TensorGPU<float> master_tensor = master_specs[spec_id];
        TensorGPU<float> opt_m_tensor = opt_m_specs[spec_id];
        TensorGPU<float> opt_v_tensor = opt_spec;

        auto out_master128 = new_tensor128(master_tensor, true);
        auto out_opt_m128 = new_tensor128(opt_m_tensor, true);
        auto out_opt_v128 = new_tensor128(opt_v_tensor, true);

        // todo - make it configurable whether weight decay applies to e.g. bias or not
        float wd = (opt_spec.flags & TENSOR_2D) ? weight_decay : 0.0f;

        if (specs[spec_id].data_type == DType::BF16) {
            TensorGPU<__nv_bfloat16> param_tensor = specs[spec_id];
            auto out_param128 = new_tensor128(param_tensor);

            __syncthreads(); // todo - hopefully results in better memory access patterns => TBC
            while (idx < current_end) {
                // always sizeof(param) <= sizeof(grad) <= sizeof(opt/master)
                // todo - maybe not true, could have FP32 param and BF16 grad?
                // todo - hack - currently assuming grad is always bfloat16
                unsigned int random = get_random_noise(seed, idx);
                for (int i = 0; i < iteration_size; i += 16 / sizeof(__nv_bfloat16)) {
                    size_t offset = (idx - current_start) + i;
                    auto param128 = load_tensor128(param_tensor, offset);
                    auto grad128 = load_tensor128(grad_tensor, offset);
                    for (int j = 0; j < sizeof(float) / sizeof(__nv_bfloat16); j++) {
                        // todo - sparse(-ish) accesses, I don't like it.
                        auto opt_m128 = load_tensor128(opt_m_tensor, offset + j * f128::size, true);
                        auto opt_v128 = load_tensor128(opt_v_tensor, offset + j * f128::size, true);
                        // optimised away if we don't use it (and pointer will be equal to opt_m128)
                        auto master128 = load_tensor128(master_tensor, offset + j * f128::size, true);

                        if (master_init_modes && init_from_master_only) {
                            for (int k = 0; k < f128::size; k++) {
                                float old_param = master128.get(k);
                                out_param128.set_stochastic(k + j*f128::size, old_param, random);
                            }
                            continue;
                        }

                        for (int k = 0; k < f128::size; k++) {
                            float grad = grad128.get(k + j*f128::size);
                            float m = opt_m128.get(k);
                            float v = opt_v128.get(k);
                            m = lerp(grad, m, beta1);
                            v = lerp(grad * grad, v, beta2);
                            out_opt_m128.set(k, m);
                            out_opt_v128.set(k, v);
                            m /= beta1_correction;
                            v /= beta2_correction;

                            float old_param;
                            if (use_master_weights && !master_init_modes) {
                                old_param = master128.get(k);
                            } else {
                                old_param = param128.get(k + j*f128::size);
                            }
                            float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + wd * old_param));
                            out_param128.set_stochastic(k + j*f128::size, param, random);
                            out_master128.set(k, param);
                        }
                        out_opt_m128.store(offset + j * f128::size);
                        out_opt_v128.store(offset + j * f128::size);
                        if constexpr (use_master_weights) {
                            out_master128.store(offset + j * f128::size);
                        }
                    }
                    out_param128.store(offset);
                }
                out_param128.update_absmax(threadIdx.x, block_size, false);
                idx_blk += stride;
                idx += stride;
            }
        }
    }
}
