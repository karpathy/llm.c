/*
AdamW kernel
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "copy_and_fp8.cuh"

#define ADAM_MAX_SLICES 64 // maximum number of slices/layers per invocation
typedef struct {
    float* scale_factor[ADAM_MAX_SLICES];
    void* absmax_output[ADAM_MAX_SLICES];
} param_absmax_for_adam_t;

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed, float* scale_factor = NULL, void* absmax_output = NULL) {
    float param = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parameters) {
        // get the gradient, m, and v for this parameter
        float grad = grad_scale * (float)grads_memory[idx];
        float m = m_memory[idx];
        float v = v_memory[idx];
        // update the first moment (momentum)
        m = lerp(grad, m, beta1);
        m_memory[idx] = m;
        // update the second moment (RMSprop)
        v = lerp(grad * grad, v, beta2);
        v_memory[idx] = v;
        m /= beta1_correction;  // m_hat
        v /= beta2_correction;  // v_hat
        // fetch the old value of this parameter as a float, from either source (FP8 must be from master weights atm)
        float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];

        // update this parameter
        param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
        float scale = scale_factor ? *scale_factor : 1.0f; // for fp8 (absmax scaling)

        // update our low precision version of the parameters using stochastic rounding
        // this will be used in the next forward pass
        stochastic_rounding(param * scale, &params_memory[idx], seed);

        // write the full, float version of the param into our master copy, if we maintain one
        // this will be used in the next update
        if (master_params_memory != NULL) { master_params_memory[idx] = param; }
    }
    if constexpr (std::is_same<Tp, __nv_fp8_e4m3>::value) {
        if (absmax_output == NULL) { return; } // this should never be the case
        // FP8 requires tracking the absmax for scaling
        unsigned int absmax_uint = 0;
        update_local_absmax<true>(absmax_uint, param, 1);
        update_global_absmax<false>((unsigned int*)absmax_output, absmax_uint);
    }
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed, NULL, NULL
                 );
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3_absmax(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed, __grid_constant__ const param_absmax_for_adam_t absmax_params) {
    float *scale_factor = absmax_params.scale_factor[blockIdx.y];
    void *absmax_output = absmax_params.absmax_output[blockIdx.y];

    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed, scale_factor, absmax_output
                 );
}

template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                          ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed, __grid_constant__ const param_absmax_for_adam_t absmax_params) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }
    float scale = absmax_params.scale_factor[blockIdx.y] ? *absmax_params.scale_factor[blockIdx.y] : 1.0f; // for fp8 (absmax scaling)
    params_memory += blockIdx.y * w_stride; // adjust for layer offset
    master_params_memory += blockIdx.y * s_stride;
    stochastic_rounding(master_params_memory[idx] * scale, &params_memory[idx], seed);
}

template <typename Tp>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream, const param_absmax_for_adam_t absmax_params) {
    int block_size = 512; // must match block size of adamw_update so that RNG also matches
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                             (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed, absmax_params);
    cudaCheck(cudaGetLastError());
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);

    // If the parameters are __nv_fp8_e4m3, we need to use the absmax history for the scale factor
    if constexpr (std::is_same<Tp, __nv_fp8_e4m3>::value) {
        // FP8 currently requires using master weights (partly because we don't track the descale factors across steps etc.)
        assert(master_params_memory != NULL);
        // todo - URGENT: if there are more slices/layers, do multiple kernel invocations (or rearchitect this part!)
        assert(num_slices <= ADAM_MAX_SLICES);
        param_absmax_for_adam_t absmax_params;
        // Pass the scale factors and absmax output pointers as function arguments
        // (CUDA supports up to 4KiB pre-12.1 and 64KiB after, so it's fine, although not very flexible due to fixed size)
        for (int i = 0; i < num_slices; i++) {
            global_current_layer = num_slices > 1 ? i+1 : 0;
            Tp* layer_params_memory = params_memory + i * num_parameters;
            float *calculated_from_absmax = absmax_tracker.get_absmax_data("adamw", layer_params_memory, num_parameters, NULL, SCALE_FP8_WEIGHTS, true, true);
            absmax_params.scale_factor[i] = calculated_from_absmax + SCALE_OFFSET;
            absmax_params.absmax_output[i] = absmax_tracker.next_absmax_ptr(layer_params_memory, num_parameters, NULL, 0.0f, false);
        }
        global_current_layer = 0;
        adamw_kernel3_absmax<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                            m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                            grad_scale, seed, absmax_params);

        // HACK - WIP - this updates the scale based on the latest update, then writes the parameters again reading from master weights
        // we need a more general solution that supports multi-GPU (the entire tensor needs to have the same scaling factor!)
        absmax_tracker.update_all_absmax(stream, 1.0f, false);
        init_from_master(params_memory, master_params_memory, num_parameters, w_stride, s_stride, num_slices, seed, stream, absmax_params);
    } else {
        adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                            m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                            grad_scale, seed);
    }
    cudaCheck(cudaGetLastError());
}
