/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel2(tensorFP8e4 out, tensorFP8e4 inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * inp.num_per_128();

    auto out128 = new_tensor128(out);
    auto inp128 = load_tensor128(inp, idx, true);
    for(int k = 0; k < inp.num_per_128(); ++k) {
        float xi = inp128.get(k);
        float cube = 0.044715f * xi * xi * xi;

        float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
        #if !defined(PRECISE_GELU_TANH) && !defined(ENABLE_FP32) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
        float half_xi = 0.5f * xi;
        out128.set(k, half_xi * tanh_in_out + half_xi);
    }
    out128.store_same_length<floatX>(idx, false);
    out128.update_absmax(threadIdx.x, blockDim.x, true);
}

//template<typename Tinp=floatX>
template<typename Tinp=floatX>
__global__ void gelu_backward_kernel(tensorFP8e5 dinp, tensorFP8e5 dout, TensorGPU<Tinp> inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * dout.num_per_128();

    auto dinp128 = new_tensor128(dinp);
    auto inp128 = load_tensor128(inp, idx, true);
    auto dout128 = load_tensor128(dout, idx);
    for (int k = 0; k < dout.num_per_128(); ++k) {
        float x = inp128.get(k);
        float cube = 0.044715f * x * x * x;

        float tanh_in_out = GELU_SCALING_FACTOR * (x + cube);
        #if !defined(PRECISE_GELU_TANH) && !defined(ENABLE_FP32) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        float sech_out = 1.0f - (tanh_in_out * tanh_in_out);
        float local_grad = 0.5f * ((1.0f + tanh_in_out) + x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x));
        float result = local_grad * (float)dout128.get(k);
        dinp128.set(k, result);
    }
    dinp128.store_same_length<floatX>(idx, false);
    dinp128.update_absmax(threadIdx.x, blockDim.x, true);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(tensorX out, tensorX inp, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    assert(inp.num_elements % (block_size * inp.num_per_128()) == 0);

    const int grid_size = CEIL_DIV(inp.num_elements, block_size * inp.num_per_128());
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(tensorX dinp, tensorX dout, tensorX inp, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int grid_size = CEIL_DIV(inp.num_elements, block_size * inp.num_per_128());
    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(dinp, dout, inp);
    cudaCheck(cudaGetLastError());
}
