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
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;

        float tanh_in_out = GELU_SCALING_FACTOR * (xi + cube);
        #if !defined(PRECISE_GELU_TANH) && !defined(ENABLE_FP32) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
        float half_xi = 0.5f * xi;
        packed_out[k] = (floatX)(half_xi * tanh_in_out + half_xi);
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

template <typename Ti, typename Tdout, typename Tdinp>
__global__ void gelu_backward_kernel(Tdinp* dinp, const Tdout* dout, const Ti* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<Tdout>::size;

    Packed128<Tdinp> packed_dinp;
    Packed128<Ti> packed_inp = load128cs(inp + idx);
    Packed128<Tdout> packed_dout = load128(dout + idx);
    for (int k = 0; k < Packed128<Tdout>::size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;

        float tanh_in_out = GELU_SCALING_FACTOR * (x + cube);
        #if !defined(PRECISE_GELU_TANH) && !defined(ENABLE_FP32) && __CUDA_ARCH__ >= 750
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_in_out) : "f"(tanh_in_out));
        #else
        tanh_in_out = tanhf(tanh_in_out);
        #endif

        float sech_out = 1.0f - (tanh_in_out * tanh_in_out);
        float local_grad = 0.5f * ((1.0f + tanh_in_out) + x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x));
        float result = local_grad * (float)packed_dout[k];
        packed_dinp[k] = (Tdinp)(result);
    }
    store128(dinp + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, d_in_out, inp);
    cudaCheck(cudaGetLastError());
}
