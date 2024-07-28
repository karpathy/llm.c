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

        float tanh_out;
        float tanh_arg = tanhf(GELU_SCALING_FACTOR * (xi + cube));
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_out) : "f"(tanh_arg));

        // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
        float half_xi = 0.5f * xi;
        packed_out[k] = (floatX)(half_xi * tanh_out + half_xi);
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

template <typename Ti, typename Tdout, typename Tdin>
__global__ void gelu_backward_kernel(Tdout* d_out, const Tdin* d_in, const Ti* inp, float* inp_descale_pointer, float* d_descale_pointer, float* d_scale_pointer, void* d_out_absmax) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<Tdout>::size;
    float inp_descale_factor = std::is_same<Ti, __nv_fp8_e4m3>::value && inp_descale_pointer ? *inp_descale_pointer : 1.0f;
    float d_descale_factor = std::is_same<Tdin, __nv_fp8_e5m2>::value && d_descale_pointer ? *d_descale_pointer : 1.0f;
    float d_scale_factor = std::is_same<Tdout, __nv_fp8_e5m2>::value && d_scale_pointer ? *d_scale_pointer : 1.0f;
    unsigned int absmax_uint = 0;

    Packed128<Tdout> packed_dinp;
    Packed128<Ti> packed_inp = load128cs(inp + idx);
    Packed128<Tdin> packed_dout = load128(d_in + idx);
    for (int k = 0; k < Packed128<Tdout>::size; ++k) {
        float x = (float)packed_inp[k] * inp_descale_factor;
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);

        float tanh_out;
        asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_out) : "f"(tanh_arg));

        float sech_out = 1.0f - (tanh_out * tanh_out);
        float local_grad = 0.5f * ((1.0f + tanh_out) + x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x));
        float result = local_grad * (float)packed_dout[k] * d_descale_factor;
        packed_dinp[k] = (Tdout)(result * d_scale_factor);

        update_local_absmax(absmax_uint, result, (std::is_same<Tdout, __nv_fp8_e5m2>::value ? 1 : 0)); // optional absmax
    }
    store128(d_out + idx, packed_dinp);
    if (d_out_absmax) {
        update_global_absmax((unsigned int*)d_out_absmax, absmax_uint);
    }
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

template <typename Ti, typename Tdout, typename Tdin>
void gelu_backward(Tdout* d_out, const Tdin* d_in, const Ti* inp, const int N, cudaStream_t stream, float* inp_descale_pointer=NULL, float* d_descale_pointer=NULL, float* d_scale_pointer=NULL, void* d_out_absmax=NULL) {
    NVTX_RANGE_FN();

    // because we are just using Packed128<Tdout>::size for the loop count, Packed128<Ti>::size must be >= that
    assert(sizeof(Tdin) >= sizeof(Tdout));

    int block_size = (N % (512 * Packed128<Tdout>::size) == 0) ? 512 : 64; // use bigger blocks if possible for absmax
    assert(N % (block_size * Packed128<Tdout>::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * Packed128<Tdout>::size);
    gelu_backward_kernel<<<grid_size, block_size, 0, stream>>>(d_out, d_in, inp, inp_descale_pointer, d_descale_pointer, d_scale_pointer, d_out_absmax);
    cudaCheck(cudaGetLastError());
}
