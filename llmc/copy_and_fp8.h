/*
Helpers for FP8 including copy and transpose with format conversion, and absmax
See /dev/cuda/advanced_copy_transpose.cu for more information and options
*/
#ifndef FP8_HELPERS_CUH
#define FP8_HELPERS_CUH

#include <assert.h>
#include <typeinfo>
#include "cuda_common.h"
#include "cuda_utils.cuh"

// todo - tune these for performance (but should be close to optimal already)
#define TRANSPOSE_TILE_SIZE 64UL

// ----------------------------------------------------------------------------
// elementwise functions which can be applied as part of the copy/transpose
// for elementwise kernels that require metadata (e.g. layernorm forward with known mean/std),
// we could maybe store it in constant buffers rather than in yet-another-function-parameter...
using elementwise_func_t = float (*) (float);
__device__ float nothing_elementwise(float x) {
    return x;
}
__device__ float gelu_forward_elementwise(float x) {
    float cube = 0.044715f * x * x * x;

    float tanh_out;
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + cube);
    asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_out) : "f"(tanh_arg));

    // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
    float half_x = 0.5f * x;
    return half_x * tanh_out + half_x;
}

// ----------------------------------------------------------------------------
// CUDA kernels

// Advanced copy with optional format conversion, absmax, scaling and elementwise operation
template <bool reversed_order=false, bool disable_scaling=false,
          elementwise_func_t elementwise_func=nothing_elementwise,
          typename Tin=float, typename Tout=float>
__global__ void copy_advanced_kernel(TensorGPU<Tout> out, TensorGPU<Tin> in) {
    constexpr size_t vec_size = 16 / ((sizeof(Tin) >= sizeof(Tout)) ? sizeof(Tin) : sizeof(Tout));
    size_t adjusted_blockidx = reversed_order ? (gridDim.x - blockIdx.x - 1) : blockIdx.x;
    size_t idx = (adjusted_blockidx * blockDim.x + threadIdx.x) * vec_size;
    if (idx >= out.num_elements) { return; }

    auto inp128 = load_tensor128(in, idx, true, disable_scaling);
    auto out128 = new_tensor128(out, disable_scaling);
    for (int k = 0; k < vec_size; k++) {
        float out_fp32 = elementwise_func(inp128.get(k));
        out128.set(k, out_fp32);
    }
    out128.template store_same_length<Tin>(idx);
    out128.update_absmax(1);
}

template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=TRANSPOSE_TILE_SIZE, typename T1>
__global__ void transpose_simple_kernel(T1* __restrict__ transposed, const T1* __restrict__ input)
{
    constexpr size_t elements = 16 / sizeof(T1);
    __shared__ T1 tile[TILE_DIM][TILE_DIM];
    int width  = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    int x = blockIdx.x * TILE_DIM + threadIdx.x * elements;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T1> in128 = load128cs<T1>(input + x + (y+j)*width);
        size_t tile_offset = (threadIdx.x * elements) + (threadIdx.y+j)*TILE_DIM;
        store128(&tile[0][0] + tile_offset, in128);
    }
    __syncthreads();

    // x/y for final write to global memory
    x = blockIdx.y * TILE_DIM + threadIdx.x * elements;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T1> out128;
        #pragma unroll
        for (int k = 0; k < elements; k++) {
            // these are tiny 8-bit loads with loads of bank conflicts for FP8
            // extremely hard to avoid and not a bottleneck when everything else is well optimised
            out128[k] = tile[k + threadIdx.x * elements][threadIdx.y + j];
        }
        store128<T1>(transposed + x + (y+j)*height, out128);
    }
}

// only calculate absmax of the input tensor (non-fused)
template <bool disable_scaling=true, typename T>
__global__ void update_absmax_kernel(TensorGPU<T> inp) {
    size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x) * inp.num_per_128();
    auto max128 = new_tensor128(inp);
    if (idx < inp.num_elements) {
        auto inp128 = load_tensor128(inp, idx, disable_scaling);
        for(int k = 0; k < inp.num_per_128(); ++k) {
            float value = inp128.get(k);
            max128.add_value_stats(value);
        }
    }
    max128.update_absmax(threadIdx.x, blockDim.x, true, true);
}

// ----------------------------------------------------------------------------

template <bool reversed_order=false, bool disable_scaling=false, elementwise_func_t elementwise_func=nothing_elementwise, typename T1, typename T2>
void copy_advanced(TensorGPU<T1> out, TensorGPU<T2> in, cudaStream_t stream=0, const size_t block_size=512) {
    size_t N = out.num_elements;
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    assert((N % fewest_elements) == 0);

    const dim3 grid_size(CEIL_DIV(N, block_size * fewest_elements));
    copy_advanced_kernel<reversed_order, disable_scaling, elementwise_func><<<grid_size, dim3(block_size), 0, stream>>>(out, in);
    cudaCheck(cudaGetLastError());
}

template<typename T1>
void transpose_simple(TensorGPU<T1> transposed, TensorGPU<T1> input, size_t w, size_t h, cudaStream_t stream=0, size_t block_size=128) {
    assert((w % TRANSPOSE_TILE_SIZE) == 0 && (h % TRANSPOSE_TILE_SIZE) == 0);
    cudaCheck(cudaGetLastError());

    size_t block_size_x = (TRANSPOSE_TILE_SIZE * sizeof(T1)) / 16;
    size_t block_size_y = min(TRANSPOSE_TILE_SIZE, block_size / block_size_x);
    dim3 grid_size(w / TRANSPOSE_TILE_SIZE, h / (TRANSPOSE_TILE_SIZE));
    dim3 block_size_dim(block_size_x, block_size_y, 1);

    switch (block_size_y) {
        case 64: transpose_simple_kernel<64, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>((T1*)transposed, (T1*)input); break;
        case 32: transpose_simple_kernel<32, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>((T1*)transposed, (T1*)input); break;
        case 16: transpose_simple_kernel<16, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>((T1*)transposed, (T1*)input); break;
        default: printf("Invalid block size (might be easy to add): %lu\n", block_size_y); exit(1);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T>
void update_absmax(TensorGPU<T> inp, bool memset_absmax=true, cudaStream_t stream=main_stream) {
    size_t N = inp.num_elements;
    if (N == 0 || inp.absmax_ptr == NULL) { return; }
    assert(N % inp.num_per_128() == 0);

    size_t block_size = 512;
    const dim3 grid_size(CEIL_DIV(N, block_size * Packed128<T>::size));
    if (memset_absmax) {
        cudaMemset(inp.absmax_ptr, 0, sizeof(unsigned int));
    }
    update_absmax_kernel<<<grid_size, block_size, 0, stream>>>(inp);
    cudaCheck(cudaGetLastError());
}

#endif