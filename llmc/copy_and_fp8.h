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
#define ABSMAX_ITERATIONS_PER_THREAD 4
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

// Same as copy_simple_kernel but with optional absmax and elementwise function options
// absmax is calculated before scaling but after the elementwise function
template <int block_size=256, bool disable_scaling=false, bool reversed_order=false,
          elementwise_func_t elementwise_func=nothing_elementwise,
          typename T1=float, typename T2=float>
__global__ void copy_advanced_kernel(TensorGPU<T1> in, TensorGPU<T2> out) {
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));
    size_t adjusted_blockidx = reversed_order ? (gridDim.x - blockIdx.x - 1) : blockIdx.x;
    size_t idx = (adjusted_blockidx * blockDim.x + threadIdx.x) * vec_size;
    if (idx >= in.num_elements) { return; }

    auto inp128 = load_tensor128(in, idx, true, disable_scaling);
    auto out128 = new_tensor128(out);
    for (int k = 0; k < vec_size; k++) {
        float out_fp32 = elementwise_func(inp128.get(k));
        out128.set(k, out_fp32);
    }
    out128.store_same_length(idx);
    out128.update_absmax(threadIdx.x, block_size, true);
}

// transpose + copy + format conversion (+ elementwise + absmax) kernel
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=TRANSPOSE_TILE_SIZE, typename T1>
__global__ void transpose_simple_kernel(T1* __restrict__ transposed, const T1* __restrict__ input, int height)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM];
    int width  = gridDim.x * TILE_DIM;
    height = gridDim.y * TILE_DIM;

    constexpr size_t elements = 16 / sizeof(T1);
    int x = blockIdx.x * TILE_DIM + (threadIdx.x * elements);
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T1> in128 = load128cs<T1>(input + x + (y+j)*width);
        size_t tile_offset = (threadIdx.x * elements) + (threadIdx.y+j)*TILE_DIM;
        store128(&tile[0][0] + tile_offset, in128);
    }
    __syncthreads();

    constexpr size_t block_size_x = (TILE_DIM * sizeof(T1)) / 16;
    constexpr size_t block_size_y = BLOCK_ROWS;

    int adjusted_tid_x = threadIdx.x % block_size_x;
    int adjusted_tid_y = (threadIdx.y) + (threadIdx.x / block_size_y);

    // x/y for final write to global memory
    x = blockIdx.y * TILE_DIM + adjusted_tid_x * elements;
    y = blockIdx.x * TILE_DIM + adjusted_tid_y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T1> out128;
        #pragma unroll
        for (int k = 0; k < elements; k++) {
            // these are tiny 8-bit loads with loads of bank conflicts for FP8
            // extremely hard to avoid and not a bottleneck when everything else is well optimised
            out128[k] = tile[k + (adjusted_tid_x) * out128.size][adjusted_tid_y + j];
        }
        store128<T1>(transposed + x + out128.size + (y+j)*height, out128);
    }
}

// only calculate absmax of the input tensor (non-fused)
template <bool disable_scaling=true, typename T>
__global__ void update_absmax_kernel(TensorGPU<T> inp) {
    size_t idx = ((blockIdx.x * blockDim.x * ABSMAX_ITERATIONS_PER_THREAD) + threadIdx.x) * inp.num_per_128();
    auto max128 = new_tensor128(inp, disable_scaling);
    if (idx < inp.num_elements) {
        #pragma unroll
        for (int i = 0; i < ABSMAX_ITERATIONS_PER_THREAD; i++) {
            auto inp128 = load_tensor128(inp, idx, disable_scaling);
            for(int k = 0; k < inp.num_per_128(); ++k) {
                float value = inp128.get(k);
                max128.add_value_stats(value);
            }
            idx += blockDim.x * inp.num_per_128();
        }
    }
    max128.update_absmax(threadIdx.x, blockDim.x, true, true);
}

// ----------------------------------------------------------------------------

template <bool reversed_order=false, elementwise_func_t elementwise_func=nothing_elementwise, bool reciprocal=true, typename T1, typename T2>
void copy_advanced(TensorGPU<T1> *copy, TensorGPU<T2> *input, size_t N, float* descale_pointer=NULL, float* scale_pointer=NULL, void* absmax_output=NULL, /*bool memset_absmax=true,*/ cudaStream_t stream=0, const size_t block_size=512) {
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(CEIL_DIV(N, block_size * fewest_elements));
    assert((N % fewest_elements) == 0);

    constexpr uint absmax_factor = 1;
    unsigned int* absmax_uint = (unsigned int*)absmax_output;

    // todo - fix this function
    assert(false);

    if (absmax_output) {
        /*if (memset_absmax) {
            cudaMemset(absmax_output, 0, sizeof(unsigned int));
        }*/
        if (scale_pointer || descale_pointer) {
            copy_advanced_kernel<reciprocal, true, reversed_order, elementwise_func, absmax_factor><<<grid_size, dim3(block_size), 0, stream>>>(copy, input, N, descale_pointer, scale_pointer, absmax_uint);
        } else {
            copy_advanced_kernel<reciprocal, false, reversed_order, elementwise_func, absmax_factor><<<grid_size, dim3(block_size), 0, stream>>>(copy, input, N, NULL, NULL, absmax_uint);
        }
    } else {
        if (scale_pointer || descale_pointer) {
            copy_advanced_kernel<reciprocal, true, reversed_order, elementwise_func><<<grid_size, dim3(block_size), 0, stream>>>(copy, input, N, descale_pointer, scale_pointer);
        } else {
            copy_advanced_kernel<reciprocal, false, reversed_order, elementwise_func><<<grid_size, dim3(block_size), 0, stream>>>(copy, input, N);
        }
    }
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
        case 64: transpose_simple_kernel<64, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>(transposed, input, h); break;
        case 32: transpose_simple_kernel<32, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>(transposed, input, h); break;
        case 16: transpose_simple_kernel<16, TRANSPOSE_TILE_SIZE><<<grid_size, block_size_dim, 0, stream>>>(transposed, input, h); break;
        default: printf("Invalid block size (might be easy to add): %lu\n", block_size_y); exit(1);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T>
void update_absmax(TensorGPU<T> inp, bool memset_absmax=false, cudaStream_t stream=main_stream, size_t max_block_size=512) {
    size_t N = inp.num_elements;
    if (N == 0 || inp.absmax_ptr == NULL) { return; }

    // find the largest block size that divides N
    size_t block_size = max_block_size;
    while ((N % (block_size * Packed128<T>::size * ABSMAX_ITERATIONS_PER_THREAD)) != 0) {
        block_size /= 2;
        assert(block_size >= 32); // block size of 1 would be OK, but so inefficient we'd rather fail and debug I think
    }

    const dim3 grid_size(CEIL_DIV(N, block_size * ABSMAX_ITERATIONS_PER_THREAD * Packed128<T>::size));
    if (memset_absmax) {
        cudaMemset(inp.absmax_ptr, 0, sizeof(unsigned int));
    }
    update_absmax_kernel<<<grid_size, block_size, 0, stream>>>(inp);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Scratch allocation for FP8 conversions etc.
// todo - consider alternatives (or at least move it somewhere else)

#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

class CudaScratchAllocator {
private:
    struct Allocation {
        void* ptr;
        size_t size;
        bool in_use;

        Allocation(void* p, size_t s) : ptr(p), size(s), in_use(false) {}
    };

    static std::vector<Allocation> allocations;
    static size_t total_allocated;

public:
    template<typename T>
    static T* getMemory(size_t count, bool exact=false) {
        size_t size = count * sizeof(T);

        // Find the smallest free allocation that fits the requested size
        auto it = std::min_element(allocations.begin(), allocations.end(),
            [size](const Allocation& a, const Allocation& b) {
                return !a.in_use && a.size >= size && (b.in_use || b.size < size || a.size < b.size);
            });

        if (it != allocations.end() && !it->in_use && it->size >= size && (!exact || it->size == size)) {
            it->in_use = true;
            return reinterpret_cast<T*>(it->ptr);
        }

        // If no suitable allocation found, create a new one
        void* new_ptr;
        cudaMalloc(&new_ptr, size);
        allocations.emplace_back(new_ptr, size);
        allocations.back().in_use = true;
        total_allocated += size;
        printf("Allocated CUDA scratch memory: %lu bytes (%p) ==> total allocated: %.1fGiB\n", size, new_ptr, total_allocated / (1024.0 * 1024.0 * 1024.0));
        return reinterpret_cast<T*>(new_ptr);
    }

    template<typename T>
    static void releaseMemory(T* ptr) {
        if (ptr == nullptr) { return; }
        auto it = std::find_if(allocations.begin(), allocations.end(),
            [ptr](const Allocation& a) { return a.ptr == (void*)ptr; });

        if (it != allocations.end()) {
            it->in_use = false;
        }
    }

    static void cleanup() {
        for (const auto& alloc : allocations) {
            cudaFree(alloc.ptr);
        }
        allocations.clear();
    }
};
std::vector<CudaScratchAllocator::Allocation> CudaScratchAllocator::allocations;
size_t CudaScratchAllocator::total_allocated = 0;

// ----------------------------------------------------------------------------
// Transposed Cache (for FP8 weights)

#include <functional>

// Custom hash function for std::pair<uint64_t, uint64_t>
// todo - why did we need this? complained about default constructor issue?
struct PairHash {
    std::size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint64_t>{}(p.second) << 1);
    }
};

class TransposedCache {
private:
    struct CacheEntry {
        void* ptr;
        size_t size;
    };

    std::unordered_map<std::pair<uint64_t, uint64_t>, CacheEntry, PairHash> cache;

public:
    TransposedCache() = default;

    template<typename T, typename Tout=T>
    Tout* getTransposed(const T* original, const void* associatedTensor, size_t m, size_t k, bool compute=true, bool find_only=false, cudaStream_t stream=0) {
        uint64_t key1 = reinterpret_cast<uint64_t>(original);
        uint64_t key2 = reinterpret_cast<uint64_t>(associatedTensor);
        auto key = std::make_pair(key1, key2);
        size_t size = m * k * sizeof(T);

        auto it = cache.find(key);
        if (it != cache.end() && it->second.size == size) {
            return reinterpret_cast<Tout*>(it->second.ptr);
        }
        if (find_only) {
            return nullptr;
        }

        Tout* transposed = CudaScratchAllocator::getMemory<Tout>(m * k, true);
        if (compute) {
            // todo
            //copy_or_transpose<false>(true, transposed, original, m, k, nullptr, nullptr, nullptr, stream);
        }

        cache[key] = {transposed, size};
        return transposed;
    }

    void clearCache() {
        for (const auto& entry : cache) {
            CudaScratchAllocator::releaseMemory(entry.second.ptr);
        }
        cache.clear();
    }
};
TransposedCache g_transposed_cache;

#endif