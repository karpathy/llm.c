/*
Helpers for FP8 including copy and transpose with format conversion, and absmax
See /dev/cuda/advanced_copy_transpose.cu for more information and options
*/
#include <assert.h>
#include <typeinfo>
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "absmax_history.cuh"

// todo - tune these for performance (but should be close to optimal already)
#define ABSMAX_ITERATIONS_PER_THREAD 4
#define DEFAULT_TILE_SIZE 32UL


// ----------------------------------------------------------------------------
// CUDA helpers
// This helper is for when we want to copy from e.g. FP32 to BF16
// so if want to load a f128 of 4 elements, and write those 4 elements to memory as 64-bit
// not needed in the case of loads, the compiler will automatically optimise away unused reads
template<class OriginalType, class ElementType>
__device__ void store_same_length(ElementType* target, Packed128<ElementType> value) {
    int4 bits = value.get_bits();
    switch (sizeof(OriginalType) / sizeof(ElementType)) {
        case 0: *reinterpret_cast<int4*>(target) = bits; // smaller
        case 1: *reinterpret_cast<int4*>(target) = bits; // same size
        case 2: *reinterpret_cast<int2*>(target) = make_int2(bits.x, bits.y); break;
        case 4: *reinterpret_cast<int*>(target) = bits.x; break;
        default: break; //assert(false);
    }
}

// elementwise functions which can be applied as part of the copy/transpose
// for elementwise kernels that require metadata (e.g. layernorm forward with known mean/std),
// we could maybe store it in constant buffers rather than in yet-another-function-parameter...
using elementwise_func_t = float (*) (float);
__host__ __device__ float nothing_elementwise(float x) {
    return x;
}
__host__ __device__ float gelu_forward_elementwise(float x) {
    float cube = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + cube)));
}

// updates the absmax for the entire threadgroup
// requires all warps in threadblock to be active
// caller can rely on there always being a __syncthreads() for other work
template <bool is2D=false> // templating to avoid useless calculations for 1D (no support for 3D)
__device__ void update_global_absmax(unsigned int* absmax_output, unsigned int absmax_uint) {
    uint bidY = is2D ? blockDim.y : 1;
    uint tidY = is2D ? threadIdx.y : 0;
    uint tidXY = threadIdx.x + blockDim.x * tidY;
    uint num_warps = (blockDim.x * bidY) / 32;
    uint lane_id = tidXY % 32;
    uint warp_id = tidXY / 32;

    // use native integer reductions as much as possible (supported on all GPUs with FP8)
    // todo - we could use cooperative groups instead of PTX here but it'd increase compile time
    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ uint tmp[32];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();

    if (warp_id == 0) {
        absmax_uint = tmp[lane_id < num_warps ? lane_id : 0];
        // compiler automatically does a warp reduction here and global atomic is single-threaded
        // if we try to do it ourselves, we might end up with *two* warp reductions :(
        atomicMax(absmax_output, absmax_uint);
    }
}

// absmax factor is related to the maximum value of the target format, e.g. 448 for FP8 e4m3
// we skip everything absmax-related if it is at the default value of 0
// absmax_factor should be a constant known at compile time for performance
template <bool always=false, bool absmax_exponent_only=false, typename T>
__device__ void update_local_absmax(unsigned int &absmax_uint, T data, uint absmax_factor=0) {
    if (always || absmax_factor != 0) {
        float data_float = (float)data;
        // make sure it's not NaN or Inf etc.
        if (data_float == data_float) {
            constexpr uint absmax_mask = absmax_exponent_only ? 0x7f800000 : 0x7fffffff;
            absmax_uint = max(absmax_uint, __float_as_uint(data_float / (float)absmax_factor) & absmax_mask);
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA kernels

// copy & format conversion kernel using store_same_length
// keeps the largest format at 128-bit and smallest at 32-bit or 64-bit
template <bool reciprocal=true, bool scaling=false, typename T1, typename T2>
__global__ void copy_simple_kernel(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=(float*)NULL) {
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; }

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    scale_factor = (reciprocal && scale_factor != 0.0f) ? (1.0f / scale_factor) : scale_factor;
    Packed128<T2> inp128 = load128cs<T2>(input + n);
    Packed128<T1> out128;
    for (int k = 0; k < vec_size; k++) {
        out128[k] = (T1)((float)inp128[k] * scale_factor);
    }
    store_same_length<T2,T1>(copy + n, out128);
}

// Same as copy_simple_kernel but with optional absmax and elementwise function options
// absmax is calculated before scaling but after the elementwise function
template <bool reciprocal=true, bool scaling=false, bool reversed_order=false,
          elementwise_func_t elementwise_func=nothing_elementwise, int absmax_factor=0, typename T1, typename T2>
__global__ void copy_advanced_kernel(T1 *copy, const T2 *input, size_t N,
                                     const float* __restrict__ scale_pointer=(float*)NULL,
                                     unsigned int* absmax_output=(unsigned int*)NULL,
                                     const void** meta=NULL) {
    uint absmax_uint = 0;
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));
    size_t adjusted_blockidx_x = reversed_order ? (gridDim.x - blockIdx.x - 1) : blockIdx.x;
    size_t n = (adjusted_blockidx_x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; }

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    scale_factor = (reciprocal && scale_factor != 0.0f) ? (1.0f / scale_factor) : scale_factor;
    Packed128<T2> inp128 = load128cs<T2>(input + n);
    Packed128<T1> out128;
    for (int k = 0; k < vec_size; k++) {
        float out_float = elementwise_func((float)inp128[k]);
        update_local_absmax(absmax_uint, out_float, absmax_factor); // optional absmax
        out128[k] = (T1)(out_float * scale_factor);
    }
    store_same_length<T2,T1>(copy + n, out128);

    if constexpr (absmax_factor != 0) {
        update_global_absmax<false>(absmax_output, absmax_uint);
    }
}

// transpose + copy + format conversion (+ elementwise + absmax) kernel
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=DEFAULT_TILE_SIZE, bool reciprocal=true, bool enable_copy=false, bool scaling=true,
         uint absmax_factor=0, elementwise_func_t elementwise_func=nothing_elementwise, typename T1, typename T2>
__global__ void transpose_kernel(T1* __restrict__ transposed, T1* __restrict__ copy, const T2* __restrict__ input,
                                 const float* __restrict__ scale_pointer=(float*)NULL, unsigned int* absmax_output=(unsigned int*)NULL, const void** meta=NULL)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM];
    int width  = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    constexpr size_t T1_elements = 16 / sizeof(T1);
    constexpr size_t T2_elements = 16 / sizeof(T2);
    constexpr size_t copy_vectors = (sizeof(T1) >= sizeof(T2)) ? (sizeof(T1) / sizeof(T2)) : 1;

    float scale_factor = (scaling && scale_pointer) ? *scale_pointer : 1.0f;
    scale_factor = (reciprocal && scale_factor != 0.0f) ? (1.0f / scale_factor) : scale_factor;
    int x = blockIdx.x * TILE_DIM + (threadIdx.x * T2_elements);
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    uint absmax_uint = 0;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T2> in128 = load128cs<T2>(input + x + (y+j)*width);
        Packed128<T1> copy128[copy_vectors];
        for (int k = 0; k < in128.size; k++) {
            T2 in = in128[k];
            float out_float = elementwise_func((float)in);
            update_local_absmax(absmax_uint, out_float, absmax_factor); // optional absmax

            T1 out = (T1)(out_float * scale_factor);
            copy128[k/T1_elements][k%T1_elements] = out; // optimised away by compiler if unused
        }

        for (int o = 0; o < copy_vectors; o++) {
            if constexpr (enable_copy) {
                store_same_length<T2,T1>(copy + x + (y+j)*width + o*T1_elements, copy128[o]);
            }
            size_t tile_offset = (threadIdx.x * T2_elements) + (threadIdx.y+j)*TILE_DIM + o*T1_elements;
            store_same_length<T2,T1>(&tile[0][0] + tile_offset, copy128[o]);
        }
    }

    if constexpr (absmax_factor != 0) {
        update_global_absmax<true>(absmax_output, absmax_uint);
    } else {
        __syncthreads();
    }

    // reduce the number of threads for the write if T1_elements > T2_elements
    // we want to keep all 32 threads in a warp active, so we try to eliminate in y dimension first
    // so we create fake/adjusted tid.x/tid.y where "extra" threadIdx.x adds to the effective tid.y
    constexpr size_t block_size_x = (TILE_DIM * sizeof(T2)) / 16;
    constexpr size_t block_size_y = BLOCK_ROWS;

    constexpr size_t desired_ratio = (sizeof(T2) >= sizeof(T1)) ? (sizeof(T2) / sizeof(T1)) : 1;
    constexpr size_t ratio = (desired_ratio <= block_size_y) ? desired_ratio : block_size_y;
    constexpr size_t block_size_x_div_r = block_size_x / ratio;
    constexpr size_t block_size_y_div_r = block_size_y / ratio;

    int adjusted_tid_x = threadIdx.x % block_size_x_div_r;
    int adjusted_tid_y = (threadIdx.y * ratio) + (threadIdx.x / block_size_x_div_r);
    if (threadIdx.y >= block_size_y_div_r) { return; }

    // if we cannot reduce block_size.y enough, also reduce x (hurting perf with partial warps)
    if (ratio != desired_ratio && adjusted_tid_x >= TILE_DIM / T1_elements) { return; }

    // x/y for final write to global memory
    x = blockIdx.y * TILE_DIM + adjusted_tid_x * T1_elements;
    y = blockIdx.x * TILE_DIM + adjusted_tid_y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // we need more instructions for the write than the read if T2_elements > T1_elements
        #pragma unroll
        for (int o = 0; o < copy_vectors; o++) {
            Packed128<T1> out128;
            #pragma unroll
            for (int k = 0; k < out128.size; k++) {
                // these are tiny 8-bit loads with loads of bank conflicts for FP8
                // extremely hard to avoid and not a bottleneck when everything else is well optimised
                out128[k] = tile[k + (adjusted_tid_x + o * blockDim.x) * out128.size][adjusted_tid_y + j];
            }
            store128<T1>(transposed + x + (o * blockDim.x * out128.size) + (y+j)*height, out128);
        }
    }
}

// only calculate absmax of the input tensor (non-fused)
template <typename T>
__global__ void get_absmax_kernel(unsigned int* absmax_output, const T* inp, size_t N, unsigned int absmax_factor=1) {
    uint absmax_uint = 0;

    size_t idx = ((blockIdx.x * blockDim.x * ABSMAX_ITERATIONS_PER_THREAD) + threadIdx.x) * Packed128<T>::size;
    if (idx < N) {
        for (int i = 0; i < ABSMAX_ITERATIONS_PER_THREAD; i++) {
            Packed128<T> packed_inp = load128(inp + idx);
            for(int k = 0; k < packed_inp.size; ++k) {
                update_local_absmax<true>(absmax_uint, packed_inp[k], absmax_factor);
            }
            idx += blockDim.x * packed_inp.size;
        }
    }
    update_global_absmax<false>(absmax_output, absmax_uint);
}

// ----------------------------------------------------------------------------
// kernel launchers

template <bool reciprocal=true, typename T1, typename T2>
void copy_simple(T1 *copy, const T2 *input, size_t N, float* scale_pointer=NULL, const size_t block_size=512) {
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(CEIL_DIV(N, block_size * fewest_elements));

    if (scale_pointer) {
        copy_simple_kernel<reciprocal, true><<<grid_size, dim3(block_size)>>>(copy, input, N, scale_pointer);
    } else {
        copy_simple_kernel<reciprocal, false><<<grid_size, dim3(block_size)>>>(copy, input, N);
    }
    cudaCheck(cudaGetLastError());
}

template <bool reversed_order=false, elementwise_func_t elementwise_func=nothing_elementwise, bool reciprocal=true, typename T1, typename T2>
void copy_advanced(T1 *copy, const T2 *input, size_t N, float* scale_pointer=NULL, void* absmax_output=NULL, bool memset_absmax=true, const size_t block_size=512) {
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(CEIL_DIV(N, block_size * fewest_elements));
    assert((N % fewest_elements) == 0);

    constexpr uint absmax_factor = 1;
    unsigned int* absmax_uint = (unsigned int*)absmax_output;

    if (absmax_output) {
        if (memset_absmax) {
            cudaMemset(absmax_output, 0, sizeof(unsigned int));
        }
        if (scale_pointer) {
            copy_advanced_kernel<reciprocal, true, reversed_order, elementwise_func, absmax_factor><<<grid_size, dim3(block_size)>>>(copy, input, N, scale_pointer, absmax_uint);
        } else {
            copy_advanced_kernel<reciprocal, false, reversed_order, elementwise_func, absmax_factor><<<grid_size, dim3(block_size)>>>(copy, input, N, NULL, absmax_uint);
        }
    } else {
        if (scale_pointer) {
            copy_advanced_kernel<reciprocal, true, reversed_order, elementwise_func><<<grid_size, dim3(block_size)>>>(copy, input, N, scale_pointer);
        } else {
            copy_advanced_kernel<reciprocal, false, reversed_order, elementwise_func><<<grid_size, dim3(block_size)>>>(copy, input, N);
        }
    }
    cudaCheck(cudaGetLastError());
}

// only 2 important template parameters: write_absmax and elementwise_func
// (use copy_and_transpose() rather than enable_copy=true for clarity)
// slight inefficiency in that we don't optimise away scaling for kernels that don't need it (kernel checks for NULL)
template <bool write_absmax=false, elementwise_func_t elementwise_func=nothing_elementwise, bool reciprocal=true,
          bool enable_copy=false, typename T1, typename T2> // advanced template options, usually don't need to be changed
void transpose(T1 *transposed, const T2 *input, size_t w, size_t h, float* scale_pointer=NULL, void* absmax_output=NULL,
               bool memset_absmax=true, const size_t block_size=64, T1 *copy=NULL) { // advanced parameters
    assert((w % DEFAULT_TILE_SIZE) == 0 && (h % DEFAULT_TILE_SIZE) == 0);
    cudaCheck(cudaGetLastError());

    // printf EVERYTHING for debug
    size_t block_size_x = (DEFAULT_TILE_SIZE * sizeof(T2)) / 16;
    size_t block_size_y = min(DEFAULT_TILE_SIZE, block_size / block_size_x);
    dim3 grid_size(w / DEFAULT_TILE_SIZE, h / DEFAULT_TILE_SIZE);
    dim3 block_size_dim(block_size_x, block_size_y);

    constexpr uint absmax_factor = write_absmax ? 1 : 0;
    unsigned int* absmax_uint = (unsigned int*)absmax_output;
    if (write_absmax && memset_absmax) {
        cudaMemset(absmax_output, 0, sizeof(unsigned int));
    }

    /*
    printf("Transposing %s -> %s, w=%lu, h=%lu, block_size=%lu, enable_copy=%d, write_absmax=%d, elementwise_func=%p\n",
           typeid(T2).name(), typeid(T1).name(), w, h, block_size, enable_copy, write_absmax, elementwise_func);
    printf("transposed=%p, copy=%p, input=%p, scale_pointer=%p, absmax_output=%p, memset_absmax=%d, absmax_factor=%u\n",
           transposed, copy, input, scale_pointer, absmax_output, memset_absmax, absmax_factor);
    printf("sizeof(T1)=%lu, sizeof(T2)=%lu, sizeof(Packed128<T1>)=%lu, sizeof(Packed128<T2>)=%lu\n",
           sizeof(T1), sizeof(T2), sizeof(Packed128<T1>), sizeof(Packed128<T2>));
    printf("block_size_x=%lu, block_size_y=%lu, grid_size=(%d,%d), block_size_dim=(%d,%d)\n",
           block_size_x, block_size_y, grid_size.x, grid_size.y, block_size_dim.x, block_size_dim.y);
    */

    switch (block_size_y) {
        case 32: transpose_kernel<32, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        case 16: transpose_kernel<16, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        case 8: transpose_kernel<8, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        case 4: transpose_kernel<4, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        case 2: transpose_kernel<2, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        case 1: transpose_kernel<1, DEFAULT_TILE_SIZE, reciprocal, enable_copy, true, absmax_factor, elementwise_func><<<grid_size, block_size_dim>>>(transposed, copy, input, scale_pointer, absmax_uint); break;
        default: printf("Invalid block size (might be easy to add): %lu\n", block_size_y); exit(1);
    }
    cudaCheck(cudaGetLastError());
    //printf("===\n");
}

// wrapper so the parameters of the standard transpose function are less messy
template <bool write_absmax=false, elementwise_func_t elementwise_func=nothing_elementwise, bool reciprocal=true, typename T1, typename T2>
void copy_and_transpose(T1 *transposed, T1 *copy, const T2 *input, size_t w, size_t h, float* scale_pointer=NULL, unsigned int* absmax_output=NULL, bool memset_absmax=true, const size_t block_size=64) {
    transpose<write_absmax, elementwise_func, reciprocal, true, T1, T2>(transposed, input, w, h, scale_pointer, absmax_output, memset_absmax, block_size, copy);
}

template <bool write_absmax=false, elementwise_func_t elementwise_func=nothing_elementwise, bool reciprocal=true, typename T1, typename T2>
void copy_or_transpose(bool transposing, T1 *output, const T2 *input, size_t w, size_t h, float* scale_pointer=NULL, unsigned int* absmax_output=NULL, bool memset_absmax=true, const size_t block_size=64) {
    if (transposing) {
        transpose<write_absmax, elementwise_func, reciprocal, false, T1, T2>(output, input, w, h, scale_pointer, absmax_output, memset_absmax, block_size);
    } else {
        copy_advanced<false, elementwise_func, reciprocal>(output, input, w*h, scale_pointer, absmax_output, memset_absmax, block_size);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T>
void get_absmax(void* absmax_output, const T* inp, size_t N, cudaStream_t stream=0,
                bool memset_absmax=true, unsigned int absmax_factor=1, size_t block_size=512) { // advanced parameters

    // find the largest block size that divides N
    while ((N % (block_size * Packed128<T>::size * ABSMAX_ITERATIONS_PER_THREAD)) != 0) {
        block_size /= 2;
        assert(block_size >= 32); // block size of 1 would be OK, but so inefficient we'd rather fail and debug I think
    }
    const dim3 grid_size(CEIL_DIV(N, block_size * ABSMAX_ITERATIONS_PER_THREAD * Packed128<T>::size));
    absmax_factor = absmax_factor ? absmax_factor : 1;
    if (memset_absmax) {
        cudaMemset(absmax_output, 0, sizeof(unsigned int));
    }

    get_absmax_kernel<<<grid_size, dim3(block_size), 0, stream>>>((unsigned int*)absmax_output, inp, N, absmax_factor);
    cudaCheck(cudaGetLastError());
}

// Scratch allocation
// ----------------------------------------------------------------------------
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

public:
    template<typename T>
    static T* getMemory(size_t count) {
        size_t size = count * sizeof(T);

        // Find the smallest free allocation that fits the requested size
        auto it = std::min_element(allocations.begin(), allocations.end(),
            [size](const Allocation& a, const Allocation& b) {
                return !a.in_use && a.size >= size && (b.in_use || b.size < size || a.size < b.size);
            });

        if (it != allocations.end() && !it->in_use && it->size >= size) {
            it->in_use = true;
            return reinterpret_cast<T*>(it->ptr);
        }

        // If no suitable allocation found, create a new one
        void* new_ptr;
        cudaMalloc(&new_ptr, size);
        allocations.emplace_back(new_ptr, size);
        allocations.back().in_use = true;
        return reinterpret_cast<T*>(new_ptr);
    }

    template<typename T>
    static void releaseMemory(T* ptr) {
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
