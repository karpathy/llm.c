/*
Kernels for copy & transpose with format conversion (+ optional elementwise operations, e.g. GELU)
Many parameters are configurable by changing the defines

Compile examples (change 90 to your SM architecture - do not trust performance without it):
nvcc -O3 --generate-code arch=compute_90,code=[compute_90,sm_90] --use_fast_math transpose.cu -o transpose
nvcc -DENABLE_GELU -DIN_TYPE=half -DOUT_TYPE=float -DSCALING_FACTOR=0.5f -DTRANSPOSE_AND_COPY=true -O3 --generate-code arch=compute_90,code=[compute_90,sm_90] --use_fast_math transpose.cu -o transpose

Useful defines (not all options available in all kernels):

IN_TYPE=float (input data type, default is __nv_bfloat16)
OUT_TYPE=half (output data type, default is __nv_fp8_e4m3)
SCALING_FACTOR=0.5f (scaling factor for the output, default is no scaling i.e. 1.0f)
TRANSPOSE_AND_COPY=true (enable extra converted copy of the input tensor for transpose kernels, default is false)
ENABLE_GELU=true (enable GELU elementwise function, default is false)
CALCULATE_ABSMAX=true (calculate absmax of the output tensor pre-scaling, default is false)
ABSMAX_EXPONENT_ONLY=true (round absmax down by clearing all the mantissa bits, default is false)
DEFAULT_TILE=64 (tile size for transpose kernels, this affects shared memory and maximum block size, default=32)
WIDTH=8192 (width of the input tensor, default=8192)
HEIGHT=3072 (height of the input tensor, default=3072)
ABSMAX_ITERATIONS_PER_THREAD=2 (outer loop iterations for absmax kernel 20)

Kernel versions:

version 0 is a non-optimized copy (not a transpose)
version 1 is a simple fast copy similar to version 3, but without all the extra functionality (GELU/absmax/scaling/etc.)
version 2 is a highly optimized copy that tries to keep all loads/stores 128-bit
version 3 is a simpler very optimized copy (with support for absmax calculation)

version 10 is a non-optimized transpose (no elementwise, no absmax)
version 11 is a fast transpose with shared memory (no support for absmax at the moment)
version 12 is a very fast transpose with shared memory and 128-bit loads/stores (with support for absmax calculation)

Usage example: ./transpose 12
*/

#define SKIP_CUBLAS // to save compile time
#include "common.h"
#include <cstring>
#include <cuda_fp8.h>

//#define IN_TYPE half
//#define OUT_TYPE __nv_fp8_e5m2
//#define SCALING_FACTOR 0.3f
//#define TRANSPOSE_AND_COPY true
//#define ENABLE_GELU true
//#define CALCULATE_ABSMAX true
//#define ABSMAX_EXPONENT_ONLY true
//#define DEFAULT_TILE 64UL
//#define WIDTH 8192
//#define HEIGHT 768
//#define ABSMAX_ITERATIONS_PER_THREAD 1
//#define FUSED_RESCALE_IN_PLACE true

#if !defined(IN_TYPE)
#define IN_TYPE __nv_bfloat16
#endif
#if !defined(OUT_TYPE)
#define OUT_TYPE __nv_fp8_e4m3
#endif

#if defined(SCALING_FACTOR)
#define SCALING true
#else
#define SCALING_FACTOR 1.0f
#define SCALING false
#endif

#if !defined(TRANSPOSE_AND_COPY)
#define TRANSPOSE_AND_COPY false
#endif
#if !defined(ENABLE_GELU)
#define ENABLE_GELU false
#endif

#if !defined(CALCULATE_ABSMAX)
#define CALCULATE_ABSMAX false
#endif
#define DEFAULT_ABSMAX_DIVIDER 4
#if CALCULATE_ABSMAX == true
#define ABSMAX_DIVIDER DEFAULT_ABSMAX_DIVIDER
#else
#define ABSMAX_DIVIDER 0
#endif
#if !defined(ABSMAX_EXPONENT_ONLY)
#define ABSMAX_EXPONENT_ONLY false
#endif

#if !defined(DEFAULT_TILE)
#define DEFAULT_TILE 32UL // 32x32 transpose is a good default but 64x64 might be better for absmax
#endif
#if !defined(WIDTH)
#define WIDTH 32768
#endif
#if !defined(HEIGHT)
#define HEIGHT 3072
#endif

#if !defined(ABSMAX_ITERATIONS_PER_THREAD)
#define ABSMAX_ITERATIONS_PER_THREAD 2
#endif

#if !defined(FUSED_RESCALE_IN_PLACE)
#define FUSED_RESCALE_IN_PLACE false // WIP not ready yet
#endif

// ----------------------------------------------------------------------------
// these are passed as default kernel parameters to avoid making everything too messy
unsigned int* d_absmax_estimate = NULL;
unsigned int* d_absmax_counter = NULL;
unsigned int* d_absmax_actual = NULL;
unsigned int absmax_storage = 0;
float* d_scaling_factor = NULL;

// misc. useful constants
constexpr int FIRST_TRANSPOSE_KERNEL = 10; // kernels 0/1/2/3 are copy kernels without transpose
constexpr int FIRST_ABSMAX_ONLY_KERNEL = 20; // kernels 20+ are absmax kernels, they do not copy or transpose

// -----./-----------------------------------------------------------------------
// elementwise functions which can be applied as part of the copy/transpose
// for elementwise kernels that require metadata (e.g. layernorm forward with known mean/std),
// we could maybe store it in constant buffers rather than in yet-another-function-parameter...
using elementwise_func_t = float (*) (float, uint, uint, uint, uint, const void**);
#if ENABLE_GELU == true
#define DEFAULT_ELEMENTWISE gelu_forward_elementwise
#else
#define DEFAULT_ELEMENTWISE nothing_elementwise
#endif

__host__ __device__ float nothing_elementwise(float in, uint x, uint y, uint width, uint height, const void** __restrict__ metadata=NULL) {
    (void)x; (void)y; (void)width; (void)height; (void)metadata; // avoid compiler warnings for unused variables
    return in;
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__host__ __device__ float gelu_forward_elementwise(float in, uint x, uint y, uint width, uint height, const void** __restrict__ metadata=NULL) {
    (void)x; (void)y; (void)width; (void)height; (void)metadata; // avoid compiler warnings for unused variables
    float cube = 0.044715f * in * in * in;
    return 0.5f * in * (1.0f + tanhf(GELU_SCALING_FACTOR * (in + cube)));
}

// ----------------------------------------------------------------------------
// CPU code reference

template <bool scaling=SCALING, typename T1, typename T2>
void transpose_cpu(T1* transposed, T1* transposed_gelu, T1* copy, T1* copy_gelu,
                   const T2* input, size_t width, size_t height, float scaling_factor=SCALING_FACTOR, const void** metadata=NULL) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // note (IN_TYPE) unlike GPU version because T2 is actually always float for simplicity
            float in = (float)((IN_TYPE)input[x + y*width]);
            float gelu = gelu_forward_elementwise(in, x, y, width, height, metadata);

            // absmax calculation is pre-scaling (but has its own ABSMAX_DIVIDER)
            float absmax_divider = (ABSMAX_DIVIDER != 0) ? (float)ABSMAX_DIVIDER : DEFAULT_ABSMAX_DIVIDER;
            #if ENABLE_GELU == true
            float absmax = gelu / (float)absmax_divider;
            #else
            float absmax = in / (float)absmax_divider;
            #endif
            constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
            absmax_storage = max(absmax_storage, *((uint*)&absmax) & absmax_mask);
            #

            if constexpr (scaling) {
                in *= scaling_factor;
                gelu *= scaling_factor;
            }
            transposed[y + x * height] = (T1)in;
            transposed_gelu[y + x * height] = (T1)gelu;
            copy[x + y*width] = (T1)in;
            copy_gelu[x + y*width] = (T1)gelu;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// ----------------------------------------------------------------------------
// This helper is for when we want to copy from e.g. FP32 to BF16
// so if want to load a f128 of 4 elements, and write those 4 elements to memory as 64-bit
// not needed in the case of loads, the compiler will automatically optimise away unused reads
// (we might want to replace this with something like a fixed vector width class though)
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

// to update the absmax for the entire threadgroup (single thread for the global memory atomic)
__device__ void update_absmax(unsigned int* absmax_output, unsigned int absmax_uint) {
    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    // use native integer reductions as much as possible (supported on all GPUs with FP8)
    // todo - we could use cooperative groups instead of PTX here but it'd increase compile time
    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ uint tmp[32];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    if (warp_id != 0) { return; }
    uint shared_idx = (lane_id < blockIdx.x) ? lane_id : 0;

    __syncthreads();
    absmax_uint = tmp[shared_idx];
    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    if (lane_id == 0) {
        atomicMax(absmax_output, absmax_uint);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels for copy

template <bool scaling=SCALING, typename T1, typename T2>
__global__ void copy_naive_kernel0(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N) { return; }
    copy[n] = (T1)((float)input[n] * (scaling ? *scale_pointer : 1.0f));
}

// simplified copy & format conversion kernel using store_same_length
// keeps the largest format at 128-bit and smallest at 32-bit or 64-bit
template <bool scaling=SCALING, typename T1, typename T2>
__global__ void copy_fast_kernel1(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor) {
    // Calculate the *smallest* of the two vector sizes in terms of elements (both are 128-bit if fully used)
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));
    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; }

    // Scaling factor of 1.0f will automatically be optimised away by the compiler
    float scale_factor = scaling ? *scale_pointer : 1.0f;

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a ldg.32 or ldg.64
    Packed128<T2> inp128;
    Packed128<T1> out128;
    inp128 = load128<T2>(input + n);
    for (int k = 0; k < vec_size; k++) {
        out128[k] = (T1)((float)inp128[k] * scale_factor);
    }

    // if sizeof(T2) < sizeof(T1), this will use stg.32 or stg.64 instead of stg.128
    store_same_length<T2,T1>(copy + n, out128);
}

// overly complicated copy & format conversion kernel without store_same_length
// this keeps all loads & stores 128-bit at the cost of more complexity and more register pressure
template <bool scaling=SCALING, elementwise_func_t elementwise_func=DEFAULT_ELEMENTWISE,
          uint absmax_divider=ABSMAX_DIVIDER, typename T1, typename T2>
__global__ void copy_advanced_kernel2(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor, unsigned int* absmax_output=d_absmax_estimate, const void** metadata=NULL) {
    // Optional fused absmax calculation
    uint absmax_uint = 0;

    size_t n = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<T1>::size;
    if (n >= N) { return; }

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a load32 or load64
    Packed128<T2> inp128;
    Packed128<T1> out128;
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    #pragma unroll
    for (int o = 0; o < max(1, out128.size/inp128.size); o++) {
        inp128 = load128cs<T2>(input + n + o*inp128.size);
        #pragma unroll
        for (int k = 0; k < min(inp128.size, out128.size); k++) {
            float out_float = elementwise_func((float)inp128[k], n+o*inp128.size, 0, N, 1, metadata);
            out128[k+o*inp128.size] = (T1)(out_float * (scaling ? scale_factor : 1.0f));

            if constexpr (absmax_divider != 0) { // absmax is calculated before scaling
                constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                absmax_uint = max(absmax_uint, __float_as_uint(out_float / (float)absmax_divider) & absmax_mask);
            }
        }
    }
    store128<T1>(copy + n, out128);

    // update absmax if required
    if constexpr (absmax_divider != 0) {
        update_absmax(absmax_output, absmax_uint);
    }
}

// simplified copy & format conversion kernel using store_same_length
// keeps the largest format at 128-bit and smallest at 32-bit or 64-bit
template <bool reversed_order=false, bool scaling=SCALING, elementwise_func_t elementwise_func=DEFAULT_ELEMENTWISE,
          uint absmax_divider=ABSMAX_DIVIDER, typename T1, typename T2>
__global__ void copy_advanced_kernel3(T1 *copy, const T2 *input, size_t N, const float* __restrict__ scale_pointer=d_scaling_factor, unsigned int* absmax_output=d_absmax_estimate, const void** metadata=NULL) {
    // Optional fused absmax calculation
    uint absmax_uint = 0;
    // Optionally process in reverse order to maximise L2 cache hits across kernels for large tensors
    size_t adjusted_blockidx_x = reversed_order ? (gridDim.x - blockIdx.x - 1) : blockIdx.x;
    // Use the *smallest* of the two vector sizes in terms of elements (both are 128-bit if fully used)
    constexpr size_t vec_size = 16 / ((sizeof(T1) < sizeof(T2)) ? sizeof(T2) : sizeof(T1));
    size_t n = (adjusted_blockidx_x * blockDim.x + threadIdx.x) * vec_size;
    if (n >= N) { return; } // out of bounds check (todo - is this always OK when calculating absmax?)

    // note: if sizeof(T1) < sizeof(T2), compiler will skip unused elements of load128
    // so it may turn out to be a ldg.32 or ldg.64
    Packed128<T2> inp128;
    Packed128<T1> out128;
    inp128 = load128cs<T2>(input + n);
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    for (int k = 0; k < vec_size; k++) {
        float out_float = elementwise_func((float)inp128[k], n+k, 0, N, 1, metadata);
        out128[k] = (T1)(out_float * scale_factor);

        if constexpr (absmax_divider != 0) { // absmax is calculated before scaling
            constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
            absmax_uint = max(absmax_uint, __float_as_uint(out_float / (float)absmax_divider) & absmax_mask);
        }
    }
    // if sizeof(T2) < sizeof(T1), this will use stg.32 or stg.64 instead of stg.128
    store_same_length<T2,T1>(copy + n, out128);

    // update absmax if required
    if constexpr (absmax_divider != 0) {
        update_absmax(absmax_output, absmax_uint);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels for transpose

// naive transpose kernel without shared memory or 128-bit load/store
template <bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY, typename T1, typename T2>
__global__ void transpose_naive_kernel(T1 *transposed, T1* copy, const T2 *input, size_t width, size_t height,
                                       const float* __restrict__ scale_pointer=d_scaling_factor) {
    float scale_factor = scaling ? *scale_pointer : 1.0f;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        T2 in = input[x + y * width];
        T1 out = scaling ? (T1)((float)in * scale_factor) : (T1)in;

        transposed[y + x*height] = out;
        if constexpr (enable_copy) {
            copy[x + y*width] = out;
        }
    }
}

// optimized transpose kernel with shared memory but *without* 128-bit load/store
// originally based on: https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/transpose/transpose.cu
// also see this blog article: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// note that neither of these sources consider less than 32-bit data formats (and associated bank conflicts)
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=DEFAULT_TILE, bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY,
         elementwise_func_t elementwise_func=DEFAULT_ELEMENTWISE, typename T1, typename T2>
__global__ void transpose_kernel1(T1 *transposed, T1 *copy, const T2 *input,
                                  const float* __restrict__ scale_pointer=d_scaling_factor, const void** metadata=NULL)
{
    __shared__ T1 tile[TILE_DIM][TILE_DIM+1]; // +1 for bank conflict avoidance
    int width = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        T2 in = input[x + (y+j)*width];
        float post_elementwise = elementwise_func((float)in, x, y+j, width, height, metadata);
        T1 out = scaling ? (T1)(post_elementwise * scale_factor) : (T1)post_elementwise;

        tile[threadIdx.y+j][threadIdx.x] = out;
        if constexpr (enable_copy) {
            copy[x + (y+j)*width] = out; // separate copy with format conversion (on top of the transpose)
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        // avoiding bank conflicts for 32-bit data types thanks to +1 above
        // (also seems to help sub-32-bit but less so, HW behaviour unclear)
        transposed[x + (y+j)*height] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// more optimized transpose kernel using 128-bit load/store and shared memory
// only slightly faster by default, but much faster with TRANSPOSE_AND_COPY as sub-32-bit store in kernel1 is inefficient
template<size_t BLOCK_ROWS=8UL, size_t TILE_DIM=DEFAULT_TILE, bool scaling=SCALING, bool enable_copy=TRANSPOSE_AND_COPY,
         uint absmax_divider=ABSMAX_DIVIDER, elementwise_func_t elementwise_func=DEFAULT_ELEMENTWISE, typename T1, typename T2>
__global__ void transpose_kernel2(T1* __restrict__ transposed, T1* __restrict__ copy, const T2* __restrict__ input,
                                  const float* __restrict__ scale_pointer=d_scaling_factor, unsigned int* absmax_output=d_absmax_estimate, const void** metadata=NULL)
{
    // Optional fused absmax calculation
    uint absmax_uint = 0;

    // no +1 for bank conflict avoidance because:
    // 1) 128-bit shared memory stores need to be aligned to 128-bit boundaries
    // 2) it doesn't help as much with sub-32-bit data types
    __shared__ T1 tile[TILE_DIM][TILE_DIM];
    int width  = gridDim.x * TILE_DIM;
    int height = gridDim.y * TILE_DIM;

    constexpr size_t T1_elements = 16 / sizeof(T1);
    constexpr size_t T2_elements = 16 / sizeof(T2);
    constexpr size_t copy_len = (sizeof(T1) >= sizeof(T2)) ? (sizeof(T1) / sizeof(T2)) : 1;

    float scale_factor = scaling ? *scale_pointer : 1.0f;
    int x = blockIdx.x * TILE_DIM + (threadIdx.x * T2_elements);
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        Packed128<T2> in128 = load128<T2>(input + x + (y+j)*width);
        Packed128<T1> copy128[copy_len];
        #pragma unroll
        for (int k = 0; k < in128.size; k++) {
            T2 in = in128[k];
            float out_float = elementwise_func((float)in, x+k, y+j, width, height, metadata);
            T1 out = (T1)(out_float * scale_factor);
            copy128[k/T1_elements][k%T1_elements] = out; // optimised away by compiler if unused

            if constexpr (absmax_divider != 0) { // absmax is calculated before scaling
                constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                absmax_uint = max(absmax_uint, __float_as_uint(out_float / (float)absmax_divider) & absmax_mask);
            }
        }

        #pragma unroll
        for (int o = 0; o < copy_len; o++) {
            if constexpr (enable_copy) {
                store_same_length<T2,T1>(copy + x + (y+j)*width + o*T1_elements, copy128[o]);
            }
            size_t tile_offset = (threadIdx.x * T2_elements) + (threadIdx.y+j)*TILE_DIM + o*T1_elements;
            store_same_length<T2,T1>(&tile[0][0] + tile_offset, copy128[o]);
        }
    }
    uint tid = threadIdx.x + threadIdx.y*blockDim.x;
    uint lane_id = tid % 32;
    uint warp_id = tid / 32;
    __shared__ uint tmp_absmax[32];

    if constexpr (absmax_divider != 0) {
        // use native integer reductions as much as possible (supported on all GPUs with FP8)
        // todo - we could use cooperative groups instead of PTX here but it'd increase compile time
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            tmp_absmax[warp_id] = absmax_uint;
        }
    }
    __syncthreads();

    // reduce the number of threads for the write if T1_elements > T2_elements
    // we want to keep all 32 threads in a warp active, so we try to eliminate in y dimension first
    // so we create fake/adjusted tid.x/tid.y where "extra" threadIdx.x adds to the effective tid.y
    constexpr size_t block_size_x = (DEFAULT_TILE * sizeof(T2)) / 16;
    constexpr size_t block_size_y = BLOCK_ROWS;
    constexpr size_t num_warps = (block_size_x * block_size_y + 31) / 32;

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
        for (int o = 0; o < copy_len; o++) {
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

    if constexpr (absmax_divider != 0) {
        // reduce the absmax for the entire threadgroup then do the atomicMax to global memory
        if (warp_id != 0) { return; }
        absmax_uint = tmp_absmax[lane_id % num_warps];
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            atomicMax(absmax_output, absmax_uint);
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels for absmax

// kernel to calculate absmax of the input tensor
template <int NUM_WARPS=32, typename T=IN_TYPE>
__global__ void get_absmax_kernel(const T* inp, unsigned int* absmax_scalar, size_t N, float absmax_divider=1.0f) {
    size_t idx = ((blockIdx.x * blockDim.x * ABSMAX_ITERATIONS_PER_THREAD) + threadIdx.x) * Packed128<T>::size;
    uint absmax_uint = 0;

    if (idx < N) {
        #pragma unroll
        for (int i = 0; i < ABSMAX_ITERATIONS_PER_THREAD; i++) {
            Packed128<T> packed_inp = load128(inp + idx);
            for(int k = 0; k < packed_inp.size; ++k) {
                constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                uint x = __float_as_uint((float)packed_inp[k] / absmax_divider) & absmax_mask;
                absmax_uint = max(absmax_uint, x);
            }
            idx += blockDim.x * packed_inp.size;
        }
    }
    // Use inline PTX for redux.sync.max.u32
    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ uint tmp[NUM_WARPS];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();
    if (warp_id == 0) {
        absmax_uint = tmp[lane_id % NUM_WARPS];
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            atomicMax(absmax_scalar, absmax_uint);
        }
    }
}

template <int NUM_WARPS=32, typename T=IN_TYPE>
__global__ void get_absmax_persistent_kernel(const T* inp, unsigned int* absmax_scalar, size_t N, float absmax_divider=1.0f) {
    int elements_per_block_iteration = blockDim.x * ABSMAX_ITERATIONS_PER_THREAD * Packed128<T>::size;
    int iterations_per_block = (int)ceil_div(N, (size_t)(gridDim.x * elements_per_block_iteration));
    size_t start_idx = blockIdx.x * elements_per_block_iteration * iterations_per_block;
    uint absmax_uint = 0;

    for (size_t i = 0; i < iterations_per_block; i++) {
        size_t idx = start_idx + (i * elements_per_block_iteration) + (threadIdx.x * Packed128<T>::size);
        if (idx < N) {
            for (int o = 0; o < ABSMAX_ITERATIONS_PER_THREAD; o++) {
                Packed128<T> packed_inp = load128(inp + idx);
                for(int k = 0; k < packed_inp.size; ++k) {
                    constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                    uint x = __float_as_uint((float)packed_inp[k] / absmax_divider) & absmax_mask;
                    absmax_uint = max(absmax_uint, x);
                }
                idx += blockDim.x * packed_inp.size;
            }
        }
    }

    // Use inline PTX for redux.sync.max.u32
    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ uint tmp[NUM_WARPS];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();
    if (warp_id == 0) {
        absmax_uint = tmp[lane_id % NUM_WARPS];
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            atomicMax(absmax_scalar, absmax_uint);
        }
    }
}

#define FUSED_ABSMAX_FIRST_PHASE_BYTES 40 * 1024 * 1024
#define FUSED_ABSMAX_CONSERVATIVE_FUDGE_FACTOR 8.0f

template <int NUM_WARPS=32, typename T=IN_TYPE, typename TOut=OUT_TYPE>
__global__ void __launch_bounds__(1024, 2) fused_absmax_scale_persistent(TOut* __restrict__ out, unsigned int* __restrict__ absmax_scaling, unsigned int* __restrict__ absmax_actual, unsigned int* __restrict__ absmax_counter,
                                                                         const T* inp, size_t N, float absmax_divider=1.0f) {
    int elements_per_block_iteration = blockDim.x * ABSMAX_ITERATIONS_PER_THREAD * Packed128<T>::size;
    int iterations_per_block = (int)ceil_div(N, (size_t)(gridDim.x * elements_per_block_iteration));
    size_t start_idx = blockIdx.x * elements_per_block_iteration * iterations_per_block;
    unsigned int absmax_uint = 0;

    // iterations to get to 40MiB read
    int bytes_per_iteration = gridDim.x * elements_per_block_iteration * sizeof(T);
    int iterations_for_first_phase = min(iterations_per_block, 1 + (FUSED_ABSMAX_FIRST_PHASE_BYTES / bytes_per_iteration));

    Packed128<T> packed_inp[ABSMAX_ITERATIONS_PER_THREAD];
    for (int i = iterations_for_first_phase-1; i >= 0; i--) {
        size_t idx = start_idx + (i * elements_per_block_iteration) + (threadIdx.x * Packed128<T>::size);
        if (idx < N) {
            for (int o = 0; o < ABSMAX_ITERATIONS_PER_THREAD; o++) {
                packed_inp[o] = (i == 0) ? load128cs(inp + idx) : load128(inp + idx); // i=0 is cached in registers, so do not keep in L2
                for(int k = 0; k < packed_inp[o].size; ++k) {
                    constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                    uint x = __float_as_uint((float)packed_inp[o][k] / absmax_divider) & absmax_mask;
                    absmax_uint = max(absmax_uint, x);
                }
                idx += blockDim.x * packed_inp[o].size;
            }
        }
    }

    // Use inline PTX for redux.sync.max.u32
    uint lane_id = threadIdx.x % 32;
    uint warp_id = threadIdx.x / 32;

    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    __shared__ unsigned int tmp[NUM_WARPS];
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();
    if (warp_id == 0) {
        absmax_uint = tmp[lane_id % NUM_WARPS];

        // apply fudge factor to reduce overflow (doesn't affect absmax_uint which is used for the "real" absmax later)
        float absmax_tmp = __uint_as_float(absmax_uint);
        absmax_tmp *= FUSED_ABSMAX_CONSERVATIVE_FUDGE_FACTOR;

        // global memory atomicMax (the compiler seems to automatically add a redux.sync.max to optimise this)
        atomicMax(absmax_scaling, __float_as_uint(absmax_tmp));

        // increment the number of blocks done with phase 1
        __threadfence(); // to make sure the atomicInc always happens after the atomicMax
        if (lane_id == 0) {
            atomicInc(absmax_counter, gridDim.x-1);
        }
    }
    __syncthreads();

    // Prefetch the very start of the next iteration so the DRAM controller has something to do
    size_t idx_prefetch = start_idx + (iterations_for_first_phase * elements_per_block_iteration) + (threadIdx.x * Packed128<T>::size);
    if (idx_prefetch < N) {
        asm volatile("prefetch.global.L1 [%0];" :: "l"(inp + idx_prefetch));
    }

    // Wait until all blocks have incremented the counter indicating they are done with phase 1
    if (threadIdx.x == 0) {
        // volatile read of absmax_counter: wait until it is reset to 0 (because val = gridDim.x-1)
        bool done = (__ldcg(absmax_counter) == 0);
        while (!done) {
            __nanosleep(100); // sleep for 100 nanoseconds, i.e. 200 cycles at 2GHz, then retry
            done = (__ldcg(absmax_counter) == 0);
        }
        tmp[0] = __ldcg(absmax_scaling);
    }
    __syncthreads();
    unsigned int absmax_uint_used = tmp[0];
    float estimated_absmax = __uint_as_float(absmax_uint_used);

    // Prefetch the 2nd part of the next iteration so the DRAM controller has something to do
    if (idx_prefetch < N && ABSMAX_ITERATIONS_PER_THREAD >= 2) {
        asm volatile("prefetch.global.L1 [%0];" :: "l"(inp + idx_prefetch));
    }

    // Now we can do the actual scaling for the 1st iteration which we cached in the packed_inp registers
    int i = 0;
    size_t idx = start_idx + (i * elements_per_block_iteration) + (threadIdx.x * Packed128<T>::size);
    if (idx < N) {
        for (int o = 0; o < ABSMAX_ITERATIONS_PER_THREAD; o++) {
            Packed128<TOut> packed_out;
            for(int k = 0; k < packed_inp[o].size; ++k) {
                packed_out[k] = (TOut)((float)packed_inp[o][k] / estimated_absmax);
            }
            store_same_length<T,TOut>(out + idx, packed_out);
            idx += blockDim.x * packed_inp[o].size;
        }
    }

    // We do the scaling for everything else, while keeping track of the absmax
    // if the absmax no longer matches, stop copying and only calculate absmax, then copy everything at the end
    for (int i = 1; i < iterations_per_block; i++) {
        size_t idx = start_idx + (i * elements_per_block_iteration) + (threadIdx.x * Packed128<T>::size);
        if (idx < N) {
            for (int o = 0; o < ABSMAX_ITERATIONS_PER_THREAD; o++) {
                Packed128<TOut> packed_out;
                Packed128<T> packed_inp = load128cs(inp + idx); // last read, do not cache in either L1 or L2
                for(int k = 0; k < packed_inp.size; ++k) {
                    constexpr uint absmax_mask = ABSMAX_EXPONENT_ONLY ? 0x7f800000 : 0x7fffffff;
                    uint x = __float_as_uint((float)packed_inp[k] / absmax_divider) & absmax_mask;
                    absmax_uint = max(absmax_uint, x);
                    packed_out[k] = (TOut)((float)packed_inp[k] / estimated_absmax);
                }
                store_same_length<T,TOut>(out + idx, packed_out);
                idx += blockDim.x * packed_inp.size;
            }
        }
    }

    // Now update the global max and wait until all other blocks are done to recheck the absmax
    asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
    if (lane_id == 0) {
        tmp[warp_id] = absmax_uint;
    }
    __syncthreads();
    if (warp_id == 0) {
        absmax_uint = tmp[lane_id % NUM_WARPS];
        atomicMax(absmax_actual, absmax_uint);
    }

    // todo - this is a WIP path that rescales the tensor in-place
    // right now, this will result in overflowed values just being scaled down, which is obviously not what we want
    // it would require separate metadata to track the scaling factor used for each part of the tensor
    // or just stop scaling as soon as we detect a value that is too big locally, and read BF16 version here instead
    #if FUSED_RESCALE_IN_PLACE == true
    if (warp_id == 0) {
        if (threadIdx.x == 0) {
            unsigned int old = atomicInc(absmax_counter, gridDim.x-1);
            bool done = (old == gridDim.x-1);
            while (!done) {
                __nanosleep(100); // sleep for 100 nanoseconds, i.e. 200 cycles at 2GHz, then retry
                done = (__ldcg(absmax_counter) == 0);
            }
            tmp[0] = __ldcg(absmax_actual);
        }
    }
    __syncthreads();

    absmax_uint = tmp[0];
    float final_absmax = __uint_as_float(absmax_uint);

    // todo - this wastes half the warps in the BF16->FP8 case, we can do better than this!
    if (threadIdx.x >= (blockDim.x * Packed128<T>::size) / Packed128<TOut>::size) { return; }
    if (final_absmax <= estimated_absmax) { return; }

    // We need to rescale the entire tensor! :(
    // We scale the FP8 tensor in-place by a power of 2 so it only affects the exponent bits
    // (except for subnormals and special numbers, but because this is a persistent kernel, it's still deterministic)
    float ratio = final_absmax / estimated_absmax;
    float ratio_power_of_2 = exp2f(ceil(__log2f(ratio)));
    float scale = 1.0f / ratio_power_of_2;

    for (int i = iterations_per_block-1; i >= 0; i--) {
        size_t idx = start_idx + (i * elements_per_block_iteration) + (threadIdx.x * Packed128<TOut>::size);
        if (idx < N) {
            for (int o = 0; o < ABSMAX_ITERATIONS_PER_THREAD; o++) {
                Packed128<TOut> packed_in_out = load128(out + idx);
                for(int k = 0; k < Packed128<TOut>::size; ++k) {
                    packed_in_out[k] = (TOut)((float)packed_in_out[k] * scale);
                }
                store128cs<TOut>(out + idx, packed_in_out);
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float rescaled_absmax = estimated_absmax * ratio_power_of_2;
        *absmax_scaling = __float_as_uint(rescaled_absmax);
    }
    #endif
}


// ----------------------------------------------------------------------------
// kernel launchers

template <typename T1, typename T2>
void copy_naive_0(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size));
    copy_naive_kernel0<<<grid_size, dim3(block_size)>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy_fast_1(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(ceil_div(N, block_size * fewest_elements));
    copy_fast_kernel1<<<grid_size, dim3(block_size)>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy_advanced_2(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    const dim3 grid_size(ceil_div(N, block_size * 16 / sizeof(T1)));
    copy_advanced_kernel2<<<grid_size, dim3(block_size)>>>(copy, input, N);
}

template <typename T1, typename T2>
void copy_advanced_3(T1 *copy, const T2 *input, size_t width, size_t height, const size_t block_size) {
    size_t N = width * height;
    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size(ceil_div(N, block_size * fewest_elements));
    copy_advanced_kernel3<<<grid_size, dim3(block_size)>>>(copy, input, N);
}

template <typename T1, typename T2>
void transpose_naive(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    // actual block size is sqrt(block_size) rounded to next power of 2 (so 128 is really 256 unfortunately...)
    size_t actual_block_size = 1 << (int)ceil(log2(sqrt(block_size)));
    const dim3 grid_size(ceil_div(width, block_size), ceil_div(height, block_size));
    transpose_naive_kernel<<<grid_size, dim3(actual_block_size)>>>(transposed, copy, input, width, height);
}

template <typename T1, typename T2>
void transpose1(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    dim3 grid_size(width / DEFAULT_TILE, height / DEFAULT_TILE);
    dim3 block_size_(DEFAULT_TILE, max(1UL, block_size / DEFAULT_TILE)); // always >=1, so might not respect block size for large tiles

    switch (block_size_.y) {
        case 32: transpose_kernel1<32, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 16: transpose_kernel1<16, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 8: transpose_kernel1<8, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 4: transpose_kernel1<4, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 2: transpose_kernel1<2, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 1: transpose_kernel1<1, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        default: printf("Invalid block size: %d\n", block_size_.y); exit(1);
    }
}

template <typename T1, typename T2>
void transpose2(T1 *transposed, const T2 *input, size_t width, size_t height, const size_t block_size, T1 *copy=NULL) {
    size_t block_size_x = (DEFAULT_TILE * sizeof(T2)) / 16;
    size_t block_size_y = min(DEFAULT_TILE, block_size / block_size_x);
    dim3 grid_size(width / DEFAULT_TILE, height / DEFAULT_TILE);
    dim3 block_size_(block_size_x, block_size_y);

    switch (block_size_y) {
        case 128: transpose_kernel2<128, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 64: transpose_kernel2<64, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 32: transpose_kernel2<32, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 16: transpose_kernel2<16, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 8: transpose_kernel2<8, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 4: transpose_kernel2<4, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 2: transpose_kernel2<2, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        case 1: transpose_kernel2<1, DEFAULT_TILE, SCALING, TRANSPOSE_AND_COPY><<<grid_size, block_size_>>>(transposed, copy, input); break;
        default: printf("Invalid block size: %lu\n", block_size_y); exit(1);
    }
}

template <typename T>
void get_absmax(const T* input, size_t N, const size_t block_size, bool memset=true, unsigned int* absmax_output=d_absmax_estimate, float absmax_divider=(float)ABSMAX_DIVIDER) {
    size_t grid_size = ceil_div(N, block_size * x128::size * ABSMAX_ITERATIONS_PER_THREAD);
    absmax_divider = absmax_divider ? absmax_divider : (float)DEFAULT_ABSMAX_DIVIDER;
    //assert((N % (Packed128<T>::size * ABSMAX_ITERATIONS_PER_THREAD)) == 0);

    if (memset) {
        cudaMemset(absmax_output, 0, sizeof(unsigned int));
    }

    switch (block_size) {
        case 32: get_absmax_kernel<1><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 64: get_absmax_kernel<2><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 128: get_absmax_kernel<4><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 256: get_absmax_kernel<8><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 512: get_absmax_kernel<16><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 768: get_absmax_kernel<24><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 1024: get_absmax_kernel<32><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        default: printf("Invalid block size: %lu\n", block_size); exit(1);
    }
    cudaCheck(cudaGetLastError());
}

template <bool reversed_copy=false, typename T1, typename T2>
void absmax_and_copy(T1* copy, const T2* input, size_t N, const size_t block_size, bool memset=true, unsigned int* absmax_output=d_absmax_estimate, float absmax_divider=(float)ABSMAX_DIVIDER) {
    get_absmax(input, N, block_size, false, absmax_output, absmax_divider);

    size_t fewest_elements = min(Packed128<T1>::size, Packed128<T2>::size);
    const dim3 grid_size_copy(ceil_div(N, block_size * fewest_elements));

    copy_advanced_kernel3<reversed_copy><<<grid_size_copy, dim3(block_size)>>>(copy, input, N);
    cudaCheck(cudaGetLastError());
}

template <typename T>
void get_absmax_persistent(const T* input, size_t N, const size_t block_size, bool memset=true, unsigned int* absmax_output=d_absmax_estimate, float absmax_divider=(float)ABSMAX_DIVIDER) {
    size_t grid_size = 114 * (2048 / block_size);
    absmax_divider = absmax_divider ? absmax_divider : (float)DEFAULT_ABSMAX_DIVIDER;
    //assert((N % (Packed128<T>::size * ABSMAX_ITERATIONS_PER_THREAD)) == 0);

    if (memset) {
        cudaMemset(absmax_output, 0, sizeof(unsigned int));
    }

    switch (block_size) {
        case 32: get_absmax_persistent_kernel<1><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 64: get_absmax_persistent_kernel<2><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 128: get_absmax_persistent_kernel<4><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 256: get_absmax_persistent_kernel<8><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 512: get_absmax_persistent_kernel<16><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 768: get_absmax_persistent_kernel<24><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        case 1024: get_absmax_persistent_kernel<32><<<grid_size, block_size>>>(input, absmax_output, N, absmax_divider); break;
        default: printf("Invalid block size: %lu\n", block_size); exit(1);
    }
    cudaCheck(cudaGetLastError());
}

template <typename T1, typename T2>
void fused_absmax_scale_persistent(T1* out, const T2* input, size_t N, const size_t block_size, bool memset=true, float absmax_divider=(float)ABSMAX_DIVIDER) {
    size_t grid_size = 114 * min(32, (int)(2048 / block_size)); // maximum of 32 blocks in flight
    absmax_divider = absmax_divider ? absmax_divider : (float)DEFAULT_ABSMAX_DIVIDER;
    //assert((N % (Packed128<T1>::size * ABSMAX_ITERATIONS_PER_THREAD)) == 0);

    if (memset) {
        cudaMemset(d_absmax_estimate, 0, sizeof(unsigned int));
        cudaMemset(d_absmax_counter, 0, sizeof(unsigned int));
        cudaMemset(d_absmax_actual, 0, sizeof(unsigned int));
    }

    // todo - ideally this should use cooperative thread launches so that the CUDA API itself guarantees all blocks can execute simultaneously
    switch (block_size) {
        case 32: fused_absmax_scale_persistent<1><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 64: fused_absmax_scale_persistent<2><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 128: fused_absmax_scale_persistent<4><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 256: fused_absmax_scale_persistent<8><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 512: fused_absmax_scale_persistent<16><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 768: fused_absmax_scale_persistent<24><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        case 1024: fused_absmax_scale_persistent<32><<<grid_size, block_size>>>(out, d_absmax_estimate, d_absmax_actual, d_absmax_counter, input, N, absmax_divider); break;
        default: printf("Invalid block size: %lu\n", block_size); exit(1);
    }
    cudaCheck(cudaGetLastError());
}


// kernel version dispatch
template <typename T1, typename T2>
void run_advanced(int kernel_num,
                  T1 *transposed, T1 *copy, const T2 *input,
                  size_t width, size_t height, size_t block_size) {
    switch (kernel_num) {
        case 0:
            copy_naive_0(copy, input, width, height, block_size);
            break;
        case 1:
            copy_fast_1(copy, input, width, height, block_size);
            break;
        case 2:
            copy_advanced_2(copy, input, width, height, block_size);
            break;
        case 3:
            // our best copy with the most features
            copy_advanced_3(copy, input, width, height, block_size);
            break;
        case 10:
            transpose_naive(transposed, input, width, height, block_size, copy);
            break;
        case 11:
            transpose1(transposed, input, width, height, block_size, copy);
            break;
        case 12:
            // our best transpose with the most features
            transpose2(transposed, input, width, height, block_size, copy);
            break;
        case 20:
            get_absmax(input, width * height, block_size, true);
            break;
        case 21:
            // no memset (negligible difference except for tiny tensors)
            get_absmax(input, width * height, block_size, false);
            break;
        case 22:
            get_absmax_persistent(input, width * height, block_size, true);
            break;
        case 23:
            absmax_and_copy(copy, input, width * height, block_size, true);
            break;
        case 24:
            // reversed copy which leads to some L2 cache hits on copy after absmax
            absmax_and_copy<true>(copy, input, width * height, block_size, true);
            break;
        case 30:
            fused_absmax_scale_persistent(copy, input, width * height, block_size, true);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();
    int W = WIDTH;
    int H = HEIGHT;

    // create host memory of random numbers (0 to 1 so there's no overflow with format conversion)
    OUT_TYPE* transposed = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* copy = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* out = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* copy_gelu = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    OUT_TYPE* transposed_gelu = (OUT_TYPE*)malloc(W * H * sizeof(OUT_TYPE));
    float* input = make_random_float_01(W * H);

    // add an outlier towards the end to make the job of fused absmax really hard
    input[(W/7) + ((H*4)/5)*W] = 435.0f;

    // read kernel_num from command line
    int kernel_num = 12;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    transpose_cpu(transposed, transposed_gelu, copy, copy_gelu, input, W, H);

    // move to GPU
    IN_TYPE *d_input;
    OUT_TYPE *d_transposed, *d_copy;
    cudaCheck(cudaMalloc(&d_transposed, W * H * sizeof(OUT_TYPE)));
    cudaCheck(cudaMalloc(&d_copy, W * H * sizeof(OUT_TYPE)));
    cudaCheck(cudaMalloc(&d_input, W * H * sizeof(IN_TYPE)));
    cudaCheck(memcpy_convert(d_input, input, W * H));

    float scaling_factor = SCALING_FACTOR;
    cudaCheck(cudaMalloc(&d_scaling_factor, sizeof(float)));
    cudaCheck(cudaMemcpy(d_scaling_factor, &scaling_factor, sizeof(float), cudaMemcpyHostToDevice));

    cudaCheck(cudaMalloc(&d_absmax_estimate, sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&d_absmax_counter, sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&d_absmax_actual, sizeof(unsigned int)));
    cudaCheck(cudaMemset(d_absmax_estimate, 0, sizeof(unsigned int)));
    cudaCheck(cudaMemset(d_absmax_counter, 0, sizeof(unsigned int)));
    cudaCheck(cudaMemset(d_absmax_actual, 0, sizeof(unsigned int)));

    // time the kernel at different block sizes
    int block_sizes[] = {1024,512,256};

    // kernel 12 specifically does not support all block sizes, so act accordingly
    size_t num_block_sizes = sizeof(block_sizes) / sizeof(int);
    if (kernel_num == 12) {
        size_t block_size_x = (DEFAULT_TILE * sizeof(OUT_TYPE)) / 16;
        size_t block_size_y = min(DEFAULT_TILE, 1024 / block_size_x);
        size_t max_block_size = block_size_y * block_size_x;
        while (block_sizes[num_block_sizes - 1] > max_block_size && num_block_sizes >= 1) {
            num_block_sizes--;
        }
    }

    bool enable_gelu = ENABLE_GELU && kernel_num != 0 && kernel_num != 1 && kernel_num != 10;
    bool enable_absmax = CALCULATE_ABSMAX && kernel_num != 0 && kernel_num != 1 && kernel_num != 10 && kernel_num != 11;

    #if ENABLE_GELU == true
    if (!enable_gelu)
        printf("WARNING: This kernel does not support GELU calculation.\n");
    #endif
    #if CALCULATE_ABSMAX == true
    if (!enable_absmax)
        printf("WARNING: This kernel does not support absmax calculation.\n");
    #endif

    for (int j = 0; j < num_block_sizes; j++) {
        printf("Checking block size %d.\n", block_sizes[j]);
        run_advanced(kernel_num, d_transposed, d_copy, d_input, W, H, block_sizes[j]);

        if (kernel_num < FIRST_ABSMAX_ONLY_KERNEL) {
            // check copy tensor for copy kernels & for all others in +copy mode
            if (kernel_num < FIRST_TRANSPOSE_KERNEL || TRANSPOSE_AND_COPY == true) {
                if (enable_gelu) {
                    validate_result(d_copy, copy_gelu, "copy_gelu", W * H, (OUT_TYPE)1e-5f);
                } else {
                    validate_result(d_copy, copy, "copy", W * H, (OUT_TYPE)1e-5f);
                }
            }

            // check transposed tensor for transpose kernels
            if (kernel_num >= FIRST_TRANSPOSE_KERNEL) {
                if (enable_gelu) {
                    validate_result(d_transposed, transposed_gelu, "transposed_gelu", W * H, (OUT_TYPE)1e-5f);
                } else {
                    validate_result(d_transposed, transposed, "transposed", W * H, (OUT_TYPE)1e-5f);
                }
            }
        }

        // check absmax if it was calculated
        if (enable_absmax || kernel_num >= FIRST_ABSMAX_ONLY_KERNEL) {
            if (kernel_num != 30) { // don't check for the WIP fused absmax kernel yet
                validate_result((float*)d_absmax_estimate, (float*)&absmax_storage, "absmax", 1, 1e-5f);
            }
        }
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < num_block_sizes; j++) {
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, run_advanced<OUT_TYPE, IN_TYPE>,
                                              kernel_num, d_transposed, d_copy, d_input,
                                              W, H, block_sizes[j]);

        // napkin math: estimate the memory bandwidth achieved
        size_t memory_ops = W * H * (sizeof(IN_TYPE) + sizeof(OUT_TYPE));
        #if TRANSPOSE_AND_COPY == true
        if (kernel_num >= FIRST_TRANSPOSE_KERNEL) {
            memory_ops += W * H * sizeof(OUT_TYPE);
        }
        #endif
        if (kernel_num >= FIRST_ABSMAX_ONLY_KERNEL && kernel_num != 30) {
            if (kernel_num < 23) {
                memory_ops = 0; // 20/21/22 only do the absmax, no copy
            }
            memory_ops += W * H * sizeof(IN_TYPE); // read-only absmax kernel (+copy for 22/23)
        }
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;
        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_sizes[j], elapsed_time, memory_bandwidth);
    }

    free(out);
    free(copy);
    free(input);
    free(transposed);
    free(copy_gelu);
    free(transposed_gelu);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_copy));
    cudaCheck(cudaFree(d_transposed));
    cudaCheck(cudaFree(d_scaling_factor));
}