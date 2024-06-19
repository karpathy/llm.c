// Utilities for use in __device__ code

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
template <typename T=float>
__device__ inline T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
// warp-level reduction for finding the minimum value
template <typename T=float>
__device__ inline T warpReduceMin(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T shuffled = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = val < shuffled ? val : shuffled;
    }
    return val;
}
// warp-level reduction for finding the maximum value
template <typename T=float>
__device__ inline T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T shuffled = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = val > shuffled ? val : shuffled;
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
template<typename T=float, T (*warp_reduction)(T)>
__device__ inline T blockReduce(T val, bool final_sync=true, T out_of_bounds=0) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ T shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    T warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    T block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ constexpr unsigned int SquirrelNoise5(int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = (unsigned int) positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
    return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    // makes sure each thread gets a different random number
    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

#include <float.h> // todo - needed for FLT_MAX, but already included elsewhere later

constexpr int STATS_NUM_HISTOGRAM_BINS  = 256;
constexpr int STATS_HISTOGRAM           = 0;
constexpr int STATS_OCP_HISTOGRAM       = STATS_HISTOGRAM+STATS_NUM_HISTOGRAM_BINS;
constexpr int STATS_DETAILED_HISTOGRAM  = STATS_OCP_HISTOGRAM+STATS_NUM_HISTOGRAM_BINS;
constexpr int STATS_ABSMIN              = STATS_DETAILED_HISTOGRAM+(STATS_NUM_HISTOGRAM_BINS*4);
constexpr int STATS_ABSMAX              = STATS_ABSMIN+1;
constexpr int STATS_MAXEXP              = STATS_ABSMAX+1;
constexpr int STATS_MINEXP              = STATS_MAXEXP+1;
constexpr int STATS_ZEROS               = STATS_MINEXP+1;
constexpr int STATS_NEGVAL              = STATS_ZEROS+1;
constexpr int STATS_POSVAL              = STATS_NEGVAL+1;
constexpr int STATS_INF                 = STATS_POSVAL+1;
constexpr int STATS_NAN                 = STATS_INF+1;
constexpr int STATS_SUM                 = STATS_NAN+1;
constexpr int STATS_ABSSUM              = STATS_SUM+1;
constexpr int STATS_VARIANCE            = STATS_ABSSUM+1;
constexpr int STATS_TENSOR_DIM          = STATS_VARIANCE+1;
constexpr int ANALYSIS_SIZE             = STATS_TENSOR_DIM+1;

constexpr int STATS_ELEMENTS_PER_THREAD = 8;

// Calculate analysis of exponent bits (8 bins) for all elements in the input matrix + ...
// templated
template<typename T>
__global__ void analysis_kernel(uint* analysis_gmem, const T* input, size_t n) {
    __shared__ unsigned int analysis[ANALYSIS_SIZE];
    // Set everything to 0
    for (int i = threadIdx.x; i < ANALYSIS_SIZE; i += blockDim.x) {
        analysis[i] = 0;
    }
    __syncthreads();

    // Local variables for each thread, e.g. absmin, absmax, etc.
    float absmin = FLT_MAX;
    float absmax = 0.0f;
    uint   maxexp = 0;
    uint   minexp = 255;
    uint   zeros  = 0;
    uint   negval = 0;
    uint   posval = 0;
    uint   inf    = 0;
    uint   nan    = 0;
    float sum    = 0.0f;

    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * STATS_ELEMENTS_PER_THREAD;
    for (int i = 0; i < STATS_ELEMENTS_PER_THREAD; i++) {
        if (i+idx >= n) {
            break;
        }
        const float x = (float)input[i+idx];
        uint exponent = ((*(uint*)&x) >> 23) & 0xff;
        bool mantissa_msb_1 = (*(uint*)&x) & 0x00400000;

        // regular histogram
        atomicInc(analysis + STATS_HISTOGRAM + exponent, 1U<<31U);
        // detailed histogram
        uint detailed_bin = exponent*4 + (mantissa_msb_1 ? 1 : 0) + ((x >= 0.0f) ? 2 : 0);
        atomicInc(analysis + STATS_DETAILED_HISTOGRAM + detailed_bin, 1U<<31U);

        absmin = (x != 0.0f) ? fminf(absmin, fabsf(x)) : absmin;
        absmax = fmaxf(absmax, fabsf(x));
        maxexp = maxexp < 255 ? max(maxexp, exponent) : maxexp;
        minexp = (x != 0.0f) ? min(minexp, exponent) : minexp;
        zeros += (x == 0.0f) ? 1 : 0;
        negval += (x < 0.0f) ? 1 : 0;
        posval += (x > 0.0f) ? 1 : 0;
        inf    += (exponent == 255 && (*(uint*)&x & 0x7fffff) == 0);
        nan    += (exponent == 255 && (*(uint*)&x & 0x7fffff) != 0);
        sum    += x;
    }
    __syncthreads();

    // Reduce histogram to global memory
    #pragma unroll
    for (int i = threadIdx.x; i < STATS_NUM_HISTOGRAM_BINS; i += blockDim.x) {
        uint increment = analysis[i];
        if (increment != 0) {
            atomicAdd(analysis_gmem + i, increment);
        }
    }

    #pragma unroll
    for (int i = threadIdx.x; i < STATS_NUM_HISTOGRAM_BINS*4; i += blockDim.x) {
        uint increment = analysis[STATS_DETAILED_HISTOGRAM + i];
        if (increment != 0) {
            atomicAdd(analysis_gmem + STATS_DETAILED_HISTOGRAM + i, increment);
        }
    }
    absmin = blockReduce<float, warpReduceMin>(absmin);
    absmax = blockReduce<float, warpReduceMax>(absmax);
    maxexp = blockReduce<uint, warpReduceMax>(maxexp);
    minexp = blockReduce<uint, warpReduceMin>(minexp);
    zeros  = blockReduce<uint, warpReduceSum>(zeros);
    negval = blockReduce<uint, warpReduceSum>(negval);
    posval = blockReduce<uint, warpReduceSum>(posval);
    inf    = blockReduce<uint, warpReduceSum>(inf);
    nan    = blockReduce<uint, warpReduceSum>(nan);
    sum    = blockReduce<float, warpReduceSum>(sum);

    if(threadIdx.x == 0) {
        // Replace absmin/minexp by maximum values if they are still at the initial memset of 0
        atomicCAS((unsigned int*)(analysis_gmem + STATS_ABSMIN), 0, __float_as_uint(FLT_MAX));
        atomicCAS(analysis_gmem + STATS_MINEXP, 0, 255);

        atomicMin(analysis_gmem + STATS_ABSMIN, __float_as_uint(absmin));
        atomicMax(analysis_gmem + STATS_ABSMAX, __float_as_uint(absmax));
        atomicMax(analysis_gmem + STATS_MAXEXP, maxexp);
        atomicMin(analysis_gmem + STATS_MINEXP, minexp);
        atomicAdd(analysis_gmem + STATS_ZEROS, zeros);
        atomicAdd(analysis_gmem + STATS_NEGVAL, negval);
        atomicAdd(analysis_gmem + STATS_POSVAL, posval);
        atomicAdd(analysis_gmem + STATS_INF, inf);
        atomicAdd(analysis_gmem + STATS_NAN, nan);
        atomicAdd((float*)(analysis_gmem + STATS_SUM), sum);

        if (blockIdx.x == 0) {
            analysis_gmem[STATS_TENSOR_DIM] = (uint)n;
            // Printf everything
            //printf("absmin: %f, absmax: %f, maxexp: %d, minexp: %d, zeros: %d, negval: %d, posval: %d, inf: %d, nan: %d, sum: %f\n", absmin, absmax, maxexp, minexp, zeros, negval, posval, inf, nan, sum);
        }
    }
}

// Calculate variance/standard deviation given we already have the mean in the analysis data
template<typename T>
__global__ void analysis_variance_kernel(unsigned int* analysis_gmem, const T* input, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float mean = *(float*)(analysis_gmem + STATS_SUM) / (float)analysis_gmem[STATS_TENSOR_DIM];
    float variance = 0;

    // Calculate variance for the elements this thread is responsible for
    #pragma unroll
    for (int i = 0; i < STATS_ELEMENTS_PER_THREAD; i++) {
        if (i+idx >= n) {
            break;
        }
        float delta = (float)input[i+idx] - mean;
        variance += delta * delta;
    }

    // Block reduction
    variance = blockReduce<float, warpReduceSum>(variance);

    // Global memory atomic add
    if(threadIdx.x == 0) {
        atomicAdd((float*)(analysis_gmem + STATS_VARIANCE), variance);
    }
}

__host__ void write_analysis() {
    unsigned int* analysis_data_cpu = (unsigned int*)malloc(ANALYSIS_MEMORY_SIZE);
    cudaMemcpy(analysis_data_cpu, analysis_memory, ANALYSIS_MEMORY_SIZE, cudaMemcpyDeviceToHost);

    // Let's open a new CSV file and write the headers on the first line
    FILE* f = fopen("analysis.csv", "wt");
    fprintf(f, "Name,Layer,Step,MicroStep,ABSMIN,ABSMAX,MAXEXP,MINEXP,ZEROS,NEGVAL,POSVAL,INF,NAN,AVG,VAR,STD,ELEMENTS");
    // All 256 bins
    for (int i = 0; i < STATS_NUM_HISTOGRAM_BINS; i++) {
        fprintf(f, ",2^%d", i - 127);
    }
    // All 1024 bins for "positive 2^n with mantissa MSB=1", "positive 2^n with mantissa MSB=0", "negative 2^n with mantissa MSB=1", "negative 2^n with mantissa MSB=0"
    for (int i = 0; i < STATS_NUM_HISTOGRAM_BINS; i++) {
        fprintf(f, ",-2^%d MSB0", i - 127);
        fprintf(f, ",-2^%d MSB1", i - 127);
        fprintf(f, ",+2^%d MSB0", i - 127);
        fprintf(f, ",+2^%d MSB1", i - 127);
    }

    for (int h = 0; h < current_analysis && h < MAX_ANALYSIS_STATS; h++) {
        unsigned int *stats = &analysis_data_cpu[h * ANALYSIS_SIZE];
        unsigned int tensor_size = stats[STATS_TENSOR_DIM];
        float stats_fp[ANALYSIS_SIZE];
        for (int i = 0; i < ANALYSIS_SIZE; i++) {
            stats_fp[i] = (float)stats[i] / (float)tensor_size;
        }

        fprintf(f, "\n%s,%d,%d,%d,", analysis_names[h], analysis_layer[h], analysis_step[h], analysis_micro_step[h]);
        fprintf(f, "%.10f,%f,%d,%d,%d,%d,%d,%d,%d,%.10f,%.10f,%.10f,%d",
                    *(float*)(&stats[STATS_ABSMIN]), *(float*)(&stats[STATS_ABSMAX]),
                    stats[STATS_MAXEXP]-127, stats[STATS_MINEXP]-127, stats[STATS_ZEROS], stats[STATS_NEGVAL], stats[STATS_POSVAL], stats[STATS_INF], stats[STATS_NAN],
                    *(float*)(&stats[STATS_SUM]) / (float)tensor_size,
                    *(float*)(&stats[STATS_VARIANCE]) / (float)tensor_size,
                    sqrtf(*(float*)(&stats[STATS_VARIANCE]) / (float)tensor_size),
                    stats[STATS_TENSOR_DIM]);

        for (int i = 0; i < STATS_NUM_HISTOGRAM_BINS; i++) {
            if (stats[i] != 0) {
                fprintf(f, ",%.6f%%", stats_fp[i] * 100.0f);
            } else {
                fprintf(f, ",");
            }
        }

        // detailed histogram
        for (int i = 0; i < STATS_NUM_HISTOGRAM_BINS*4; i++) {
            if (stats[STATS_DETAILED_HISTOGRAM + i] != 0) {
                fprintf(f, ",%d", stats[STATS_DETAILED_HISTOGRAM + i]);
            } else if (i+1 < STATS_NUM_HISTOGRAM_BINS*4) {
                fprintf(f, ",");
            }
        }

        continue;

        printf("==================\n%s (layer %d, step %d[%d])\n==================\n", analysis_names[h], analysis_layer[h], analysis_step[h], analysis_micro_step[h]);
        free(analysis_names[h]);
        analysis_names[h] = NULL;

        for (int i = 0; i < ANALYSIS_SIZE; i++) {
            if (stats[i] != 0) {
                // For i less than 256, this represents an exponent after subtracting the FP32 exponent bias
                // also show in 2^n format
                if (i < 256) {
                    printf("2^%d: %.5f%%\n", i - 127, stats_fp[i] * 100.0f);
                } else {
                    // Show names from this list:
                    switch (i) {
                        case STATS_ABSMIN: printf("ABS MIN: %.10f\n", *(float*)(&stats[i])); break;
                        case STATS_ABSMAX: printf("ABS MAX: %f\n", *(float*)(&stats[i])); break;
                        case STATS_MAXEXP: printf("MAX EXP: %d\n", stats[i]-127); break;
                        case STATS_MINEXP: printf("MIN EXP: %d\n", stats[i]-127); break;
                        case STATS_ZEROS:  printf("ZEROS: %d\n", stats[i]); break;
                        case STATS_NEGVAL: printf("NEG VAL: %d\n", stats[i]); break;
                        case STATS_POSVAL: printf("POS VAL: %d\n", stats[i]); break;
                        case STATS_INF:    printf("INF: %d\n", stats[i]); break;
                        case STATS_NAN:    printf("NAN: %d\n", stats[i]); break;
                        case STATS_SUM:    printf("AVG: %.10f\n", *(float*)(&stats[i]) / (float)tensor_size); break;
                        case STATS_VARIANCE: {
                            float variance = *(float*)(&stats[i]) / (float)tensor_size;
                            printf("VAR: %.10f\n", variance);
                            printf("STD: %.10f\n", sqrtf(variance));
                            break;
                        }
                        case STATS_TENSOR_DIM: printf("ELEMENTS: %d\n", stats[i]); break;
                        default: printf("UNKNOWN: %d\n", stats[i]); break;
                    }
                }
            }
        }
    }

    fclose(f);

    current_analysis = 0;
    free(analysis_data_cpu);
    cudaMemset(analysis_memory, 0, ANALYSIS_MEMORY_SIZE);
    cudaCheck(cudaGetLastError());

    exit(1);
}

template<typename T>
__host__ void generate_analysis(const T* tensor, size_t count, const char* name) {
    if (current_analysis >= MAX_ANALYSIS_STATS) {
        if (current_analysis == MAX_ANALYSIS_STATS) { // only warn 1st time we run out of space
            printf("Exceeded maximum number of analysis stats per dump (%d)\n", MAX_ANALYSIS_STATS);
            current_analysis++;
        }
        return;
    }
    cudaCheck(cudaGetLastError());

    // Check if tensor name is in the hashmap
    if (analysis_tensor_names.find(std::make_pair(name, global_current_layer)) != analysis_tensor_names.end()) {
        // Add " [2]" unless also already in hashmap, then add " [3]", etc...
        analysis_names[current_analysis] = (char*)malloc(strlen(name) + 5);
        int i = 2;
        do {
            assert(i < 100);
            sprintf(analysis_names[current_analysis], "%s [%d]", name, i++);
        } while (analysis_tensor_names.find(std::make_pair(analysis_names[current_analysis], global_current_layer)) != analysis_tensor_names.end());
    } else {
        analysis_names[current_analysis] = (char*)malloc(strlen(name) + 1);
        memcpy(analysis_names[current_analysis], name, strlen(name) + 1);
    }
    analysis_tensor_names.insert(std::make_pair(analysis_names[current_analysis], global_current_layer));

    analysis_layer[current_analysis] = global_current_layer;
    analysis_step[current_analysis] = global_current_step;
    analysis_micro_step[current_analysis] = global_current_micro_step;

    // Run analysis_kernel
    unsigned int* analysis = &analysis_memory[current_analysis * ANALYSIS_SIZE];
    analysis_kernel<<<CEIL_DIV(count, 1024*STATS_ELEMENTS_PER_THREAD), 1024>>>(analysis, tensor, count);
    analysis_variance_kernel<<<CEIL_DIV(count, 1024*STATS_ELEMENTS_PER_THREAD), 1024>>>(analysis, tensor, count);
    current_analysis++;

    cudaCheck(cudaGetLastError());
}

#endif