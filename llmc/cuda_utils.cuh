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
__device__ Packed128<ElementType> load128(const ElementType* __restrict__ address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* __restrict__ address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* __restrict__ target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* __restrict__ target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* __restrict__ target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// This helper is for when we want to copy from e.g. FP32 to BF16
// so if want to load a f128 of 4 elements, and write those 4 elements to memory as 64-bit
// not needed in the case of loads, the compiler will automatically optimise away unused reads
template<class OriginalType, class ElementType>
__device__ void store128_same_length(ElementType* target, Packed128<ElementType> value) {
    int4 bits = value.get_bits();
    switch (sizeof(OriginalType) / sizeof(ElementType)) {
        case 0: *reinterpret_cast<int4*>(target) = bits; break; // smaller
        case 1: *reinterpret_cast<int4*>(target) = bits; break; // same size
        case 2: *reinterpret_cast<int2*>(target) = make_int2(bits.x, bits.y); break;
        case 4: *reinterpret_cast<int*>(target) = bits.x; break;
        default: break; //assert(false);
    }
}

// todo - can we unify this with non-cs function somehow?
template<class OriginalType, class ElementType>
__device__ void store128_same_length_cs(ElementType* target, Packed128<ElementType> value) {
    int4 bits = value.get_bits();
    switch (sizeof(OriginalType) / sizeof(ElementType)) {
        case 0: __stcs(reinterpret_cast<int4*>(target), bits); break; // smaller
        case 1: __stcs(reinterpret_cast<int4*>(target), bits); break; // same size
        case 2: __stcs(reinterpret_cast<int2*>(target), make_int2(bits.x, bits.y)); break;
        case 4: __stcs(reinterpret_cast<int*>(target), bits.x); break;
        default: break; //assert(false);
    }
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// DType support

// enumerator to indentify the datatype of a tensor.
enum class DType : uint8_t {
    FP32, FP16, BF16, FP8E4M3, FP8E5M2
};

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        case DType::FP8E4M3:
            return sizeof(__nv_fp8_e4m3);
        case DType::FP8E5M2:
            return sizeof(__nv_fp8_e5m2);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }
DType dtype_of(__nv_fp8_e4m3 * f) { return DType::FP8E4M3; }
DType dtype_of(__nv_fp8_e5m2 * f) { return DType::FP8E5M2; }

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding (defined here as used by TensorGPU)

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed) {
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
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

// rely on default values of 0 being optimised away for 1D/2D/3D (shorter than original code)
__device__ __host__ unsigned int get_random_noise(unsigned int seed, unsigned int x,
                                                  unsigned int y=0, unsigned int z=0, unsigned int t=0) {
	constexpr unsigned int PRIME1 = 198491317u; // Large prime number with non-boring bits
	constexpr unsigned int PRIME2 = 6542989u; // Large prime number with distinct and non-boring bits
	constexpr unsigned int PRIME3 = 357239u; // Large prime number with distinct and non-boring bits
	return SquirrelNoise5(x + (PRIME1 * y) + (PRIME2 * z) + (PRIME3 * t), seed);
}

// stochastic rounding (typicalling using Squirel Noise above to go from a seed to a random number)
// new algorithm that calculates distance from rounded up/down values to correctly handle denorms
// (didn't matter with BF16 because denorms are so tiny they're irrelevant, unlike in FP8/FP16)
template<typename Ti=__nv_fp8_e4m3>
__device__ void stochastic_rounding(float in, Ti &out, unsigned int random, float prob_offset=0.0f) {
    if constexpr (std::is_same<Ti, float>::value) {
        out = in;
        return;
    }

    // prob_offset allows rounding towards gradient more of the time (one paper recommends that)
    // e.g. +0.3f ==> 65% chance up, 35% chance down
    float threshold_percentage = ((float)random / (float)0xFFFFFFFF) - prob_offset;

    Ti rounded_down = (Ti)0.0f, rounded_up = (Ti)0.0f;
    if constexpr (std::is_same<Ti, half>::value) {
        rounded_down = __float2half_rd(in);
        rounded_up = __float2half_ru(in);
    } else if constexpr (std::is_same<Ti, __nv_bfloat16>::value) {
        rounded_down = __float2bfloat16_rd(in);
        rounded_up = __float2bfloat16_ru(in);
    } else if constexpr (std::is_same<Ti, __nv_fp8_e4m3>::value) {
        // CUDA doesn't have round down/up instructions for FP8 (in SW or HW) so we do it ourselves
        // ARM-Intel-NVIDIA style FP8 E4M3 (different for AMD-Graphcore-Qualcomm format!)
        // tried this approach to avoid fake_fp8 bug (didn't help), keeping it for now...
        // todo: compare perf & accuracy to bit shifting method (do exhaustive testing)
        float low = in;
        float high = in;

        if (fabsf(in) < 0.0156f) {
            low -= 0.000975f;
            high += 0.000975f;
        } else {
            if (in > 0.0f) {
                low *= (15.5f / 16.0f);
                high *= (8.5f / 8.0f);
            } else {
                low *= (8.5f / 8.0f);
                high *= (15.5f / 16.0f);
            }
        }
        rounded_up = (__nv_fp8_e4m3)high;
        rounded_down = (__nv_fp8_e4m3)low;
    } else {
        assert(false);
    }

    float diff = (float)rounded_up - (float)rounded_down;
    float lerp = (in - (float)rounded_down) / diff; // division by 0 is OK as it means (up == down) anyway
    out = (lerp > threshold_percentage) ? rounded_up : rounded_down;
}

// ----------------------------------------------------------------------------
__device__ float fake_fp8(bool faking, float input, float scale, float descale, bool mode_e5, bool stochastic=false) {
#ifdef FAKE_FP8
    unsigned int random_number;
    if (faking && scale != 1.0f) {
        assert(scale == 1.0f/descale || descale == 1.0f/scale || scale == 1.0f);
        if (stochastic) {
            unsigned int clock, laneid;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
            random_number = get_random_noise(clock, laneid, blockIdx.x * blockDim.x);
        }

        if (mode_e5) {
            __nv_fp8_e5m2 value_fp8 = __nv_fp8_e5m2(input * scale);
            return ((float)value_fp8) * descale;
        } else {
            __nv_fp8_e4m3 value_fp8 = __nv_fp8_e4m3(input * scale);
            if (stochastic) {
                // BUGGED(?) - spent 6+ hours debugging and I genuinely suspect a compiler bug *sigh*
                stochastic_rounding(input * scale, value_fp8, random_number);
            }
            return ((float)value_fp8) * descale;
        }
    }
#endif
    return input;
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
// warp-level reduction for finding the maximum value
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any 1D(!) block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// Performs a _deterministic_ sum reduction. determinism is achieved by requiring that only
// a single block be used.
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

#endif