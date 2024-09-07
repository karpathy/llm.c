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
    FP32, FP16, BF16
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
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }

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
template<bool noise=true, typename highp=float, typename Ti=__nv_fp8_e4m3>
__device__ __forceinline__ void stochastic_rounding(float in, Ti *out, unsigned int seed, float prob_offset=0.0f) {
    unsigned int random = noise ? get_random_noise(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed) : seed;

    // prob_offset allows rounding towards gradient more of the time (one paper recommends that)
    // e.g. +0.3f ==> 65% chance up, 35% chance down
    highp threshold_percentage = ((highp)random / (highp)0xFFFFFFFF) - prob_offset;

    Ti rounded_down, rounded_up;
    if constexpr (std::is_same<Ti, half>::value) {
        rounded_down = __float2half_rd(in);
        rounded_up = __float2half_ru(in);
    } else if constexpr (std::is_same<Ti, __nv_bfloat16>::value) {
        rounded_down = __float2bfloat16_rd(in);
        rounded_up = __float2bfloat16_ru(in);
    } else if constexpr (std::is_same<Ti, __nv_fp8_e4m3>::value) {
        // CUDA doesn't have round down/up instructions for FP8 (in SW or HW) so we do it ourselves
        // ARM-Intel-NVIDIA style FP8 E4M3 (different for AMD-Graphcore-Qualcomm format!)
        Ti rounded = __nv_fp8_e4m3(in);
        unsigned char rounded_bits = rounded.__x;
        unsigned char absolute_bits = rounded_bits & 127;
        unsigned char rounded_up_bits = absolute_bits + 1;
        unsigned char rounded_down_bits = absolute_bits - 1;

        // compiler likes the following code atm, but small changes may increase instructions by a lot
        // as it may suddenly decide to use branches rather than predication...
        if (absolute_bits >= 126) { // maximum normal value (+NaN)
            rounded_up_bits = absolute_bits;
            if (absolute_bits == 127) { // NaN (not always preserving sign)
                rounded_down_bits = 127;
            }
        } else if (absolute_bits == 0) { // zero
            rounded_down_bits = 0;
        } else {
            unsigned char mantissa_bits = absolute_bits & 7;
            if (mantissa_bits == 7) { // maximum mantissa (already known non-NaN/non-max)
                rounded_up_bits = (absolute_bits - mantissa_bits) + 8; // clear mantissa, add 1 to exponent
            } else if (mantissa_bits == 0) { // minimum mantissa (already known non-zero)
                rounded_down_bits = (absolute_bits + 7) - 8; // max mantissa, subtract 1 from exponent
            }
        }
        if (in < 0) { // negative input: swap rounded up/down and add negative sign
            unsigned char swap_tmp = rounded_down_bits | 128;
            rounded_down_bits = rounded_up_bits | 128;
            rounded_up_bits = swap_tmp;
        }

        // rounding to nearest even already gave us 1 of the 2 rounded values surrounding the input
        // we only need the other one (but no point skipping anything above given SIMT divergence)
        rounded_down.__x = ((float)rounded <= in) ? rounded.__x : rounded_down_bits;
        rounded_up.__x = ((float)rounded >= in) ? rounded.__x : rounded_up_bits;
    } else if constexpr (std::is_same<Ti, __nv_fp8_e5m2>::value) {
        assert(false); // todo
    } else {
        assert(false);
    }

    highp diff = (highp)rounded_up - (highp)rounded_down;
    highp lerp = ((highp)in - (highp)rounded_down) / diff; // division by 0 is OK as it means (up == down) anyway
    *out = (lerp > threshold_percentage) ? rounded_up : rounded_down;
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

// ----------------------------------------------------------------------------
// ...
template<typename ElementType=float>
struct TensorGPU {
    ElementType* data_ptr;
    float* scale_descale_ptr;
    unsigned int* absmax_ptr;
    size_t num_elements;

    static __device__ __host__ TensorGPU from(ElementType* ptr=nullptr) {
        TensorGPU tmp = {0};
        tmp.data_ptr = ptr;
        return tmp;
    }

    template<typename T>
    __device__ __host__ T* as() {
        return reinterpret_cast<T*>(data_ptr);
    }

    __device__ __host__  operator ElementType*() const {
        return data_ptr;
    }

    __device__ __host__ ElementType& operator[](size_t index) {
        return data_ptr[index];
    }

    __device__ __host__ const ElementType& operator[](size_t index) const {
        return data_ptr[index];
    }

    __device__ __host__ int num_per_128() const {
        return sizeof(int4) / sizeof(ElementType);
    }

    __device__ __host__ float get_scalar(size_t index, bool disable_scaling=true) const {
        ElementType* __restrict__ data_ptr_restricted = data_ptr;
        float* __restrict__ scale_ptr_restricted = scale_descale_ptr;

        float value = (float)data_ptr_restricted[index];
        float descale = (scale_descale_ptr && !disable_scaling) ? scale_ptr_restricted[1] : 1.0f;
        return value * descale; // [1] = descale
    }

    __device__ __host__ ElementType set_scalar(size_t index, float value, bool disable_scaling=true) {
        ElementType* __restrict__ data_ptr_restricted = data_ptr;
        float* __restrict__ scale_ptr_restricted = scale_descale_ptr;

        float scale = (scale_descale_ptr && !disable_scaling) ? scale_ptr_restricted[0] : 1.0f;
        ElementType output = (ElementType)(value * scale);
        data_ptr_restricted[index] = output;
        return output;
    }
};

// short-form typedefs
typedef TensorGPU<floatX> tensorX;
typedef TensorGPU<float> tensorFP32;
typedef TensorGPU<half> tensorFP16;
typedef TensorGPU<nv_bfloat16> tensorBF16;

typedef TensorGPU<floatX> tensorFP8e4;
typedef TensorGPU<floatX> tensorFP8e5;

extern TensorGPU<floatX> null_tensorX;
extern TensorGPU<floatX> null_tensorFP32;

template<typename ElementType=float>
struct tensor128 {
private:
    Packed128<ElementType> data128;
    ElementType* data_ptr;
    unsigned int *absmax_ptr;
    float scale;
    float descale;
    float new_absmax = 0.0f;
    bool wrote_data = false;
    bool wrote_absmax = false;

public:
    bool scaling = (sizeof(ElementType) == 33); // todo - fp8 only
    static constexpr const size_t elements = sizeof(int4) / sizeof(ElementType);

    __device__ tensor128(TensorGPU<ElementType> tensor, bool disable_scaling=false) {
        float2* __restrict__ ptr_restricted = (float2*)tensor.scale_descale_ptr;
        float2 scale_descale = *ptr_restricted;
        scale = scale_descale.x;
        descale = scale_descale.y;
        data_ptr = tensor.data_ptr;
        absmax_ptr = tensor.absmax_ptr;
        if (disable_scaling) {
            scaling = false;
        }
        scaling = false;
    }

    __device__ void load(size_t offset, bool cache_streaming=false) {
        ElementType* addr = data_ptr + offset;
        data128 = cache_streaming ? load128cs(addr) : load128(addr);
    }

    __device__ void store(size_t offset, bool cache_streaming=false) {
        if (cache_streaming) {
            store128cs(data_ptr + offset, data128);
        } else {
            store128(data_ptr + offset, data128);
        }
        wrote_data = true;
    }

    template <typename OriginalType>
    __device__ void store_same_length(size_t offset, bool cache_streaming=false) {
        if (cache_streaming) {
            store128_same_length_cs<OriginalType, ElementType>(data_ptr + offset, data128);
        } else {
            store128_same_length<OriginalType, ElementType>(data_ptr + offset, data128);
        }
        wrote_data = true;
    }

    __device__ const Packed128<ElementType>& get128() const {
        return data128;
    }

    __device__ Packed128<ElementType>& get128() {
        return data128;
    }

    // call this manually if e.g. you use set_scalar() to update the tensor
    // todo - in the future, this could consider more than just absmax
    __device__ void add_value_stats(float value, ElementType output) {
        new_absmax = max(new_absmax, fabsf(value));
    }

    __device__ float get(int index) {
        return (float)data128[index] * (scaling ? descale : 1.0f);
    }

    __device__ void set(int index, float value) {
        data128[index] = (ElementType)(value * (scaling ? scale : 1.0f));
        add_value_stats(value, data128[index]);
    }

    __device__ void set_stochastic(int index, float value, unsigned int random_number,
                                   bool rotate_by_index=true, bool non_deterministic_rng=false) {
        float scaled_value = value * (scaling ? scale : 1.0f);

        // rotate the random number by the index so we can cheaply reuse the same RNG
        // obviously less good than having true per-index RNG, but should be good enough
        // when rounding FP32 to FP8, most of the bits make extremely little difference anyway...
        // x10 is used so that it never repeats for indices [0;15] with a minimum difference of 2 etc.
        if (rotate_by_index) {
            assert(index < 16); // >=16 would repeat and be extremely bad RNG
            random_number = __funnelshift_l(random_number, random_number, index * 10);
        }
        // RNG without a seed from the host for quick testing, but obviously not deterministic!
        #ifdef FORCE_NON_DETERMINISM
        non_deterministic_rng = true;
        #endif
        if (non_deterministic_rng) {
            unsigned int clock, laneid;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(clock));
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
            random_number = get_random_noise(clock, laneid, blockIdx.x * blockDim.x);
        }

        stochastic_rounding<false>(scaled_value, &data128[index], random_number);
        add_value_stats(value, data128[index]);
    }

    __device__ bool update_absmax(int thread_id, int num_threads, bool exit=false, bool forced=false) {
        if (!forced && !scaling) {
            return false; // if we return true, we can skip __syncthreads() in some kernels
        }
        wrote_absmax = true;

        // use native integer reductions as much as possible (supported on all GPUs with FP8)
        // this might treat NaN/INF slightly differently but that is the least of our problems
        unsigned int absmax_uint = *(unsigned int*)&new_absmax;
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        __shared__ unsigned int shared[32];

        // lane_id must be obtained directly from the special register
        // otherwise, the compiler does silly things related to the redux/atomicMax
        unsigned int lane_id ;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(lane_id));
        unsigned int num_warps = num_threads >> 5;
        unsigned int warp_id = thread_id & 31;

        // with this condition instead of lane_id == 0, we have shared[lane_id] both here and below
        // this reduces the number of instructions for addressing
        if (lane_id == warp_id) {
            shared[lane_id] = absmax_uint;
        }

        // sync can be after exit (dead threads don't count) but must be before return
        // if this is the end of the kernel, the compiler puts a conditional EXIT right after BAR
        // but this way the EXIT is right before the barrier which frees the warps slightly quicker
        bool done = (warp_id != 0 || lane_id >= num_warps);
        if (done && exit) asm volatile("exit;");
        __syncthreads();
        if (done && !exit) return true;

        // one more warp reduction then global memory atomic
        // we want as few global atomics as possible (i.e. 1 per threadblock)
        absmax_uint = shared[lane_id];
        asm volatile("redux.sync.max.u32 %0, %0, 0xff;" : "+r"(absmax_uint));
        if (lane_id == 0) {
            atomicMax(absmax_ptr, absmax_uint);
        }
        return true;
    }
    __device__ void update_absmax_1D(bool exit=false) {
        update_absmax(threadIdx.x & 31, blockDim.x >> 5, exit);
    }
    __device__ void skip_absmax() {
        wrote_absmax = true;
    }

    template <typename ForcedType>
    __device__ void force_precision(bool stochastic=false, int microtensor_scale=false,
                                    int zeroed_mantissa_bits=0, bool two_four_sparsity=false) {
        for (int k = 0; k < elements; k++) {
            // todo: fancy stuff
            if (scaling || scale == 0.0f) { // already scaled
                data128[k] = (ElementType)((ForcedType)(data128[k]));
            } else { // need to scale & descale
                float scaled_value = (float)data128[k] * scaling;
                ForcedType converted_value = (ForcedType)scaled_value;
                float descaled_value = (float)converted_value * descale;
                data128[k] = (ElementType)descaled_value;
            }
        }
    }

    __device__ ~tensor128() {
        // this should ~always be optimised away by the compiler
        assert(wrote_absmax || !scaling || !wrote_data);
    }
};

template <typename T>
__device__ tensor128<T> new_tensor128(TensorGPU<T> tensor, bool disable_scaling=false) {
    return tensor128<T>(tensor, disable_scaling);
}

template <typename T>
__device__ tensor128<T> load_tensor128(TensorGPU<T> tensor, size_t offset,
                                       bool cache_streaming = false, bool disable_scaling=false) {
    tensor128<T> t128(tensor, disable_scaling);
    t128.load(offset, cache_streaming);
    return t128;
}

// ----------------------------------------------------------------------------
// ...

constexpr size_t MAX_TENSORS = 16*1024;
constexpr size_t MAX_ABSMAX_HISTORY = 32; // todo - should make this a command line option
extern int num_tensor_specs;
extern int current_absmax_index;
extern void* gpu_tensor_scale_memory;
extern void* gpu_tensor_absmax_memory;

enum TT : uint8_t {
    PARAMETER=0, PARAMETER_GRAD, PARAMETER_MASTER, PARAMETER_OPT_M, PARAMETER_OPT_V, // 1 allocation each
    ACTIVATIONS_MULTIUSE, // single buffer shared for activations, activation gradients, and scratch
    DEFAULT, COUNT=DEFAULT, NUM_TYPES_PARAM=PARAMETER_OPT_V+1
};

enum TFlags : uint8_t {
    NONE=0,
    REUSED_MEMORY=1,
    GRADIENT=2,
    TENSOR_2D=4,
    BIAS=8,
    LAYERNORM=16,
    RESIDUAL=32,
    EMBEDDING=64,
    STATS=128
};

typedef struct {
    char* ptr;
    size_t offset; // into base pointer
    size_t num_elements; // per shard
    int id;
    short num_shards;
    short remaining_layers;
    DType data_type;
    TT tensor_type;
    int flags;
    char name[16];

    template <typename T>
    operator T*() const {
        if (std::is_same<T, float>::value && data_type != DType::FP32 ||
            std::is_same<T, __half>::value && data_type != DType::FP16 ||
            std::is_same<T, nv_bfloat16>::value && data_type != DType::BF16) {
            printf("ERROR: Unexpected data type (%d) for tensor %s\n", (int)data_type, name);
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<T*>(ptr);
    }

    template <typename T>
    operator TensorGPU<T>() const {
        TensorGPU<T> tensor;
        int absmax_idx = id + (current_absmax_index * num_tensor_specs);

        tensor.num_elements = num_elements;
        tensor.data_ptr = this->operator T*();
        tensor.scale_descale_ptr = reinterpret_cast<float*>(gpu_tensor_scale_memory) + id;
        tensor.absmax_ptr = reinterpret_cast<unsigned int*>(gpu_tensor_absmax_memory) + absmax_idx;

        return tensor;
    }
} TensorSpec;

// ----------------------------------------------------------------------------
// Copy, cast functions

using elementwise_func_t = float (*) (float);
__device__ float nothing_elementwise(float x) {
    return x;
}
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

// todo - move to GELU etc.
__device__ float gelu_forward_elementwise(float x) {
    float cube = 0.044715f * x * x * x;

    float tanh_out;
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + cube);
    asm ("tanh.approx.f32 %0,%1;" : "=f"(tanh_out) : "f"(tanh_arg));

    // the following uses FMUL+FMA instead of FMUL+FADD+FMUL for "0.5f * x * (1.0f + tanh_out)"
    float half_x = 0.5f * x;
    return half_x * tanh_out + half_x;
}

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
// requires all 32 threads in the warp to be active, but should work for any block size
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