// Utilities for use in __device__ code

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sycl_common.h"
#include <cmath>
#include <dpct/blas_utils.hpp>
 
#define CUBLAS_LOWP dpct::library_data_t::real_bfloat16
#define CUBLAS_LOWP_COMPUTE dpct::library_data_t::real_float
#define CUBLAS_COMPUTE_32F CUBLAS_LOWP_COMPUTE
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
void* cublaslt_workspace = NULL;
dpct::library_data_t cublas_compute = CUBLAS_COMPUTE_32F;
// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    explicit Packed128(sycl::int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    static Packed128 zeros() {
        return constant(0.f);
    }
    static Packed128 ones() {
        return constant(1.f);
    }

    ElementType& operator[](int index) {
        return payload[index];
    }
    const ElementType& operator[](int index) const {
        return payload[index];
    }
    sycl::int4 get_bits() const {
        sycl::int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size =
        sizeof(sycl::int4) / sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{
        *reinterpret_cast<const sycl::int4 *>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{
        *reinterpret_cast<const sycl::int4 *>(address)};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4 *>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
void store128cs(ElementType* target, Packed128<ElementType> value) {
    
    *reinterpret_cast<sycl::int4 *>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
void store128cg(ElementType* target, Packed128<ElementType> value) {

    *reinterpret_cast<sycl::int4 *>(target) = value.get_bits();
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
Td cast_value(Ts val);

template<>
float cast_value<float, float>(float val) {
    return val;
}

template <> float cast_value<float, sycl::half>(sycl::half val) {
    return sycl::vec<sycl::half, 1>(val)
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

template <>
float cast_value<float, sycl::ext::oneapi::bfloat16>(
    sycl::ext::oneapi::bfloat16 val) {
    return static_cast<float>(val);
}

template<typename Td, typename Ts>
void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n,
                          const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx] = cast_value<Td, Ts>(src[idx]);
    }
}

// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
inline float warpReduceSum(float val, const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        
        val += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val,
                                              offset);
    }
    return val;
}
// warp-level reduction for finding the maximum value
SYCL_EXTERNAL inline float warpReduceMax(float val,
                                         const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        
        val = sycl::fmax(val, dpct::permute_sub_group_by_xor(
                                  item_ct1.get_sub_group(), val, offset));
    }
    return val;
}
// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = float (*) (float, const sycl::nd_item<3>&);
template <reduction_func_t warp_reduction>
SYCL_EXTERNAL inline float
blockReduce(float val, const sycl::nd_item<3> &item_ct1, float *shared_val,
            bool final_sync = false, float out_of_bounds = 0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)

    const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
    const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const int num_warps = item_ct1.get_local_range(2) / WARP_SIZE;

    float warp_val = warp_reduction(val, item_ct1);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    /*
    DPCT1065:188: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val, item_ct1);

    if (final_sync) {
        
        item_ct1.barrier(sycl::access::fence_space::local_space); 
    }
    return block_val;
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
inline constexpr unsigned int SquirrelNoise5(int positionX, unsigned int seed)
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
inline constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr int PRIME_NUMBER = 198491317; // Large prime number with non-boring bits
    return SquirrelNoise5(indexX + (PRIME_NUMBER * indexY), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
__dpct_inline__ void stochastic_rounding(float in,
                                         sycl::ext::oneapi::bfloat16 *out,
                                         unsigned int seed,
                                         const sycl::nd_item<3> &item_ct1) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    unsigned int random =
        Get2dNoiseUint(item_ct1.get_local_id(2), item_ct1.get_group(2), seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = sycl::bit_cast<unsigned int>(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    //*out = sycl::ext::intel::math::float2bfloat16_rn(sycl::bit_cast<float>(float_bits));
    *out =(sycl::bit_cast<float>(float_bits));
}
__dpct_inline__ void stochastic_rounding(float in, sycl::half *out,
                                         unsigned int random,
                                         const sycl::nd_item<3> &item_ct1) {
    *out = (float)in; // todo - implement this...
}
__dpct_inline__ void stochastic_rounding(float in, float *out,
                                         unsigned int random,
                                         const sycl::nd_item<3> &item_ct1) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}



template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance = 1e-4) {
    sycl::queue q(sycl::default_selector_v);

    // Allocate host memory using SYCL
    D* out_gpu = sycl::malloc_host<D>(num_elements * sizeof(D), q);

    // Copy results from device to host
    q.memcpy(out_gpu, device_result, num_elements * sizeof(D)).wait();

    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = 0.079f;
#else
    float epsilon = 0.079f;
#endif

    for (std::size_t i = 0; i < num_elements; ++i) {
        // Skip masked elements
        if (!std::isfinite(cpu_reference[i])) {
            continue;
        }

        // Print the first few comparisons
        if (i < 5) {
            std::cout << cpu_reference[i] << " " << static_cast<T>(out_gpu[i]) << std::endl;
        }

        // Effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + std::fabs(cpu_reference[i]) * epsilon;

        // Ensure correctness for all elements
        if (std::fabs(cpu_reference[i] - static_cast<T>(out_gpu[i])) > t_eff) {
            std::cerr << "Mismatch of " << name << " at " << i << ": CPU_ref: " << cpu_reference[i] << " vs GPU: " << static_cast<T>(out_gpu[i]) << std::endl;
            nfaults++;
            if (nfaults >= 10) {
                sycl::free(out_gpu, q);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        sycl::free(out_gpu, q);
        std::exit(EXIT_FAILURE);
    }

    sycl::free(out_gpu, q);
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = (int)rand() %V; // range -1..1
    }
    return arr;
}

float* make_random_floatX(size_t N) {
    floatX* arr = (floatX*)malloc(N * sizeof(floatX));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((floatX)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}


float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_zeros_floatX(size_t N) {
    floatX* arr = (floatX*)malloc(N * sizeof(floatX));
    memset(arr, 0, N * sizeof(floatX)); // all zero
    return arr;
}




#endif