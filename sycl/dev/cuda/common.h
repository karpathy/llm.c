#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <dpct/blas_utils.hpp>

#include <float.h>
#include <dpct/lib_common_utils.hpp>

#include <cmath>

template<class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

float warpReduceSum(float val, const sycl::nd_item<3> &item_ct1) {
    for (int offset = 16; offset > 0; offset /= 2) {
        /*
        DPCT1096:551: The right-most dimension of the work-group used in the
        SYCL kernel that calls this function may be less than "32". The function
        "dpct::permute_sub_group_by_xor" may return an unexpected result on the
        CPU device. Modify the size of the work-group to ensure that the value
        of the right-most dimension is a multiple of "32".
        */
        val += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val,
                                              offset);
    }
    return val;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(dpct::err0 error, const char *file, int line) {

};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(int status, const char *file, int line)
{
    if (status != 0) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static dpct::library_data_t cublas_compute_type;
dpct::blas::descriptor_ptr cublas_handle;
cublasLtHandle_t cublaslt_handle;
int cuda_arch_major = 0;
int cuda_arch_minor = 0;
int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#if DPCT_COMPATIBILITY_TEMP == 800 || DPCT_COMPATIBILITY_TEMP >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
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
        return constant(0);
    }

    static Packed128 ones() {
        return constant(1);
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
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(sycl::int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedef
typedef Packed128<float> f128;

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
        __ldcs(reinterpret_cast<const sycl::int4 *>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<sycl::int4 *>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
void store128cs(ElementType* target, Packed128<ElementType> value) {
    /*
    DPCT1098:317: The '=' expression is used instead of the __stcs call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    *target = value.get_bits();
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
void store128cg(ElementType* target, Packed128<ElementType> value) {
    /*
    DPCT1098:318: The '=' expression is used instead of the __stcg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    *target = value.get_bits();
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#if defined(ENABLE_BF16)

typedef sycl::ext::oneapi::bfloat16 floatX;
typedef sycl::ext::oneapi::bfloat16 floatN;
#define CUBLAS_LOWP                                                            \
    dpct::library_data_t::real_bfloat16 // CUDA_R_16F or CUDA_R_16BF (or
                                        // CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#elif defined(ENABLE_FP16)

typedef half floatX;
typedef half floatN;

#else

typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;

/*
// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#if defined(ENABLE_BF16) && (DPCT_COMPAT_RT_MAJOR_VERSION < 12) &&             \
    !((DPCT_COMPATIBILITY_TEMP >= 800) || !defined(DPCT_COMPATIBILITY_TEMP))
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

*/
// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
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
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template <class TargetType>
[[nodiscard]] dpct::err0 memcpy_convert(TargetType *d_ptr, float *h_ptr,
                                        size_t count) try {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    dpct::err0 status = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(d_ptr, converted, count * sizeof(TargetType))
            .wait());
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void setup_main() {
    srand(0);   // determinism

    // set up the device
    int deviceIdx = 0;
    /*
    DPCT1093:319: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    dpct::get_device_info(deviceProp,
                          dpct::dev_mgr::instance().get_device(deviceIdx));
    cuda_num_SMs = deviceProp.get_max_compute_units();
    cuda_threads_per_SM = deviceProp.get_max_work_items_per_compute_unit();
    /*
    DPCT1005:320: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    cuda_arch_major = deviceProp.get_major_version();
    /*
    DPCT1005:321: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    cuda_arch_minor = deviceProp.get_minor_version();

    // setup cuBLAS and cuBLASLt
    cublasCheck(DPCT_CHECK_ERROR(cublas_handle = new dpct::blas::descriptor()));
    /*
    DPCT1007:322: Migration of cublasLtCreate is not supported.
    */
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    /*
    DPCT1064:331: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    cudaCheck(DPCT_CHECK_ERROR(
        cublaslt_workspace = (void *)sycl::malloc_device(
            cublaslt_workspace_size, dpct::get_in_order_queue())));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    int cublas_math_mode =
        enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    /*
    DPCT1027:323: The call to cublasSetMathMode was replaced with 0 because this
    functionality is redundant in SYCL.
    */
    cublasCheck(0);
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue()
            .memcpy(out_gpu, device_result, num_elements * sizeof(D))
            .wait()));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    dpct::event_ptr start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    /*
    DPCT1093:324: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_device_info(
        deviceProp, dpct::dev_mgr::instance().get_device(deviceIdx))));
    void* flush_buffer;
    /*
    DPCT1051:325: SYCL does not support a device property functionally
    compatible with l2CacheSize. It was migrated to global_mem_cache_size. You
    may need to adjust the value of global_mem_cache_size for the specific
    device.
    */
    /*
    DPCT1064:332: Migrated cudaMalloc call is used in a macro/template
    definition and may not be valid for all macro/template uses. Adjust the
    code.
    */
    cudaCheck(DPCT_CHECK_ERROR(flush_buffer = (void *)sycl::malloc_device(
                                   deviceProp.get_global_mem_cache_size(),
                                   dpct::get_in_order_queue())));

    cudaCheck(DPCT_CHECK_ERROR(start = new sycl::event()));
    cudaCheck(DPCT_CHECK_ERROR(stop = new sycl::event()));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        /*
        DPCT1051:326: SYCL does not support a device property functionally
        compatible with l2CacheSize. It was migrated to global_mem_cache_size.
        You may need to adjust the value of global_mem_cache_size for the
        specific device.
        */
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(flush_buffer, 0, deviceProp.get_global_mem_cache_size())
                .wait()));
        // now we can start recording the timing of the kernel
        /*
        DPCT1024:327: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::sync_barrier(start, &dpct::get_in_order_queue())));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        /*
        DPCT1024:328: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::sync_barrier(stop, &dpct::get_in_order_queue())));
        cudaCheck(DPCT_CHECK_ERROR(start->wait_and_throw()));
        cudaCheck(DPCT_CHECK_ERROR(stop->wait_and_throw()));
        float single_call;
        cudaCheck(DPCT_CHECK_ERROR(
            single_call = (stop->get_profiling_info<
                               sycl::info::event_profiling::command_end>() -
                           start->get_profiling_info<
                               sycl::info::event_profiling::command_start>()) /
                          1000000.0f));
        elapsed_time += single_call;
    }

    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(flush_buffer, dpct::get_in_order_queue())));

    return elapsed_time / repeats;
}
