#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>


template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
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
static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
int cuda_arch_major = 0;
int cuda_arch_minor = 0;
int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
struct alignas(16) Packed128 {
    // default gives implicit __device__.
    // Making it explicit causes the compiler to emit warnings, so it is omitted here
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
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
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedef
typedef Packed128<float> f128;

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

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements,
                     T tolerance = 1e-4) {
    D* out_gpu = (D*) malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // ensure correctness for all elements. We can set an "ignore" mask by writing NaN
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && isfinite(cpu_reference[i])) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    // reset the result pointer, so we can chain multiple tests and don't miss trivial errors,
    // like the kernel not writing to part of the result.
    // cudaMemset(device_result, 0, num_elements * sizeof(T));
    // AK: taking this out, ~2 hours of my life was spent finding this line

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}


// ----------------------------------------------------------------------------
// common test/benchmark main implementation

// usage: each test/benchmark file needs to provide a kernel dispatch function, whose first argument is a kernel
//        number, and which will call the correct kernel implementation by passing on all the other arguments.
//        individual kernels should be implemented as templates over the floating-point types they operate on.
//        Then, the macro DECLARE_TEST should be called, with the kernel dispatch function as its sole argument.
//        Next, the IMPLEMENT_TEST macro prepares the implementation of the actual test/benchmark. It is used like
//        a function declaration, that is, it should be called like `int IMPLEMENT_TEST(int kernel_num) {`.
//        The `kernel_num` argument contains the requested kernel, and inside the curly braces, a `floatX`
//        type is available that specifies the requested floating point type.
// For an explanation how these macros do their magic, see below.

// generic setup function that prepares the device and sets up cublas
void setup_main() {
    srand(0);   // determinism

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

// this defines the generic structure of the main function. This contains everything we can specify without
// knowing the actual test, which is communicated through the TestCase class template.
// Note: Since the TestCase itself has to be a template (needs to run on float, half, bfloat), we get the
//       `template<class> class TestCase` syntax: TestCase is a class template, that takes a single class (floatX) as its
//       template parameter.
// You are not intended to invoke this function manually, nor to write a TestCase yourself; instead, this setup will
// be handled by the macros below.
template<template<class> class TestCase>
int main_fn(int argc, const char** argv) {
    setup_main();

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int dtype = 0;
    if (argc > 2) {
        if (strcmp(argv[2], "f32") == 0) {
            dtype = 0;
        } else if (strcmp(argv[2], "f16") == 0) {
            dtype = 1;
        } else if (strcmp(argv[2], "b16") == 0) {
            dtype = 2;
        } else {
            fprintf(stderr, "Invalid dtype %s", argv[2]);
            exit(EXIT_FAILURE);
        }
    }

    switch (dtype) {
        case 0:
            printf("Using float\n");
            return TestCase<float>::run(kernel_num);
        case 1:
            printf("Using half\n");
            return TestCase<half>::run(kernel_num);
        case 2:
            printf("Using bfloat16\n");
            return TestCase<nv_bfloat16>::run(kernel_num);
        default:
            exit(EXIT_FAILURE);
    }
}

// We have one more challenge to overcome: The kernel dispatch function has to be a template,
// but that prevents us from passing it around as a parameter (cannot take the address of an overloaded function)
// to the `benchmark_kernel` function above.
// What we can pass around instead is a function object; but we don't want to require the test files to implement
// these (which are a bit unnatural, and really just a workaround), so instead we have this macro that creates
// a helper type `dispatcher_s` whose call operator is just forwarding to the original kernel dispatch.
#define DECLARE_DISPATCHER(fptr) \
struct dispatcher_s {            \
    template<class... Args>         \
    void operator()(Args&&... args) {   \
        fptr(std::forward<Args>(args)...);  \
    }   \
}; \

// This macro declares the TestCase class. It is templated over the floating-point type `floatX`, and defines a
// run function to actually run the test.  It also provides a `dispatcher_s` variable of name `dispatcher` (which
// is the name of the original dispatch function template). Therefore, inside this class, the dispatcher function
// template is shadowed by the `dispatcher_s` object of the same name, and `benchmark_kernel` calls work seamlessly.
#define DECLARE_TEST(dispatcher)    \
DECLARE_DISPATCHER(dispatcher)      \
template<class floatX>              \
struct TestCase {                   \
    using x128 = Packed128<floatX>; \
    static int run(int kernel_num); \
    static dispatcher_s dispatcher; \
}


// Finally, the actual test implementation. This macro is designed to have minimal syntax effect, to make it look as
// close to a "regular" test function. Therefore, the implementation below does not specify `int main` but just main---
// the `int` part is expected to be provided as part of the macro invocation in the test file. Similarly, we do not
// auto-generate the `int kernel_num` parameter, but instead expect it to be given as "function arguments" to the macro.
// Thus, in the test file, the only thing that "magically" appears is the floatX type.
#define IMPLEMENT_TEST(...)                 \
main(int argc, const char** argv) {     \
    return main_fn<TestCase>(argc, argv);   \
}                                           \
template<class floatX>                      \
int TestCase<floatX>::run(__VA_ARGS__)
