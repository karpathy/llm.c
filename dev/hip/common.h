#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

template <class T>
T ceil_div(T dividend, T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(hipError_t error, const char *file, int line)
{
    if (error != hipSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               hipGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(hipblasStatus_t status, const char *file, int line)
{
    if (status != HIPBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status)                        \
    {                                              \
        cublasCheck((status), __FILE__, __LINE__); \
    }

// ----------------------------------------------------------------------------
// random utils

float *make_random_float_01(int N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float *make_random_float(int N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int *make_random_int(int N, int V)
{
    int *arr = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float *make_zeros_float(int N)
{
    float *arr = (float *)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template <class T>
void validate_result(T *device_result, const T *cpu_reference, const char *name, std::size_t num_elements, T tolerance = 1e-4)
{
    T *out_gpu = (T *)malloc(num_elements * sizeof(T));
    cudaCheck(hipMemcpy(out_gpu, device_result, num_elements * sizeof(T), hipMemcpyDeviceToHost));
    for (int i = 0; i < num_elements; i++)
    {
        // print the first few comparisons
        if (i < 5)
        {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance)
        {
            printf("Mismatch of %s at %d: %f vs %f\n", name, i, cpu_reference[i], out_gpu[i]);
            free(out_gpu);
            exit(EXIT_FAILURE);
        }
    }

    // reset the result pointer, so we can chain multiple tests and don't miss trivial errors,
    // like the kernel not writing to part of the result.
    hipMemset(device_result, 0, num_elements * sizeof(T));
    free(out_gpu);
}

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs &&...kernel_args)
{
    hipEvent_t start, stop;
    cudaCheck(hipEventCreate(&start));
    cudaCheck(hipEventCreate(&stop));
    cudaCheck(hipEventRecord(start, nullptr));
    for (int i = 0; i < repeats; i++)
    {
        kernel(std::forward<KernelArgs>(kernel_args)...);
    }
    cudaCheck(hipEventRecord(stop, nullptr));
    cudaCheck(hipEventSynchronize(start));
    cudaCheck(hipEventSynchronize(stop));
    float elapsed_time;
    cudaCheck(hipEventElapsedTime(&elapsed_time, start, stop));

    return elapsed_time / repeats;
}