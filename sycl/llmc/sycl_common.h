/*
Common utilities for CUDA code.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
extern dpct::device_info deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#if DPCT_COMPATIBILITY_TEMP == 800 || DPCT_COMPATIBILITY_TEMP >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking
void inline cudaCheck(dpct::err0 error, const char *file, int line){

};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

enum PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
//#if defined(ENABLE_FP32)
typedef float floatX;
//#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
//#elif defined(ENABLE_FP16)
//typedef sycl::half floatX;
//#define PRECISION_MODE PRECISION_FP16
//#else // Default to bfloat16
//typedef sycl::ext::oneapi::bfloat16 floatX;
//#define PRECISION_MODE PRECISION_BF16
//#endif

// ----------------------------------------------------------------------------



#endif // CUDA_COMMON_H