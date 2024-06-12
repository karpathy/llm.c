/*
Kernels for gelu forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_forward.cu -o gelu_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_forward 1

version 2 is bfloat16 with the Packed128 data structure
./gelu_forward 2
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

#define ENABLE_BF16
#include "common.h"
#include <cmath>

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sycl::sqrt((float)(2.0f / M_PI))

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
void gelu_forward_kernel1(floatX* out, const floatX* inp, int N,
                          const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] =
            0.5f * xi * (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

// elementwise ops are nice and ez
void gelu_forward_kernel2(floatX* out, const floatX* inp, int N,
                          const sycl::nd_item<3> &item_ct1) {
    int i = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2)) *
            x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi *
                                     (1.0f + sycl::tanh(GELU_SCALING_FACTOR *
                                                        (xi + cube))));
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + i, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:172: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_forward_kernel1(out, inp, N, item_ct1);
        });
    /*
    DPCT1010:517: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void gelu_forward2(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    /*
    DPCT1049:173: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_forward_kernel2(out, inp, N, item_ct1);
        });
    /*
    DPCT1010:518: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void gelu_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);

    // move to GPU
    floatX* d_out;
    floatX* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(d_out = sycl::malloc_device<floatX>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<floatX>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "out", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * (int)sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);

    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
    return 0;
}