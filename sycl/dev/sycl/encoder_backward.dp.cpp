/*
Kernels for the positional encoder forward pass in GPT-2.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_backward.cu -o encoder_backward

version 1 is naive port from CPU code to kernel
parallelizes over B,T,C, uses atomics to add to dwte, dwpe
./encoder_backward 1

version 2 is another naive port
parallelizes over C, loops over B,T; much slower than version 1
./encoder_backward 2
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 positional encoder forward pass
void encoder_backward_cpu(float* dwte, float* dwpe,
                            float* dout, int* inp,
                            int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive implementation with atomics
void encoder_backward_kernel1(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C,
                                        const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        const float* dout_btc = dout + b * T * C + t * C + c;
        float* dwte_ix = dwte + ix * C + c;
        float* dwpe_tc = dwpe + t * C + c;

        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            dwte_ix, *dout_btc);
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            dwpe_tc, *dout_btc);
    }
}

// naive implementation that parallelizes over C and loops over B,T
// but it gets rid of atomics
void encoder_backward_kernel2(float* dwte, float* dwpe,
                                        const float* dout, const int* inp,
                                        int B, int T, int C,
                                        const sycl::nd_item<3> &item_ct1) {
    int c = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (c >= C) { return; } // guard
    int BT = B * T;
    for (int i = 0; i < BT; i++) {
        int t = i % T;
        int ix = inp[i];
        float dout_btc = dout[i * C + c];
        dwte[ix * C + c] += dout_btc;
        dwpe[t * C + c] += dout_btc;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void encoder_backward1(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:135: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            encoder_backward_kernel1(dwte, dwpe, dout, inp, B, T, C, item_ct1);
        });
    /*
    DPCT1010:480: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void encoder_backward2(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    const int grid_size = ceil_div(C, block_size);
    /*
    DPCT1049:136: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            encoder_backward_kernel2(dwte, dwpe, dout, inp, B, T, C, item_ct1);
        });
    /*
    DPCT1010:481: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void encoder_backward(int kernel_num,
                     float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_backward1(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        case 2:
            encoder_backward2(dwte, dwpe, dout, inp, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;

    int deviceIdx = 0;
    /*
    DPCT1093:482: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));

    // create host memory of random numbers
    float* dout = make_random_float(B * T * C);
    int* inp = make_random_int(B * T, V);
    float* dwte = make_zeros_float(V * C);
    float* dwpe = make_zeros_float(T * C);

    // move to GPU
    float* d_dout;
    int* d_inp;
    float* d_dwte;
    float* d_dwpe;
    cudaCheck(DPCT_CHECK_ERROR(d_dout = sycl::malloc_device<float>(
                                   B * T * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        d_inp = sycl::malloc_device<int>(B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dwte = sycl::malloc_device<float>(
                                   V * C, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dwpe = sycl::malloc_device<float>(
                                   T * C, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dout, dout, B * T * C * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_inp, inp, B * T * sizeof(int))
                                   .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    encoder_backward_cpu(dwte, dwpe, dout, inp, B, T, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memset(d_dwte, 0, V * C * sizeof(float))
                                       .wait()));
        cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memset(d_dwpe, 0, T * C * sizeof(float))
                                       .wait()));
        printf("Checking block size %d.\n", block_size);
        encoder_backward(kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        validate_result(d_dwte, dwte, "dwte", V * C, 1e-5f);
        validate_result(d_dwpe, dwpe, "dwpe", T * C, 1e-5f);
    }
    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_backward,
                                              kernel_num, d_dwte, d_dwpe, d_dout, d_inp, B, T, C, block_size);
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(dout);
    free(inp);
    free(dwte);
    free(dwpe);
    dpct::dpct_free(d_dout, dpct::get_in_order_queue());
    dpct::dpct_free(d_inp, dpct::get_in_order_queue());
    dpct::dpct_free(d_dwte, dpct::get_in_order_queue());
    dpct::dpct_free(d_dwpe, dpct::get_in_order_queue());

    return 0;
}
