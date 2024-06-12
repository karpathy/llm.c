/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_softmax_backward 1
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_softmax_backward_cpu(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel that just parallelizes over B,T,V
void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const sycl::nd_item<3> &item_ct1) {
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_softmax_backward1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    const int N = B * T * V;
    const int grid_size = ceil_div(N, block_size);
    /*
    DPCT1049:82: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            crossentropy_softmax_backward_kernel1(dlogits, dlosses, probs,
                                                  targets, B, T, V, item_ct1);
        });
    /*
    DPCT1010:423: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel version dispatch
void crossentropy_softmax_backward(int kernel_num,
                           float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_softmax_backward1(dlogits, dlosses, probs, targets, B, T, V, block_size);
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
    int V = 50257;

    int deviceIdx = 0;
    /*
    DPCT1093:424: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));

    // create host memory of random numbers
    float* probs = make_random_float(B * T * V);
    int* targets = make_random_int(B * T, V);
    float* dlosses = make_random_float(B * T);
    float* dlogits = make_zeros_float(B * T * V);

    // move to GPU
    float* d_probs;
    int* d_targets;
    float* d_dlosses;
    float* d_dlogits;
    cudaCheck(DPCT_CHECK_ERROR(d_probs = sycl::malloc_device<float>(
                                   B * T * V, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_targets = sycl::malloc_device<int>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dlosses = sycl::malloc_device<float>(
                                   B * T, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_dlogits = sycl::malloc_device<float>(
                                   B * T * V, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_probs, probs, B * T * V * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_targets, targets, B * T * sizeof(int))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dlosses, dlosses, B * T * sizeof(float))
                             .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue()
                .memset(d_dlogits, 0, B * T * V * sizeof(float))
                .wait()));
        printf("Checking block size %d.\n", block_size);
        crossentropy_softmax_backward(kernel_num, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_dlogits, dlogits, "dlogits", B * T * V, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                              kernel_num, d_dlogits, d_dlosses, d_probs, d_targets,
                                              B, T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_probs, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_targets, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_dlosses, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_dlogits, dpct::get_in_order_queue())));

    return 0;
}