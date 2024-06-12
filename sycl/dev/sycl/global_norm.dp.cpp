/*
Kernels for a global norm.
Global norm in this context means that we want to calculate a single norm cooperatively using all avalailable SMs, instead
 of multiple norms that can be handled by separate blocks.

Compile example:
nvcc -O3 --use_fast_math global_norm.cu -o global_norm
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>

// turn on bf16 as default, done up here for now
#define ENABLE_BF16
#include "common.h"


float global_norm_cpu(const float* data, size_t count) {
    // accumulate in double so we have an accurate numerical reference
    double acc = 0.0;
    for(size_t i = 0; i < count; ++i) {
        acc  += (double)data[i] * (double)data[i];
    }
    return (float)acc;
}


template<class T>
void norm_kernel1(float* out, const T* data, size_t count,
                  const sycl::nd_item<3> &item_ct1, float *block_result) {
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    // out will be updated atomically from all thread blocks
    size_t index = item_ct1.get_local_id(2) +
                   item_ct1.get_local_range(2) * item_ct1.get_group(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // warp-level reduce
    float warp_result = sycl::reduce_over_group(
        item_ct1.get_sub_group(), accumulator, sycl::plus<float>{});
    block_result[item_ct1.get_sub_group().get_group_linear_id()] = warp_result;
    /*
    DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_sub_group().get_group_linear_id() == 0) {
        /*
        DPCT1007:354: Migration of
        cooperative_groups::thread_block_tile::meta_group_size is not supported.
        */
        float gather =
            item_ct1.get_sub_group().get_local_linear_id() <
                    warp.meta_group_size()
                ? block_result[item_ct1.get_sub_group().get_local_linear_id()]
                : 0.f;
        float block_sum = sycl::reduce_over_group(item_ct1.get_sub_group(),
                                                  gather, sycl::plus<float>{});
        if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                out, block_sum);
        }
    }
}

template<class T>
void norm_kernel2(float* out, const T* data, size_t count,
                  const sycl::nd_item<3> &item_ct1) {
    // concrete example for an A100 GPU (108 SMs, 2048 max threads each)
    // so there are 2048 * 108 = 221,184 threads total
    // say the block_size is 512, then we would launch 432 blocks in total
    // say num_params is ~100M, each thread will process ~500 elements
    // warps reduce with warp-level reduce, we have 221,184/32 = 6,912 warps
    // and then each warp atomicAdd's to global memory, total of 6,912 atomics

    // no shared memory; but one atomic per warp instead of per block

    sycl::group<3> block = item_ct1.get_group();
    sycl::sub_group warp = item_ct1.get_sub_group();

    // out will be updated atomically from all thread blocks
    size_t index = item_ct1.get_local_id(2) +
                   item_ct1.get_local_range(2) * item_ct1.get_group(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // warp-level reduce
    float warp_result = sycl::reduce_over_group(
        item_ct1.get_sub_group(), accumulator, sycl::plus<float>{});
    // and atomic in global buffer
    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            out, warp_result);
    }
}

template<typename T>
void global_norm1(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    /*
    DPCT1049:40: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> block_result_acc_ct1(sycl::range<1>(32),
                                                            cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                norm_kernel1(out, values, count, item_ct1,
                             block_result_acc_ct1
                                 .get_multi_ptr<sycl::access::decorated::no>()
                                 .get());
            });
    });
    /*
    DPCT1010:355: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

template<typename T>
void global_norm2(float* out, const T* values, size_t count, int block_size) {
    // ditto
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    /*
    DPCT1049:41: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            norm_kernel2(out, values, count, item_ct1);
        });
    /*
    DPCT1010:356: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

void global_norm(int kernel_num, float* out, const floatX* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(out, values, count, block_size);
        case 2:
            return global_norm2(out, values, count, block_size);
    }
}

int main(int argc, const char **argv) {
    setup_main();

    int C = 768;
    int L = 12;

    size_t num_params = (size_t)(C * 4*C + C*C) * 2 * L;

    // create host memory of random numbers
    float* inp = make_random_float(num_params);
    // scale them down
    for(size_t i = 0; i < num_params; ++i) {
        inp[i] *= 1e-3;
    }

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    float out = global_norm_cpu(inp, num_params);

    // move to GPU
    float* d_out;
    floatX* d_inp;
    cudaCheck(DPCT_CHECK_ERROR(
        d_out = sycl::malloc_device<float>(1, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(d_inp = sycl::malloc_device<floatX>(
                                   num_params, dpct::get_in_order_queue())));
    cudaCheck(memcpy_convert(d_inp, inp, num_params));

    int block_sizes[] = {32, 64, 128, 256, 512, 768, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        cudaCheck(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memset(d_out, 0, sizeof(float)).wait()));
        global_norm(kernel_num, d_out, d_inp, num_params, block_size);
        validate_result(d_out, &out, "out", 1, 1e-2f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, global_norm,
                                              kernel_num, d_out, d_inp,
                                              num_params, block_size);
        size_t memory_ops = num_params * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(inp);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
}