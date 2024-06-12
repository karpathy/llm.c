/*
Global norm, used in gradient clipping
*/
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// CUDA kernels

template<class T>
void global_norm_squared_kernel(float* out, const T* data, size_t count,
                                const sycl::nd_item<3> &item_ct1,
                                float *shared_val) {
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.
    // out will be updated atomically from all thread blocks. It is a float, so the
    // atomic op is unproblematic
    size_t index = item_ct1.get_local_id(2) +
                   item_ct1.get_local_range(2) * item_ct1.get_group(2);
    size_t grid_width =
        item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // warp-level reduce
    float block_sum =
        blockReduce<warpReduceSum>(accumulator, item_ct1, shared_val);
    if (item_ct1.get_local_id(2) == 0) {
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            out, block_sum);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

template<typename T>
void global_norm_squared(float* out, const T* values, size_t count, sycl::queue &q) {
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    
try{
    dpct::device_info deviceProp;
    const int grid_size = deviceProp.get_max_work_items_per_compute_unit() *
                          deviceProp.get_max_compute_units() / block_size;
    //assert(grid_size > 0);      // gives a better error than letting the call below fail
    // initialize out with zero
    q.memset(out, 0, sizeof(float)).wait();
    q.submit([&](sycl::handler &cgh) {
        
        sycl::local_accessor<float, 1> shared_val_acc_ct1(
            sycl::range<1>(32 /*WARP_SIZE*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                global_norm_squared_kernel(
                    out, values, count, item_ct1,
                    shared_val_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
}
  
int main(int argc, char** argv) {
    int C = 768;
    int L = 12;

    size_t num_params = (size_t)(C * 4 * C + C * C) * 2 * L;

    // Create host memory of random numbers
    float* inp = make_random_float(num_params);
    float* out = make_random_float(num_params);
    // Scale them down
    for (size_t i = 0; i < num_params; ++i) {
        inp[i] *= 1e-3;
    }
    
     // SYCL queue
    sycl::queue q;
    
    float* d_out = sycl::malloc_device<float>(num_params, q);
    float* d_inp = sycl::malloc_device<float>(num_params, q);

    q.memcpy(d_inp, inp, num_params * sizeof(float)).wait();
    q.memcpy(d_out, out, num_params * sizeof(float)).wait();
    // Time the kernel at different block sizes
    
    global_norm_squared(d_out, d_inp, num_params, q);
    validate_result(d_out, out, "out", 1, 1e-2f);
    

    std::cout << "All results match. Starting benchmarks.\n\n";


    return 0;
}
   


