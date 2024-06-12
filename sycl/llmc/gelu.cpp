/*
(Approximate) GeLU non-linearity layer
*/
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

#define GELU_SCALING_FACTOR sycl::sqrt((float)(2.0f / M_PI))
void gelu_forward_kernel2(floatX* out, const floatX* inp,
                          const sycl::nd_item<3> &item_ct1) {
    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) *
              x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] =
            (floatX)(0.5f * xi *
                     (1.0f + sycl::tanh(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp,
                                  const sycl::nd_item<3> &item_ct1) {
    int idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2)) *
              x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = sycl::tanh(tanh_arg);
        float coshf_out = sycl::cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(floatX* out, const floatX* inp, int N, sycl::queue &q) {
    
try{    
    const int block_size = 512;
    assert(N % block_size == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_forward_kernel2(out, inp, item_ct1);
        });
      });
  }
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
  }
}

void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, sycl::queue &q) {
    
    const int block_size = 128;
    assert(N % block_size == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            gelu_backward_inplace_kernel(d_in_out, inp, item_ct1);
        });
      });
    
}

int main(){


  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q = dev_ct1.in_order_queue();
  sycl::context ctx = q.get_context();
  
  int B = 8;
  int T = 1024;
  int C = 768;
  int N = B * T * C;

  // Create host memory of random numbers
  
  float* out = new float[N];
  float* inp = make_random_float(N);
  float tol =1e-2;
  
  auto d_out = sycl::malloc_device<floatX>(B * T * C, q);
  auto d_inp = sycl::malloc_device<floatX>(B * T * C, q);
  q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
  q.memcpy(d_out, out, B * T * C * sizeof(float)).wait();
  
  gelu_forward(d_out, d_inp, N, q);
  
  validate_result(d_out, out, "out", B * T * C, tol);
  
  sycl::free(d_out, q);
  sycl::free(d_inp, q);
  
  float* dinp = new float[N];
  inp = make_random_float(N);
  out = make_random_float(N);
  
  d_out = sycl::malloc_device<floatX>(B * T * C, q);
  d_inp = sycl::malloc_device<floatX>(B * T * C, q);
  auto d_dinp = sycl::malloc_device<floatX>(B * T * C, q);
  q.memcpy(d_inp, inp, B * T * C * sizeof(float)).wait();
  q.memcpy(d_out, out, B * T * C * sizeof(float)).wait();
  q.memcpy(d_dinp, dinp, B * T * C * sizeof(float)).wait();
  
  gelu_backward_inplace(d_dinp, d_inp, N, q);
  
  validate_result(d_dinp, dinp, "out", B * T * C, tol);
  
  sycl::free(d_dinp, q);
  sycl::free(d_inp, q);

  return 0;


}