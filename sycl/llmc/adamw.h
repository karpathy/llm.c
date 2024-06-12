/*
AdamW kernel
*/

// llmc internal imports
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "sycl_common.h"
#include "sycl_utils.h"

// ----------------------------------------------------------------------------
// SYCL kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
SYCL_EXTERNAL float lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

template <typename Tp, typename Tg>
void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float                                          weight_decay,
                              float grad_scale, unsigned int seed,
                              const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // fetch the old value of this parameter as a float, from either source
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // update this parameter
    float param = old_param - (learning_rate * (m / (sycl::sqrt(v) + eps) +
                                                weight_decay * old_param));
    // update our low precision version of the parameters using stochastic rounding
    // this will be used in the next forward pass
    // TODO: simply doing `params_memory[i] = (floatX)param;` breaks everything (why?)
    unsigned int random =
        Get2dNoiseUint(item_ct1.get_local_id(2), item_ct1.get_group(2), seed);
    //stochastic_rounding(param, &params_memory[idx], random, item_ct1);
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
void adamw(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float                                          weight_decay,
                              float grad_scale, unsigned int seed, sycl::queue &q) {
    
    
    //dpct::has_capability_or_fail(q.get_device(), {sycl::aspect::fp16});
    q.submit(
		[&](sycl::handler &cgh) {
   
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_parameters), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1) {
            adamw_kernel3<Tp, Tg>( params_memory, master_params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate,
                          beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed, item_ct1);
        });
    });
    q.wait();
    
   }

int main(){

    const long num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;
    const int seed =42;
    const float grad_scale = 1.0f;
    const float beta1_correction = 1.0f;
    const float beta2_correction = 0.99f;

    srand(time(nullptr));

    // create random data on host
    float* master_params_memory = make_random_float(num_parameters);
    float* params_memory = make_random_float(num_parameters);
    float* grads_memory = make_random_float(num_parameters);
    float* m_memory = make_random_float(num_parameters);
    float* v_memory = make_random_float(num_parameters);

    // Allocate device memory
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q = dev_ct1.in_order_queue();
    sycl::context ctx = q.get_context();

    //sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
    float* d_master_params_memory = sycl::malloc_device<float>(num_parameters, q);
    float* d_params_memory = sycl::malloc_device<float>(num_parameters, q);
    float* d_grads_memory = sycl::malloc_device<float>(num_parameters, q);
    float* d_m_memory = sycl::malloc_device<float>(num_parameters, q);
    float* d_v_memory = sycl::malloc_device<float>(num_parameters, q);

    // Copy data to device
    q.memcpy(d_master_params_memory, master_params_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_params_memory, params_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_grads_memory, grads_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_m_memory, m_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_v_memory, v_memory, num_parameters * sizeof(float)).wait();

    
    adamw<float, float>(d_params_memory, d_master_params_memory, d_grads_memory, d_m_memory, d_v_memory, num_parameters,
          learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed, q);

    std::cout << "parameters:" << num_parameters<< std::endl;
    validate_result(d_params_memory, params_memory, "master_params_memory", num_parameters);
    validate_result(d_master_params_memory, params_memory, "params_memory", num_parameters);
    validate_result(d_grads_memory, grads_memory, "grads_memory", num_parameters);
    validate_result(d_m_memory, m_memory, "m_memory", num_parameters);
    validate_result(d_v_memory, v_memory, "v_memory", num_parameters);
    
    // Free device memory
    sycl::free(d_master_params_memory, q);
    sycl::free(d_params_memory, q);
    sycl::free(d_grads_memory, q);
    sycl::free(d_m_memory, q);
    sycl::free(d_v_memory, q);

    // cleanup
    delete[] master_params_memory;
    delete[] params_memory;
    delete[] grads_memory;
    delete[] m_memory;
    delete[] v_memory;

  return 0;

}