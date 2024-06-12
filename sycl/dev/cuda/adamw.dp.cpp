/*
Kernels for the AdamW optimizer.

References:
  * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
nvcc -lcublas -lcublasLt adamw.cu -o adamw
nvcc -O3 --use_fast_math -lcublas -lcublasLt adamw.cu -o adamw

./adamw

TODO(general):
amsgrad=True

TODO(perf):
dtype
thread coarsening/ILP
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
//#include "common.h"

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // adapted from: train_gpt2.c

    for (int i = 0; i < num_parameters; i++) {
        float param = params_memory[i];
        float grad = grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// utility functions

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
SYCL_EXTERNAL inline float lerp(float start, float end, float weight) {
    return sycl::fma(weight, end, sycl::fma(-weight, start, start));
}

// naive fused kernel
void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              const sycl::nd_item<3> &item_ct1) {
   int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
           item_ct1.get_local_id(2);
   if (i >= num_parameters) return;  // guard
   // update the first moment (momentum)
   m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
   // update the second moment (RMSprop)
   v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];
   float m_hat = m_memory[i] / beta1_correction;
   float v_hat = v_memory[i] / beta2_correction;
   params_memory[i] -= learning_rate * (m_hat / (sycl::sqrt(v_hat) + eps) +
                                        weight_decay * params_memory[i]);
}

// Slightly more optimized AdamW kernel by:
// * loading data that is accessed more than once into registers,
// * using optimized linear interpolation for the moment updates.
void adamw_kernel2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              const sycl::nd_item<3> &item_ct1) {
   int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
           item_ct1.get_local_id(2);
   if (i >= num_parameters) return;  // guard
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];
   // update the first moment (momentum)
   m = lerp(grad, m, beta1);
   m_memory[i] = m;
   // update the second moment (RMSprop)
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;
   m /= beta1_correction;  // m_hat
   v /= beta2_correction;  // v_hat
   params_memory[i] -= learning_rate * (m / (sycl::sqrt(v) + eps) +
                                        weight_decay * params_memory[i]);
}


// ----------------------------------------------------------------------------
// kernel launcher

// version 1: naive dispatch to naive kernel
void adamw_dispatch1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            adamw_kernel1(params_memory, grads_memory, m_memory, v_memory,
                          num_parameters, learning_rate, beta1, beta2,
                          beta1_correction, beta2_correction, eps, weight_decay,
                          item_ct1);
        });
   
}

// version 2: naive dispatch to slightly optimized kernel
void adamw_dispatch2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            adamw_kernel2(params_memory, grads_memory, m_memory, v_memory,
                          num_parameters, learning_rate, beta1, beta2,
                          beta1_correction, beta2_correction, eps, weight_decay,
                          item_ct1);
        });
    
    
}

void adamw(int kernel_num,
           float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
           float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num) {
        case 1:
            adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}
int main(int argc, char** argv) {
    const long num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;

    srand(time(nullptr));

    // create random data on host
    float* params_memory = make_random_float(num_parameters);
    float* grads_memory = make_random_float(num_parameters);
    float* m_memory = make_random_float_01(num_parameters);
    float* v_memory = make_random_float_01(num_parameters);

    // Allocate device memory
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
    float* d_params = sycl::malloc_device<float>(num_parameters, q);
    float* d_grads = sycl::malloc_device<float>(num_parameters, q);
    float* d_m_memory = sycl::malloc_device<float>(num_parameters, q);
    float* d_v_memory = sycl::malloc_device<float>(num_parameters, q);

    // Copy data to device
    q.memcpy(d_params, params_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_grads, grads_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_m_memory, m_memory, num_parameters * sizeof(float)).wait();
    q.memcpy(d_v_memory, v_memory, num_parameters * sizeof(float)).wait();

    // calculate the CPU reference
    clock_t start = clock();
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_parameters);
    clock_t end = clock();
    double elapsed_time_cpu = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = std::atoi(argv[1]);
    }
    std::cout << "Using kernel " << kernel_num << std::endl;

    // calculate the GPU version
    adamw(kernel_num, d_params, d_grads, d_m_memory, d_v_memory, t, num_parameters,
          learning_rate, beta1, beta2, eps, weight_decay);
    /*
    // compare
    std::cout << "Checking correctness..." << std::endl;
    std::cout << "parameters:" << std::endl;
    validate_result(d_params, params_memory, "params_memory", num_parameters);
    std::cout << "first moment:" << std::endl;
    validate_result(d_m_memory, m_memory, "m_memory", num_parameters);
    std::cout << "second moment:" << std::endl;
    validate_result(d_v_memory, v_memory, "v_memory", num_parameters);
    std::cout << "All results match." << std::endl;

    // benchmark the kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(
        repeat_times,
        adamw,
        kernel_num, q,
        d_params, d_grads, d_m_memory, d_v_memory, t, num_parameters,
        learning_rate, beta1, beta2, eps, weight_decay
    );

    std::cout << "time gpu " << elapsed_time << " ms" << std::endl;
    std::cout << "time cpu " << elapsed_time_cpu << " ms" << std::endl;
    */
    // Free device memory
    sycl::free(d_params, q);
    sycl::free(d_grads, q);
    sycl::free(d_m_memory, q);
    sycl::free(d_v_memory, q);

    // cleanup
    delete[] params_memory;
    delete[] grads_memory;
    delete[] m_memory;
    delete[] v_memory;
    

    return 0;
}


