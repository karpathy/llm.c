/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "sycl_common.h"
#include "sycl_utils.h"
//#include "cublas_common.h"
#include <dpct/lib_common_utils.hpp>
#include <dpct/dpl_utils.hpp>
#include <dpct/dpl_extras/dpcpp_extensions.h>
#include <oneapi/dnnl/dnnl.hpp> 
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <dpct/blas_utils.hpp>

// ----------------------------------------------------------------------------
// SYCL kernels

using namespace dnnl;

template<typename OutFloat, bool UseAuxBuffer>
void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>,
                                             const sycl::nd_item<3> &item_ct1,
                                             sycl::local_accessor<float, 3> sub_results) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(0);
    assert(0);

    int warp_d = (int)item_ct1.get_local_id(2);
    int warp_c = (int)item_ct1.get_local_id(1);
    int block_d = (int)item_ct1.get_local_id(0);

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = item_ct1.get_group(2) * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * item_ct1.get_local_range(0);

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = item_ct1.get_group(1) * bt_per_block + local_bt;
             idx < B * T; idx += item_ct1.get_group_range(1) * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        
        v += dpct::shift_sub_group_left(item_ct1.get_sub_group(), v, 1, 4);
        
        v += dpct::shift_sub_group_left(item_ct1.get_sub_group(), v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += item_ct1.get_local_range(0)) {
        float a = 0.f;
        for (int r = warp_d; r < item_ct1.get_local_range(0); r += bdx) {
            float v = sub_results[k][r][warp_c];
            
            v += dpct::shift_sub_group_left(item_ct1.get_sub_group(), v, 1, 4);
            
            v += dpct::shift_sub_group_left(item_ct1.get_sub_group(), v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + item_ct1.get_group(1) * OC] = a;
            }
        }
    }
}

SYCL_EXTERNAL void reduce_add_sum_kernel(floatX *dst, const float *src,
                                         size_t n, size_t m,
                                         const sycl::nd_item<3> &item_ct1) {
    const size_t idx = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                        item_ct1.get_local_id(2)) *
                       f128::size;
    assert(0);
    if (idx < n) {
        f128 acc;
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, sycl::queue &q_ct1) {
try{    
    int has_bias = (bias != NULL);

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    // these need to be in FP16 if and only if alpha/beta are CUBLAS_COMPUTE_16F
    const float alpha = 1.0f, beta = 0.0f;

    int returnedResults = 0;
    
    
    using tag = memory::format_tag;
    using dt = memory::data_type;
    dpct::device_ext &dev = dpct::get_current_device();
    sycl::context ctx = q_ct1.get_context();
    //auto dev = sycl::device(sycl::gpu_selector_v);
    //auto ctx = sycl::context(dev);
     
    dnnl::engine engine = sycl_interop::make_engine(dev, ctx);
    // column major 
    const memory::dims weight_strides = memory::dims {1, C};
    const auto weight_md = memory::desc({OC, C}, dt::f32, weight_strides);
    const memory::dims input_strides = memory::dims {C, 1};
    const auto input_md = memory::desc({C, B * T}, dt::f32, input_strides);
    const memory::dims output_strides = memory::dims {OC, 1};
    const auto output_md =  memory::desc({OC, B * T}, dt::f32, output_strides);
    
    //memory align
    memory weight_mem(weight_md, engine);
    memory input_mem(input_md, engine);
    memory output_mem(output_md, engine);
    
    
    //create dnnl stream
    //auto q_ct1 = sycl::queue(ctx, dev);
    dnnl::stream stream = sycl_interop::make_stream(engine, q_ct1);
    
    primitive_attr attr;
    
    
    auto matmul_pd = matmul::primitive_desc(engine, weight_md, input_md, output_md, attr);
    auto matmul_prim = matmul(matmul_pd);
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, weight_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, input_mem});
    matmul_args.insert({DNNL_ARG_DST, output_mem});

    
    matmul_prim.execute(stream, matmul_args);
    stream.wait();

   
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, sycl::queue &q) {
try{
    dpct::device_info deviceProp;
    float one = 1.0f, zero = 0.0f;

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size =
            deviceProp.get_max_work_items_per_compute_unit() == 1536 ? 768
                                                                     : 1024;

        sycl::range<3> block_dim = {(unsigned)block_size / WARP_SIZE, 8, 4};
        const int OC_per_warp = block_dim[1] * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y =
            std::max(1, deviceProp.get_max_work_items_per_compute_unit() *
                            deviceProp.get_max_compute_units() /
                            (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            
            
            q.submit([&](sycl::handler &cgh) {
                
                sycl::local_accessor<float, 3> sub_results_acc_ct1(
                    sycl::range<3>(8 /*x128::size*/, 32 /*WARP_SIZE*/,
                                   8 /*bdy*/),
                    cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, grid_size_y, grid_size_x) * block_dim,
                        block_dim),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            matmul_backward_bias_kernel9(
                                dbias, dout, B, T, OC,
                                std::bool_constant<false>{}, item_ct1,
                                sub_results_acc_ct1);
                        });
            });
           
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            
            q.submit([&](sycl::handler &cgh) {
                
                sycl::local_accessor<float, 3> sub_results_acc_ct1(
                    sycl::range<3>(8 /*x128::size*/, 32 /*WARP_SIZE*/,
                                   8 /*bdy*/),
                    cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, grid_size_y, grid_size_x) * block_dim,
                        block_dim),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            matmul_backward_bias_kernel9(
                                dbias_buffer, dout, B, T, OC,
                                std::bool_constant<true>{}, item_ct1,
                                sub_results_acc_ct1);
                        });
            });
            
            
            dpct::get_in_order_queue().parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, CEIL_DIV(OC, 256 * f128::size)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    reduce_add_sum_kernel(dbias, dbias_buffer, OC, grid_size_y,
                                          item_ct1);
                });
            
        }
    }

    // backward to input, uses = in the backward pass (set the gradient)
    
        dpct::gemm(q, oneapi::mkl::transpose::nontrans,
                   oneapi::mkl::transpose::nontrans, C, B * T, OC, &one, weight,
                   CUBLAS_LOWP, C, dout, CUBLAS_LOWP, OC, &zero, dinp,
                   CUBLAS_LOWP, C, cublas_compute);
    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    dpct::gemm(
        q, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, C, OC, B * T, &one, inp, CUBLAS_LOWP, C,
        dout, CUBLAS_LOWP, OC, &one, dweight, CUBLAS_LOWP, C, cublas_compute);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

   
}

int main(){



  return 0;
}
