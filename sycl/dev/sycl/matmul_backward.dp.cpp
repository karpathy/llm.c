/*
Kernels for matmul backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dpct/blas_utils.hpp>

#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC,
                                       const sycl::nd_item<3> &item_ct1) {
    int o = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (o < OC) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

// use shared memory and coarsening + reductions
void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC,
                                        const sycl::nd_item<3> &item_ct1,
                                        uint8_t *dpct_local) {
    auto shared = (float *)dpct_local;
    int o = item_ct1.get_group(2);      // range [0, OC)
    int tid = item_ct1.get_local_id(2); // range [0, block_size)
    int block_size = item_ct1.get_local_range(2);
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        /*
        DPCT1118:169: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = shared[0];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // backward to input
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, C, B * T, OC,
        dpct::get_value(&alpha, cublas_handle->get_queue()), weight, C, dout,
        OC, dpct::get_value(&beta, cublas_handle->get_queue()), dinp, C)));
    // backward to weight
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, C, OC, B * T,
        dpct::get_value(&alpha, cublas_handle->get_queue()), inp, C, dout, OC,
        dpct::get_value(&beta, cublas_handle->get_queue()), dweight, C)));
    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        sycl::range<3> block_dim(1, 1, block_size);
        sycl::range<3> grid_dim(1, 1, OC);
        /*
        DPCT1083:171: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        size_t shared_mem_size = block_size * sizeof(float);
        /*
        DPCT1049:170: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            dpct::has_capability_or_fail(
                dpct::get_in_order_queue().get_device(), {sycl::aspect::fp64});

            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem_size), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                    [=](sycl::nd_item<3> item_ct1) {
                        matmul_backward_bias_kernel_faster(
                            dbias, dout, B, T, OC, item_ct1,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
}

void matmul_backward(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
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
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    /*
    DPCT1093:515: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    dpct::get_device_info(deviceProp,
                          dpct::dev_mgr::instance().get_device(deviceIdx));
    printf("Device %d: %s\n", deviceIdx, deviceProp.get_name());

    // setup cuBLAS and its mathmodes, ensure fp32
    int enable_tf32 = 0; // use fp32 to get accurate results for checking w.r.t. CPU
    cublasCheck(DPCT_CHECK_ERROR(cublas_handle = new dpct::blas::descriptor()));
    printf("enable_tf32: %d\n", enable_tf32);
    int cublas_math_mode =
        enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    /*
    DPCT1027:516: The call to cublasSetMathMode was replaced with 0 because this
    functionality is redundant in SYCL.
    */
    cublasCheck(0);

    // create host memory of random numbers
    float* dinp = make_zeros_float(B * T * C);
    float* dweight = make_zeros_float(OC * C);
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_ones, OC * sizeof(float)));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dinp, dinp, B * T * C * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dweight, dweight, OC * C * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_dbias, dbias, OC * sizeof(float))
                                   .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_dout, dout, B * T * OC * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, B * T * C * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_weight, weight, OC * C * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_ones, ones, OC * sizeof(float))
                                   .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    printf("time %.4f ms\n", elapsed_time);

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    free(ones);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_dinp, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_dweight, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_dbias, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_dout, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_weight, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_ones, dpct::get_in_order_queue())));
    cublasCheck(DPCT_CHECK_ERROR(delete (cublas_handle)));

    return 0;
}