/*
Kernels for matmul forward pass.
It's advised to use OpenMP here because the CPU implementation is fairly slow otherwise

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 calls cuBLAS, very fast
OMP_NUM_THREADS=32 ./matmul_forward 2

version 3 calls cuBLASLt, should be even faster
OMP_NUM_THREADS=32 ./matmul_forward 3
*/

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <dpct/blas_utils.hpp>

#include <omp.h>
#include "common.h"
#include <dpct/lib_common_utils.hpp>

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC,
                                       const sycl::nd_item<3> &item_ct1) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    int oc = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
             item_ct1.get_local_id(1);
    if (bt < BT && oc < OC) {
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + bt * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a cuBLAS function that does this
void add_bias(float* out, const float* bias, int B, int T, int OC,
              const sycl::nd_item<3> &item_ct1) {
    int idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    int stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    sycl::range<3> gridDim(ceil_div(B * T, sqrt_block_size),
                           ceil_div(OC, sqrt_block_size));
    sycl::range<3> blockDim(1, sqrt_block_size, sqrt_block_size);
    /*
    DPCT1049:109: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        int B_T_ct4 = B * T;

        cgh.parallel_for(sycl::nd_range<3>(gridDim * blockDim, blockDim),
                         [=](sycl::nd_item<3> item_ct1) {
                             matmul_forward_kernel1(out, inp, weight, bias,
                                                    B_T_ct4, C, OC, item_ct1);
                         });
    });
    /*
    DPCT1010:443: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    cudaCheck(0);
}

// kernel 2 calls cuBLAS, which should be very efficient
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublas_handle->get_queue(), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, OC, B * T, C,
        dpct::get_value(&alpha, cublas_handle->get_queue()), weight, C, inp, C,
        dpct::get_value(&beta, cublas_handle->get_queue()), out, OC)));
    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        /*
        DPCT1049:110: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) {
                add_bias(out, bias, B, T, OC, item_ct1);
            });
        /*
        DPCT1010:444: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        cudaCheck(0);
    }
}

// uses cublasLt to fuse the bias and gelu
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    oneapi::mkl::transpose opNoTranspose = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose opTranspose = oneapi::mkl::transpose::trans;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;
    }
    /*
    DPCT1007:445: Migration of cublasLtMatmulDescCreate is not supported.
    */
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type,
                                         dpct::library_data_t::real_float));
    /*
    DPCT1007:446: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose,
        sizeof(opTranspose)));
    /*
    DPCT1007:447: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose,
        sizeof(opNoTranspose)));
    /*
    DPCT1007:448: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias,
        sizeof(epilogueBias)));
    /*
    DPCT1007:449: Migration of cublasLtMatmulDescSetAttribute is not supported.
    */
    cublasCheck(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    /*
    DPCT1007:450: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &weightLayout, dpct::library_data_t::real_float, C, OC, C));
    /*
    DPCT1007:451: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &inputLayout, dpct::library_data_t::real_float, C, B * T, C));
    /*
    DPCT1007:452: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &outputLayout, dpct::library_data_t::real_float, OC, B * T, OC));
    /*
    DPCT1007:453: Migration of cublasLtMatrixLayoutCreate is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutCreate(
        &biasLayout, dpct::library_data_t::real_float, OC, 1, OC));

    // create a preference handle with specified max workspace
    /*
    DPCT1007:454: Migration of cublasLtMatmulPreferenceCreate is not supported.
    */
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    /*
    DPCT1007:455: Migration of cublasLtMatmulPreferenceSetAttribute is not
    supported.
    */
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    /*
    DPCT1007:456: Migration of cublasLtMatmulAlgoGetHeuristic is not supported.
    */
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, weightLayout, inputLayout, outputLayout,
        outputLayout, preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    /*
    DPCT1007:457: Migration of cublasLtMatmul is not supported.
    */
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, weight,
                               weightLayout, inp, inputLayout, &beta, out,
                               outputLayout, out, outputLayout, &heuristic.algo,
                               cublaslt_workspace, cublaslt_workspace_size,
                               &dpct::get_in_order_queue()));

    // cleanups
    /*
    DPCT1007:458: Migration of cublasLtMatmulPreferenceDestroy is not supported.
    */
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    /*
    DPCT1007:459: Migration of cublasLtMatmulDescDestroy is not supported.
    */
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    /*
    DPCT1007:460: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    /*
    DPCT1007:461: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    /*
    DPCT1007:462: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    /*
    DPCT1007:463: Migration of cublasLtMatrixLayoutDestroy is not supported.
    */
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
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
    DPCT1093:464: The "deviceIdx" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    cudaCheck(DPCT_CHECK_ERROR(dpct::select_device(deviceIdx)));
    dpct::device_info deviceProp;
    dpct::get_device_info(deviceProp,
                          dpct::dev_mgr::instance().get_device(deviceIdx));
    printf("Device %d: %s\n", deviceIdx, deviceProp.get_name());

    // setup cuBLAS and cuBLASLt
    cublasCheck(DPCT_CHECK_ERROR(cublas_handle = new dpct::blas::descriptor()));
    /*
    DPCT1007:465: Migration of cublasLtCreate is not supported.
    */
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    /*
    DPCT1005:466: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    int enable_tf32 = deviceProp.get_major_version() >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    int cublas_math_mode =
        enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    /*
    DPCT1027:467: The call to cublasSetMathMode was replaced with 0 because this
    functionality is redundant in SYCL.
    */
    cublasCheck(0);
    // setup the (global) cuBLASLt workspace
    cudaCheck(DPCT_CHECK_ERROR(
        cublaslt_workspace = (void *)sycl::malloc_device(
            cublaslt_workspace_size, dpct::get_in_order_queue())));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_inp, inp, B * T * C * sizeof(float))
                             .wait()));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(d_weight, weight, C * OC * sizeof(float))
                             .wait()));
    cudaCheck(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(d_bias, bias, OC * sizeof(float))
                                   .wait()));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_out, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_inp, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(d_weight, dpct::get_in_order_queue())));
    cudaCheck(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_bias, dpct::get_in_order_queue())));
    cudaCheck(DPCT_CHECK_ERROR(
        dpct::dpct_free(cublaslt_workspace, dpct::get_in_order_queue())));
    cublasCheck(DPCT_CHECK_ERROR(delete (cublas_handle)));
    /*
    DPCT1007:468: Migration of cublasLtDestroy is not supported.
    */
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    return 0;
}