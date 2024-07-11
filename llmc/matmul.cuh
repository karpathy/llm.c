/*
Matrix Multiplication, with help from cuBLASLt
*/
#include <assert.h>
#include <type_traits>      // std::bool_constant
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"
// GELU can be either fused (cublasLt) or non-fused (gelu.h)
#include "gelu.cuh"
#include "copy_and_fp8.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
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

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
template <typename Td=floatX, typename Ta=floatX, typename Tb=floatX>
void matmul_cublaslt(Td* d, const Ta* a, const Tb* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, void* pre_gelu=NULL, bool backward=false, float* absmax_a=NULL, float* absmax_b=NULL)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL);
    bool has_gelu = (pre_gelu != NULL);

    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    auto a_precision = CUBLAS_LOWP;
    auto b_precision = CUBLAS_LOWP;
    auto d_precision = CUBLAS_LOWP;
    bool recompute_due_to_d_absmax = false; // sigh
    Ta* a_new = (Ta*)a;
    Tb* b_new = (Tb*)b;

    #if FORCE_FP8_MATMUL
    // hack - todo - only skip embedding matmul (end of forward, start of backward)
    bool allow_fp8 = (m < 50000 && n < 50000 && k < 50000);

    if (batch_count == 0 && allow_fp8) {
        a_precision = CUDA_R_8F_E4M3;
        if (!std::is_same<Ta, __nv_fp8_e4m3>::value) {
            __nv_fp8_e4m3* a_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e4m3>(m*k);

            float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(a, k*m, b, SCALE_A);
            float *next_absmax = absmax_tracker.getNextAbsMaxPtr(a, k*m, b);
            copy_or_transpose<true> (!transA, a_fp8, a, m, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax);
            absmax_a = (float*)calculated_from_absmax+DESCALE_OFFSET;
            a_new = (Ta*)a_fp8;
        } else {
            assert(transA);
        }

        if (backward) {
            b_precision = CUDA_R_8F_E5M2;
            if (!std::is_same<Tb, __nv_fp8_e5m2>::value) {
                __nv_fp8_e5m2* b_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e5m2>(n*k);

                float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(b, k*n, a, SCALE_BACKWARDS_B);
                float *next_absmax = absmax_tracker.getNextAbsMaxPtr(b, k*n, a);
                copy_or_transpose<true> (transB, b_fp8, b, n, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax);
                absmax_b = (float*)calculated_from_absmax+DESCALE_OFFSET;
                b_new = (Tb*)b_fp8;
            } else {
                assert(!transB);
            }
        } else {
            b_precision = CUDA_R_8F_E4M3;
            if (!std::is_same<Tb, __nv_fp8_e4m3>::value) {
                __nv_fp8_e4m3* b_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e4m3>(n*k);

                float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(b, k*n, a, SCALE_FORWARD_B);
                float *next_absmax = absmax_tracker.getNextAbsMaxPtr(b, k*n, a);
                copy_or_transpose<true> (transB, b_fp8, b, n, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax);
                absmax_b = (float*)calculated_from_absmax+DESCALE_OFFSET;
                b_new = (Tb*)b_fp8;
            } else {
                assert(!transB);
            }
        }
        transA = true;
        transB = false;
        int8_t fast_accum = backward ? 0 : 1;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

        if (absmax_a) {
            cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &absmax_a, sizeof(absmax_a)));
        }
        if (absmax_b) {
            cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &absmax_b, sizeof(absmax_b)));
        }
    } else {
        if (!std::is_same<Ta, floatX>::value || !std::is_same<Tb, floatX>::value || !std::is_same<Td, floatX>::value) {
            printf("Unsupported FP8 inputs for this matmul: %d x %d x %d\n", m, n, k);
            exit(EXIT_FAILURE);
        }
    }

    // handle direct FP8 output
    // todo - aux scale type - but we don't currently suppport forward GELU fusion with FP8 activations anyway
    if (std::is_same<Td, __nv_fp8_e4m3>::value) {
        d_precision = CUDA_R_8F_E4M3;

        float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(d, m*n, a, SCALE_FORWARD_B, false);
        if (!calculated_from_absmax) {
            recompute_due_to_d_absmax = true; // will need to do the matmul twice :(
            // unknown absmax - call it again, scale should have been initialised to 1.0f
            calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(d, m*n, a, SCALE_FORWARD_B, false);
        }
        float *absmax_d = calculated_from_absmax + SCALE_OFFSET;
        float *next_absmax_d = absmax_tracker.getNextAbsMaxPtr(d, m*n, a);
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &absmax_d, sizeof(absmax_d)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &next_absmax_d, sizeof(next_absmax_d)));
    }
    #endif

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, a_precision, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, a_precision, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, b_precision, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, b_precision, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, d_precision, m, n, m));

    // Strided Batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a_new, ALayout, b_new, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    #if FORCE_FP8_MATMUL == true
    if (recompute_due_to_d_absmax) {
        // FP8: redo the matmul with the correct scale factor
        absmax_tracker.updateSingleTensorAbsMax(d, m*n, a, 1.0f, stream);
        float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(d, m*n, a, SCALE_FORWARD_B, false);
        float *absmax_d = calculated_from_absmax + SCALE_OFFSET;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &absmax_d, sizeof(absmax_d)));
        cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                                   &alpha, a_new, ALayout, b_new, BLayout, &beta, d, CLayout, d, DLayout,
                                   &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));
    }
    #endif

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());

    // these will only do anything if they are currently allocated in our scratch allocator
    CudaScratchAllocator::releaseMemory(a_new);
    CudaScratchAllocator::releaseMemory(b_new);
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
template <typename Td, typename Ti, typename Tw>
void matmul_forward_cublaslt(Td* out,
                     Ti* inp, Tw* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     Td* pre_gelu=(Td*)NULL, int gelu_fusion=1,
                     void* inp_associated_tensor=(void*)NULL) { // hack - todo - for FP8 to find fch_gelu history


    float* inp_descale = NULL;
    if constexpr (std::is_same<Ti, __nv_fp8_e4m3>::value) {
        float *calculatedValues = absmax_tracker.getCalculatedValuesPtr(inp, B*T*C, inp_associated_tensor, SCALE_FORWARD_B, false);
        assert(calculatedValues);
        inp_descale = calculatedValues + DESCALE_OFFSET;
    }

    float* weight_descale = NULL;
    if constexpr (std::is_same<Tw, __nv_fp8_e4m3>::value) {
        float *calculatedValues = absmax_tracker.getCalculatedValuesPtr(weight, C*OC, NULL, SCALE_FP8_WEIGHTS, false);
        if (calculatedValues) {
            weight_descale = calculatedValues + DESCALE_OFFSET;
        }
    }

    if (pre_gelu) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false, weight_descale, inp_descale);

        if constexpr (std::is_same<Td, __nv_fp8_e4m3>::value) {
            float *pre_gelu_from_absmax = absmax_tracker.getCalculatedValuesPtr(pre_gelu, B*T*OC, weight, SCALE_FORWARD_B, false);
            float *pre_gelu_descale = pre_gelu_from_absmax + DESCALE_OFFSET;
            assert(pre_gelu_from_absmax);

            float *out_from_absmax = absmax_tracker.getCalculatedValuesPtr(out, B*T*OC, weight, SCALE_FORWARD_B, false);
            float *out_next_absmax = absmax_tracker.getNextAbsMaxPtr(out, B*T*OC, weight);

            if (!out_from_absmax) {
                // do it once just to get the correct absmax for the next step
                out_from_absmax = absmax_tracker.getCalculatedValuesPtr(out, B*T*OC, weight, SCALE_FORWARD_B, false);
                copy_advanced<false, gelu_forward_elementwise, false>(out, pre_gelu, B*T*OC, pre_gelu_descale, NULL, out_next_absmax, false, stream);
                absmax_tracker.updateSingleTensorAbsMax(out, B*T*OC, weight, 1.0f, stream);
            }

            float *out_scale = out_from_absmax + SCALE_OFFSET;
            copy_advanced<false, gelu_forward_elementwise, false>(out, pre_gelu, B*T*OC, pre_gelu_descale, out_scale, out_next_absmax, false, stream);
        } else {
            copy_advanced<false, gelu_forward_elementwise>(out, pre_gelu, B*T*OC, NULL, NULL, NULL, false, stream);
        }
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false, weight_descale, inp_descale);
    }

    /*
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false, weight_descale);
        copy_advanced<false, gelu_forward_elementwise>(out, pre_gelu, B*T*OC, NULL, NULL, false, stream);
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false, weight_descale);
    }
    */
}

template <typename Ti=floatX, typename Tw=floatX>
void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, Ti* inp, Tw* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     Ti* pre_gelu=(Ti*)NULL, int gelu_fusion=1,
                     void* inp_associated_tensor=(void*)NULL) { // hack - todo - for FP8 to find fch_gelu history)
    NVTX_RANGE_FN();

    #if FORCE_FP8_MATMUL == true
    bool transposed_inp = false;
    float* inp_descale = NULL;
    Ti *inp_new = inp;
    if constexpr (std::is_same<Ti, __nv_fp8_e4m3>::value) {
        float *calculatedValues = absmax_tracker.getCalculatedValuesPtr(inp, B*T*C, inp_associated_tensor, SCALE_FORWARD_B, false);
        assert(calculatedValues);
        inp_descale = calculatedValues + DESCALE_OFFSET;

        inp_new = CudaScratchAllocator::getMemory<Ti>(B*T*C);
        copy_or_transpose<false> (true, inp_new, inp, C, B*T, NULL, NULL, NULL);
        transposed_inp = true;
    }
    bool transposed_weight = false;
    float *weight_descale = NULL;
    Tw *weight_new = weight;
    if (std::is_same<Tw, __nv_fp8_e4m3>::value) {
        float *calculatedValues = absmax_tracker.getCalculatedValuesPtr(weight, C*OC, NULL, SCALE_FP8_WEIGHTS, false);
        assert(calculatedValues);
        weight_descale = calculatedValues + DESCALE_OFFSET;

        // Move descale factor
        float descale_cpu;
        cudaMemcpy(&descale_cpu, weight_descale, sizeof(float), cudaMemcpyDeviceToHost);

        weight_new = CudaScratchAllocator::getMemory<Tw>(C*OC);
        copy_or_transpose<false> (true, weight_new, weight, C, OC, NULL, NULL, NULL);
        transposed_weight = true;
    }
    #endif

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    #if FORCE_FP8_MATMUL == true
    bool allow_fp8 = (C < 50000) && (B*T < 50000) && (OC < 50000);
    if (allow_fp8) {
        // Get allocations for dout and dout transposed, then handle both at the same time
        __nv_fp8_e5m2 *dout_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e5m2>(B*T*OC);
        __nv_fp8_e5m2 *dout_transposed_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e5m2>(B*T*OC);

        float *calculated_from_absmax = absmax_tracker.getCalculatedValuesPtr(dout, B*T*OC, weight, SCALE_BACKWARDS_B);
        float *next_absmax = absmax_tracker.getNextAbsMaxPtr(dout, B*T*OC, weight);
        copy_and_transpose<true> (dout_transposed_fp8, dout_fp8, dout, OC, B*T, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax);

        // backward to input, uses = in the backward pass (set the gradient)
        matmul_cublaslt(dinp, weight_new, dout_fp8, NULL, C, B*T, OC, stream, transposed_weight, false, 0, 0, 0, 0, false,
                        gelu_fusion >= 2 ? pre_gelu : NULL, true, weight_descale, calculated_from_absmax+DESCALE_OFFSET);

        // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
        matmul_cublaslt(dweight, inp_new, dout_transposed_fp8, NULL /*dbias*/, C, OC, B*T, stream, transposed_inp, false, 0, 0, 0, 0,
                        true /* accumulate */, NULL, true, inp_descale, calculated_from_absmax+DESCALE_OFFSET);

        CudaScratchAllocator::releaseMemory(dout_transposed_fp8);
        CudaScratchAllocator::releaseMemory(dout_fp8);
    } else
    #endif
    {
        // backward to input, uses = in the backward pass (set the gradient)
        matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                        gelu_fusion >= 2 ? pre_gelu : NULL, true);

        // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
        matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                        true /* accumulate */, NULL, true);
    }

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        float* pre_gelu_descale = NULL;
        #if FORCE_FP8_MATMUL == true
        if constexpr (std::is_same<Ti, __nv_fp8_e4m3>::value) {
            float *calculatedValues = absmax_tracker.getCalculatedValuesPtr(pre_gelu, B*T*C, inp_associated_tensor, SCALE_FORWARD_B, false);
            assert(calculatedValues);
            pre_gelu_descale = calculatedValues + DESCALE_OFFSET;

            // Move descale factor
            float pre_gelu_descale_cpu;
            cudaMemcpy(&pre_gelu_descale_cpu, pre_gelu_descale, sizeof(float), cudaMemcpyDeviceToHost);

        }
        #endif
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream, pre_gelu_descale);
    }

    #if FORCE_FP8_MATMUL == true
    if (inp_new != NULL) {
        CudaScratchAllocator::releaseMemory(inp_new);
    }
    if (weight_new != NULL) {
        CudaScratchAllocator::releaseMemory(weight_new);
    }
    #endif
}
