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
                     bool accumulate=false, void* pre_gelu=NULL, bool backward=false, float* absmax_a=NULL, float* absmax_b=NULL,
                     void* associated_with_a=NULL, float* absmax_pre_gelu=NULL)
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
    bool release_scratch_a = false;
    bool release_scratch_b = false;
    auto a_precision = CUBLAS_LOWP;
    auto b_precision = CUBLAS_LOWP;
    auto d_precision = CUBLAS_LOWP;
    Ta* a_converted = (Ta*)a;
    Tb* b_converted = (Tb*)b;

    #if FORCE_FP8_MATMUL
    bool recompute_due_to_d_absmax = false;

    // hack - todo - only skip embedding matmul (end of forward, start of backward)
    bool allow_fp8 = (m < 50000 && n < 50000 && k < 50000);

    if (batch_count == 0 && allow_fp8) {
        a_precision = CUDA_R_8F_E4M3;
        if constexpr (!std::is_same<Ta, __nv_fp8_e4m3>::value) {
            __nv_fp8_e4m3* a_fp8 = NULL;
            float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_a", a, k*m, associated_with_a ? associated_with_a : b, SCALE_A, true, true);
            absmax_a = (float*)calculated_from_absmax+DESCALE_OFFSET;

            if (!transA && backward && use_act_transpose_cache) {
                a_fp8 = g_transposed_cache.getTransposed<Ta, __nv_fp8_e4m3>(a, m, k, /* compute */ false, /* find_only*/ true);
            }
            if (a_fp8 == NULL) {
                release_scratch_a = true;
                a_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e4m3>(m*k);
                float *next_absmax = absmax_tracker.next_absmax_ptr(a, k*m, associated_with_a ? associated_with_a : b, 0.0f, true);
                copy_or_transpose<true> (!transA, a_fp8, a, m, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax, stream);
            }
            a_converted = (Ta*)a_fp8;
        } else {
            if (!transA) {
                if (use_weights_transpose_cache && associated_with_a == NULL) {
                    a_converted = g_transposed_cache.getTransposed(a, m, k, stream);
                } else {
                    a_converted = g_transposed_cache.getTransposed(a, m, k, false, true);
                    if (a_converted == NULL) {
                        a_converted = CudaScratchAllocator::getMemory<__nv_fp8_e4m3>(m*k);
                        copy_or_transpose<false> (true, a_converted, a, m, k, NULL, NULL, NULL, stream);
                        release_scratch_a = true;
                    }
                }
            }
        }

        if (backward) {
            b_precision = CUDA_R_8F_E5M2;
            if constexpr (!std::is_same<Tb, __nv_fp8_e5m2>::value) {
                __nv_fp8_e5m2* b_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e5m2>(n*k);

                float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_b_e5", b, k*n, a, SCALE_BACKWARDS_B, true, true);
                float *next_absmax = absmax_tracker.next_absmax_ptr(b, k*n, a, 0.0f, true);
                copy_or_transpose<true> (transB, b_fp8, b, n, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax, stream);
                absmax_b = (float*)calculated_from_absmax+DESCALE_OFFSET;
                b_converted = (Tb*)b_fp8;
                release_scratch_b = true;
            } else {
                assert(!transB);
            }
        } else {
            b_precision = CUDA_R_8F_E4M3;
            if constexpr (!std::is_same<Tb, __nv_fp8_e4m3>::value) {
                __nv_fp8_e4m3* b_fp8 = CudaScratchAllocator::getMemory<__nv_fp8_e4m3>(n*k);
                float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_b_e4", b, k*n, a, SCALE_FORWARD_B, true, true);
                float *next_absmax = absmax_tracker.next_absmax_ptr(b, k*n, a, 0.0f, true);
                absmax_b = (float*)calculated_from_absmax+DESCALE_OFFSET;
                if (use_act_transpose_cache) {
                    __nv_fp8_e4m3* transposed = g_transposed_cache.getTransposed<Tb, __nv_fp8_e4m3>(b, n, k, /* compute */ false);
                    copy_and_transpose<true> (transposed, b_fp8, b, n, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax, stream);
                } else {
                    copy_or_transpose<true> (transB, b_fp8, b, n, k, NULL, calculated_from_absmax+DESCALE_OFFSET, (unsigned int*)next_absmax, stream);
                }
                b_converted = (Tb*)b_fp8;
                release_scratch_b = true;
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
            printf("Unsupported FP8 inputs for this matmul: %d x %d x %d (sizeof: %lu %lu %lu)\n", m, n, k, sizeof(Ta), sizeof(Tb), sizeof(Td));
            exit(EXIT_FAILURE);
        }
    }

    // handle direct FP8 output
    // todo - aux scale type - but we don't currently suppport forward GELU fusion with FP8 activations anyway
    if (std::is_same<Td, __nv_fp8_e4m3>::value) {
        d_precision = CUDA_R_8F_E4M3;

        // calculate_if_needed must be false so it returns null and we know it was never seen before
        float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_d", d, m*n, a, SCALE_FORWARD_B, false);
        if (!calculated_from_absmax) {
            recompute_due_to_d_absmax = true; // will need to do the matmul twice :(
            // unknown absmax - call it again, scale should have been initialised to 1.0f
            calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_d_1st", d, m*n, a, SCALE_FORWARD_B, false, true);
        }
        float *absmax_d = calculated_from_absmax + SCALE_OFFSET;
        float *next_absmax_d = absmax_tracker.next_absmax_ptr(d, m*n, a);
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &absmax_d, sizeof(absmax_d)));
        //cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &next_absmax_d, sizeof(next_absmax_d)));
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
    // cuBLASLt requires C in FP8 mode to be BF16/FP16/FP32 (matching FP16/BF16 bias)... (sigh)
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
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias (if needed, disable GELU fusion)
            epilogue = CUBLASLT_EPILOGUE_DGELU;
            assert(absmax_pre_gelu || sizeof(Td) > 1); // we need absmax for FP8
            if (absmax_pre_gelu) {
                cublasDataType_t gelu_data_type = CUDA_R_8F_E4M3;
                cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &gelu_data_type, sizeof(gelu_data_type)));
                cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &absmax_pre_gelu, sizeof(absmax_pre_gelu)));
            }
        } else {
            #if FORCE_FP8_MATMUL
            if constexpr (std::is_same<Td, __nv_fp8_e4m3>::value) {
                epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
                float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_pre_gelu", (Td*)pre_gelu, m*n, a, SCALE_FORWARD_B, false);
                if (!calculated_from_absmax) {
                    assert(recompute_due_to_d_absmax);
                    calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_pre_gelu", (Td*)pre_gelu, m*n, a, SCALE_FORWARD_B, false, true);
                }
                if (!absmax_pre_gelu) {
                    absmax_pre_gelu = calculated_from_absmax + SCALE_OFFSET;
                }
                float *next_absmax_pre_gelu = absmax_tracker.next_absmax_ptr((Td*)pre_gelu, m*n, a);
                cublasDataType_t gelu_data_type = CUDA_R_8F_E4M3;
                cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &gelu_data_type, sizeof(gelu_data_type)));
                cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, &absmax_pre_gelu, sizeof(absmax_pre_gelu)));
                //cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER, &next_absmax_pre_gelu, sizeof(next_absmax_pre_gelu)));
            }
            #endif
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        cublasDataType_t bias_data_type = CUBLAS_LOWP; // note that cuBLASLt requires bias in FP8 mode to be BF16
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
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d, gelu: %d (precisions: %d %d %d) - accum: %d, backward: %d, batch_count: %d, bias_alignment: %d\n", m, n, k, has_bias, has_gelu, a_precision, b_precision, d_precision, accumulate, backward, batch_count, (int)((uintptr_t)bias % 16));
        exit(EXIT_FAILURE);
    } else {
        //printf("Found cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d, gelu: %d (precisions: %d %d %d)\n", m, n, k, has_bias, has_gelu, a_precision, b_precision, d_precision);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a_converted, ALayout, b_converted, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    #if FORCE_FP8_MATMUL == true
    if (recompute_due_to_d_absmax) {
        // FP8: now redo the entire matmul with the correct scaling factor
        absmax_tracker.update_single_absmax(d, m*n, a, 1.0f, stream);
        if (pre_gelu) {
            absmax_tracker.update_single_absmax((Td*)pre_gelu, m*n, a, 1.0f, stream);
        }
        cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                                   &alpha, a_converted, ALayout, b_converted, BLayout, &beta, d, CLayout, d, DLayout,
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
    if (release_scratch_a) { CudaScratchAllocator::releaseMemory(a_converted); }
    if (release_scratch_b) { CudaScratchAllocator::releaseMemory(b_converted); }
}

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
template <typename Td, typename Ti, typename Tw>
void matmul_forward_cublaslt(Td* out,
                     Ti* inp, Tw* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     Td* pre_gelu=(Td*)NULL, int gelu_fusion=1,
                     void* inp_associated_tensor=(void*)NULL) { // hack - todo - for FP8 to find fch_gelu history

    float* inp_descale = absmax_tracker.get_descale_ptr(inp, B*T*C, inp_associated_tensor);
    float* weight_descale = NULL;
    if constexpr (std::is_same<Tw, __nv_fp8_e4m3>::value) {
        // todo - this must be allowed to return null (for now) in case the weights were never scaled before
        float *calculatedValues = absmax_tracker.get_absmax_data("matmul_forward_weight", weight, C*OC, NULL, SCALE_FP8_WEIGHTS, false, true);
        if (calculatedValues) {
            weight_descale = calculatedValues + DESCALE_OFFSET;
        }
    }

    // todo: add back gelu fusion?
    if (pre_gelu && gelu_fusion < 1) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false, weight_descale, inp_descale);

        if constexpr (std::is_same<Td, __nv_fp8_e4m3>::value) {
            float *pre_gelu_descale = absmax_tracker.get_descale_ptr(pre_gelu, B*T*OC, weight);
            float *out_from_absmax = absmax_tracker.get_absmax_data("matmul_forward_out", out, B*T*OC, weight, SCALE_FORWARD_B, false);
            float *out_next_absmax = absmax_tracker.next_absmax_ptr(out, B*T*OC, weight);

            if (!out_from_absmax) {
                // do it once just to get the correct absmax for the next step
                out_from_absmax = absmax_tracker.get_absmax_data("matmul_forward_out_1st", out, B*T*OC, weight, SCALE_FORWARD_B, false, true);
                copy_advanced<false, gelu_forward_elementwise, false>(out, pre_gelu, B*T*OC, pre_gelu_descale, NULL, out_next_absmax, stream);
                absmax_tracker.update_single_absmax(out, B*T*OC, weight, 1.0f, stream);
            }

            float *out_scale = out_from_absmax + SCALE_OFFSET;

            if (use_act_transpose_cache) {
                __nv_fp8_e4m3* transposed = g_transposed_cache.getTransposed<__nv_fp8_e4m3, __nv_fp8_e4m3>(out, OC, B*T, /* compute */ false);
                copy_and_transpose<true, gelu_forward_elementwise, false> (transposed, out, pre_gelu, OC, B*T, pre_gelu_descale, out_scale, (unsigned int*)out_next_absmax, stream);
            } else {
                copy_advanced<false, gelu_forward_elementwise, false>(out, pre_gelu, B*T*OC, pre_gelu_descale, out_scale, out_next_absmax, stream);
            }
        } else {
            copy_advanced<false, gelu_forward_elementwise>(out, pre_gelu, B*T*OC, NULL, NULL, NULL, stream);
        }
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false, weight_descale, inp_descale);
    }
}

template <typename Ti=floatX, typename Tw=floatX>
void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, Ti* inp, Tw* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     Ti* pre_gelu=(Ti*)NULL, int gelu_fusion=1,
                     void* inp_associated_tensor=(void*)NULL) { // hack - todo - for FP8 to find fch_gelu history)
    NVTX_RANGE_FN();

    inp_associated_tensor = inp_associated_tensor ? inp_associated_tensor : inp;

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

        float *calculated_from_absmax = absmax_tracker.get_absmax_data("matmul_dout_e5", dout, B*T*OC, weight, SCALE_BACKWARDS_B, true, true);
        float *next_absmax = absmax_tracker.next_absmax_ptr(dout, B*T*OC, weight, 0.0f, true);
        float *dout_descale_ptr = calculated_from_absmax + DESCALE_OFFSET;
        copy_and_transpose<true> (dout_transposed_fp8, dout_fp8, dout, OC, B*T, NULL, dout_descale_ptr, (unsigned int*)next_absmax);

        float* inp_descale = absmax_tracker.get_descale_ptr(inp, B*T*C, inp_associated_tensor);
        float* weight_descale = absmax_tracker.get_descale_ptr(weight, C*OC, NULL, false, false, true);
        float* pre_gelu_descale = pre_gelu ? absmax_tracker.get_descale_ptr(pre_gelu, B*T*C, inp_associated_tensor) : NULL;

        // backward to input, uses = in the backward pass (set the gradient)
        matmul_cublaslt(dinp, weight, dout_fp8, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                        gelu_fusion >= 2 ? pre_gelu : NULL, true, weight_descale, dout_descale_ptr, NULL, pre_gelu_descale);

        // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
        matmul_cublaslt(dweight, inp, dout_transposed_fp8, NULL /*dbias*/, C, OC, B*T, stream, false, false, 0, 0, 0, 0,
                        true /* accumulate */, NULL, true, inp_descale, dout_descale_ptr, inp_associated_tensor);

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
        float* pre_gelu_descale = absmax_tracker.get_descale_ptr(pre_gelu, B*T*C, inp_associated_tensor);
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream, pre_gelu_descale);
    }
}
