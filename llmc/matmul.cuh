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

// todo - does this need to be included globally?
#include "copy_and_fp8.h"

// ----------------------------------------------------------------------------
// CUDA kernels

template<typename Tdout, typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(TensorGPU<OutFloat> dbias, TensorGPU<Tdout> dout, int BT, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    // todo - this kernel is way more complicated than it needs to be
    // (should look at my old PR to simplify it again after this)
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * Packed128<Tdout>::size;  // 64 at BF16

    int local_oc = warp_c * Packed128<Tdout>::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[Packed128<Tdout>::size];
    for (int k = 0; k < Packed128<Tdout>::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < BT; idx += gridDim.y * bt_per_block) {
            auto dout128 = load_tensor128(dout, global_oc + idx*OC);
            for (int k = 0; k < Packed128<Tdout>::size; k++) {
                accumulators[k] += dout128.get(k);
            }
        }
    }

    __shared__ float sub_results[Packed128<Tdout>::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < Packed128<Tdout>::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < Packed128<Tdout>::size; k += blockDim.z) {
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
void matmul_cublaslt(tensorX d, const tensorX a, const tensorX b, const tensorX bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, tensorX pre_gelu=null_tensorX, bool backward=false)
{
    NVTX_RANGE_FN();
    bool has_bias = (bias.data_ptr != NULL);
    bool has_gelu = (pre_gelu.data_ptr != NULL);

    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a.data_ptr % 16) != 0 || ((uintptr_t)b.data_ptr % 16) != 0 || ((uintptr_t)d.data_ptr % 16) != 0 || ((uintptr_t)bias.data_ptr % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CUBLAS_LOWP, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

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
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu.data_ptr, sizeof(pre_gelu.data_ptr)));
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
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias.data_ptr, sizeof(bias.data_ptr)));
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
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    #ifdef FAKE_FP8
    update_absmax(d, false); // fake FP8 requires the absmax to work
    #endif

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

#ifdef ENABLE_FP8
template<typename Td=float8, typename Ta=float8, typename Tb=float8>
void matmul_cublaslt_fp8(TensorGPU<Td> d, const TensorGPU<Ta> a, const TensorGPU<Tb> b, const tensorX bias,
                         int m, int n, int k, cudaStream_t stream=main_stream,
                         bool accumulate=false, bool backward=false)
{
    NVTX_RANGE_FN();
    if(((uintptr_t)a.data_ptr % 16) != 0 || ((uintptr_t)b.data_ptr % 16) != 0 || ((uintptr_t)d.data_ptr % 16) != 0 || ((uintptr_t)bias.data_ptr % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    cublasOperation_t opTranspose = CUBLAS_OP_T, opNoTranspose = CUBLAS_OP_N;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DLayout;
    cublasDataType_t typeA = std::is_same<Ta, float8>::value ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
    cublasDataType_t typeB = std::is_same<Tb, float8>::value ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
    cublasDataType_t typeD = std::is_same<Td, floatX>::value ? CUBLAS_LOWP :
                            (std::is_same<Td, float8>::value ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2);

    cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, typeA, k, m, k)); // always transposed for FP8
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, typeB, k, n, k)); // never transposed for FP8
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, CUBLAS_LOWP, m, n, m)); // must be BF16 for accumulation in cuBLASLt
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, typeD, m, n, m));

    // setup epilogue and associated pointers for bias
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if(bias.data_ptr != NULL) {
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
        cublasDataType_t bias_data_type = CUBLAS_LOWP; // BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias.data_ptr, sizeof(bias.data_ptr)));
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // FP8 scale factors and absmax pointers
    float* a_descale_ptr = a.scale_descale_ptr + 1;
    float* b_descale_ptr = b.scale_descale_ptr + 1;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_descale_ptr, sizeof(float*)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_descale_ptr, sizeof(float*)));
    if (sizeof(Td) == 1) {
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d.scale_descale_ptr, sizeof(float*)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, &d.absmax_ptr, sizeof(float*)));
    }

    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // create a preference handle with specified max workspace
    cublasLtMatmulPreference_t preference;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);

    if (returnedResults == 0) {
        printf("No cuBLASLt FP8 algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, (bias.data_ptr != NULL));
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}
#endif

template<typename Tout=float8, typename Tin=float8>
// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward(TensorGPU<Tout> out,
                     TensorGPU<Tin> inp, TensorGPU<Tin> weight, tensorX bias, int BT, int C, int OC,
                     TensorGPU<Tout> pre_gelu=TensorGPU<Tout>(), int gelu_fusion=1, cudaStream_t stream=main_stream) {
    if constexpr (sizeof(Tin) == 1) {
        matmul_cublaslt_fp8(pre_gelu.enabled() ? pre_gelu : out, weight, inp, bias, OC, BT, C, stream, false, false);
        if (pre_gelu.enabled()) {
            gelu_forward(out, pre_gelu, stream);
        }
    } else {
        if (pre_gelu.enabled() && gelu_fusion < 1) {
            matmul_cublaslt(pre_gelu, weight, inp, bias, OC, BT, C, stream, true, false, 0, 0, 0, 0, false, null_tensorX, false);
            gelu_forward(out, pre_gelu, stream);
        } else {
            matmul_cublaslt(out, weight, inp, bias, OC, BT, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
        }
    }
}

template<typename Tdout=floatX>
void matmul_backward_bias(tensorX dbias, TensorGPU<Tdout> dout, tensorFP32 scratch, int BT, int OC, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
    if (dbias != null_tensorX) {
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
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, BT, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(scratch, dout, BT, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, scratch, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
    }
}

template<typename Tdout=grads8>
void matmul_backward_fp8(tensorFP8e5 dinp, tensorX dweight, tensorX dbias,
                     TensorGPU<Tdout> dout, tensorFP8e4 inp, tensorFP8e4 weight,
                     tensorFP32 scratch1_big, tensorFP32 scratch2_huge,
                     int BT, int C, int OC,
                     tensorFP8e4 pre_gelu_activation=tensorFP8e4(), cudaStream_t stream=main_stream) {
#ifndef ENABLE_FP8
    // FP8 is not enabled so we use the regular floatX matmul path
    matmul_backward(dinp, dweight, dbias, dout, inp, weight, scratch1_big, BT, C, OC, pre_gelu_activation, 1, stream);
#else
    NVTX_RANGE_FN();
    matmul_backward_bias(dbias, dout, scratch1_big, BT, OC, stream);

    // N.B.: Both scratch1 and scratch2 are guaranteed to be big enough for 4BTC and 4CC in FP8
    // IMPORTANT: inp is allowed to be the same buffer as scratch2_huge (e.g. for fch_gelu)
    // ==> this MUST be done first and write to scratch1_big!
    // transpose input
    TensorGPU<float8> inp_fp8_transposed = inp;
    inp_fp8_transposed.data_ptr = (float8*)scratch1_big.data_ptr;
    transpose_simple<float8>(inp_fp8_transposed, inp, C, BT, stream);

    // convert dout to FP8e5 if it is not already, and transpose it
    // the buffer is guaranteed to be at least twice as big as 4BTC, so we can split it in 2
    // todo - merge conversion and tranposition like we did before?
    TensorGPU<grads8> dout_fp8 = *(TensorGPU<grads8>*)&dout;
    if constexpr (std::is_same<Tdout, grads8>::value == false) {
        dout_fp8.data_ptr = (grads8*)(scratch2_huge.data_ptr);
        copy_advanced(dout_fp8, dout, stream);
    }
    TensorGPU<grads8> dout_fp8_transposed = dout_fp8;
    dout_fp8_transposed.data_ptr = (grads8*)(scratch2_huge.data_ptr + (scratch2_huge.num_elements / 2));
    transpose_simple(dout_fp8_transposed, dout_fp8, OC, BT, stream);

    // GEMM 1: dweight, inp_fp8_transposed, dout_fp8_transposed
    matmul_cublaslt_fp8(dweight, inp_fp8_transposed, dout_fp8_transposed, null_tensorX, C, OC, BT, stream, false, true);

    // transpose weight (todo: option to cache this / do it at optimizer time)
    TensorGPU<float8> weight_fp8_transposed = weight;
    weight_fp8_transposed.data_ptr = (float8*)scratch1_big.data_ptr;
    transpose_simple(weight_fp8_transposed, weight, C, OC, stream);

    matmul_cublaslt_fp8(dinp, weight_fp8_transposed, dout_fp8, null_tensorX, C, BT, OC, stream, false, true);

    if (pre_gelu_activation.enabled()) {
        gelu_backward(dinp, dinp, pre_gelu_activation, stream);
    }
#endif
}


void matmul_backward(tensorX dinp, tensorX dweight, tensorX dbias,
                     tensorX dout, tensorX inp, tensorX weight,
                     tensorFP32 dbias_scratch,
                     int BT, int C, int OC,
                     tensorX pre_gelu_activation=null_tensorX, int gelu_fusion=1, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    matmul_backward_bias(dbias, dout, dbias_scratch, BT, OC, stream);

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, null_tensorX, C, BT, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu_activation : null_tensorX, true);

    // backward GELU (if it wasn't fused into the matmul above)
    if ( pre_gelu_activation.enabled() && gelu_fusion < 2) {
        gelu_backward(dinp, dinp, pre_gelu_activation, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout, null_tensorX /*dbias*/, C, OC, BT, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, null_tensorX, true);
}
