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

// ----------------------------------------------------------------------------
// CUDA kernels

// (in the description below, we assume "column major" layout like cuBLAS/FORTRAN but unlike C/C++)
// general reduction kernel for any minor axis, could be used for other things than bias backward!
// bias backward: OC columns and B*T rows ==> per-column sum reduction with OC outputs in dbias
// each block handles (blockIdx.x * x128::size) columns and (blockIdx.y) rows
// data layout is column major => contiguous for column X and X+1 (row_stride elements across rows)
// ==> 128B coalesced loads with BF16 require blockIdx.x >= 8 (64 columns per block)
// with few columns (OC), we want smaller blockIdx.x to get more blocks and better GPU utilisation
// (see comments in /dev/cuda/matmul_backward_bias.cu for even more information)
template <int block_dim_x=2, int block_dim_y=512, bool accumulate=true, typename OutFloat=floatX>
__global__ void column_reduction_kernel(OutFloat* output, const floatX* input,
                                        int num_rows, int num_columns, int row_stride) {
    assert(block_dim_x == blockDim.x && block_dim_y == blockDim.y); // check template parameters
    assert(num_columns == gridDim.x * block_dim_x * x128::size); // must match, no partial blocks
    constexpr int block_size = block_dim_x * block_dim_y;
    __shared__ float smem[block_size * x128::size];

    float column_sum[x128::size] = {0.0f}; // per-thread (partial column) FP32 accumulator
    int column_idx = (blockIdx.x * block_dim_x + threadIdx.x) * x128::size;
    int smem_idx = threadIdx.x + threadIdx.y * block_dim_x; // smem idx for this thread with k=0

    #pragma unroll 4
    for (int row = threadIdx.y; row < num_rows; row += block_dim_y) {
        x128 packed_dout = load128(input + column_idx + row * row_stride);
        for (int k = 0; k < x128::size; k++) {
            column_sum[k] += (float)packed_dout[k];
        }
    }
    // todo - currently don't use f128 for smem, so we stride by block_size to avoid bank conflicts
    for (int k = 0; k < x128::size; k++) {
        smem[smem_idx + k * block_size] = column_sum[k]; // write column partial sums to shared mem
    }

    // blockDim.y threads are all processing the same column, so we need to add up their sums
    // i.e. we calculate (blockDim.x * x128::size) final sums in parallel (one per column)
    // so with blockDim.x = 8, we avoid the parts of the reduction with only 1/2/4 active threads
    for (int stride = block_size/2; stride >= block_dim_x; stride /= 2) {
        __syncthreads();
        if (threadIdx.y * block_dim_x < stride) {
            for (int k = 0; k < x128::size; k++) {
                int smem_idx_k = smem_idx + k * block_size;
                smem[smem_idx_k] = smem[smem_idx_k] + smem[smem_idx_k + stride];
            }
        }
    } // no __syncthreads() needed because smem read below was written by the same thread

    if (threadIdx.y == 0) {
        // accumulate if necessary (e.g. gradient accumulation for multiple micro-batches per batch)
        // one output per column (e.g. 1 bias parameter gradient per OC)
        x128 output128 = accumulate ? load128(output + column_idx) : x128::zeros();
        for (int k = 0; k < x128::size; k++) {
            output128[k] = (OutFloat)((float)output128[k] + smem[threadIdx.x + k * block_size]);
        }
        store128(output + column_idx, output128);
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// Wrapper around cublasLtMatmul that is meant to support everything we need in llm.c
// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
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
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m));
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

// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward_cublaslt(floatX* out,
                     floatX* inp, floatX* weight, floatX* bias,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    // By default only fuse GELU for H100+ as cuBLAS seems to be inefficient for fused GELU on Ada/Ampere (?)
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
}

void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,
                     floatX* dout, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,
                     floatX* pre_gelu=NULL, int gelu_fusion=1) {
    NVTX_RANGE_FN();

    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // 1 block per SM and blockIdx.x=2 ==> need (2*2*x128::size) columns per SM ==> 16 at BF16
        // 768/16 ==> 48 SMs (out of 132 on H100) active for small bias kernels on 124M GPT2 models
        // 3072/16 ==> 192 which is good but 96 with blockIdx.x=4 is faster due to better coalescing
        // ===>
        // 1) Set block_size_x = 8. If we get less than 0.5 or 0.25 blocks per SM, reduce to 4 or 2.
        // 2) 1024-wide blocks unless block_size_x=8 with more than 2 blocks per SM, then use 512-wide
        int block_size_x = 8;
        int total_blocks = OC / (block_size_x * x128::size);
        int num_SMs = deviceProp.multiProcessorCount;

        if (total_blocks <= num_SMs / 4) { block_size_x = 2, total_blocks *= 4; }
        else if (total_blocks <= num_SMs / 2) { block_size_x = 4, total_blocks *= 2; }
        assert(OC == total_blocks * block_size_x * x128::size);

        int block_size = (total_blocks <= num_SMs * 2) ? 1024 : 512;
        //printf("block_size_x: %d, total_blocks: %d, block_size_512: %d\n", block_size_x, total_blocks, block_size_512);
        switch (block_size_x) {
            case 2: column_reduction_kernel<2, 512><<<total_blocks, dim3(2, 512)>>>(dbias, dout, B*T, OC, OC); break;
            case 4: column_reduction_kernel<4, 256><<<total_blocks, dim3(4, 256)>>>(dbias, dout, B*T, OC, OC); break;
            case 8: if (block_size == 1024) { column_reduction_kernel<8, 128><<<total_blocks, dim3(8, 128)>>>(dbias, dout, B*T, OC, OC); }
                    else { column_reduction_kernel<8, 64><<<total_blocks, dim3(8, 64)>>>(dbias, dout, B*T, OC, OC); }
                    break;
        }
        cudaCheck(cudaGetLastError());
        dbias = NULL; // prevent dbias calculation from also being fused in matmul_cublaslt below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
}
