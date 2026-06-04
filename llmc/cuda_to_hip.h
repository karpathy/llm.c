/*
CUDA-to-HIP compatibility shim for the AMD/ROCm build of llm.c.

This is the only file that knows about HIP. It is force-included on every
HIP translation unit by the Makefile (-include llmc/cuda_to_hip.h), so the
CUDA-spelled sources (cudaMalloc, cublasLt*, __nv_bfloat16, ...) compile
unchanged on ROCm. On NVIDIA this header is never included and the CUDA path
is byte-for-byte the original.

Layout:
  1. libc headers BEFORE <hip/hip_runtime.h> so host memcpy/memset win over
     HIP's __device__ overloads inside a .cu compiled as HIP.
  2. runtime / bf16 / fp16 / library headers.
  3. symbol aliases (runtime, bf16/fp16 types, cuBLAS(Lt), profiler, NVTX).
  4. warp-size and full-warp-mask abstractions for the fault classes.
  5. streaming load/store and cooperative-groups reduce shims HIP lacks.
*/
#ifndef LLMC_CUDA_TO_HIP_H
#define LLMC_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/library_types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// ----------------------------------------------------------------------------
// Runtime API
#define cudaError_t                          hipError_t
#define cudaSuccess                          hipSuccess
#define cudaGetErrorString                   hipGetErrorString
#define cudaGetLastError                     hipGetLastError
#define cudaErrorMemoryAllocation            hipErrorOutOfMemory

#define cudaMalloc                           hipMalloc
#define cudaFree                             hipFree
#define cudaMallocManaged                    hipMallocManaged
#define cudaMemAdvise                        hipMemAdvise
#define cudaMemAdviseSetPreferredLocation    hipMemAdviseSetPreferredLocation
#define cudaCpuDeviceId                      hipCpuDeviceId
#define cudaMallocHost                       hipHostMalloc
#define cudaFreeHost                         hipHostFree
#define cudaHostAllocWriteCombined           hipHostMallocWriteCombined

#define cudaMemcpy                           hipMemcpy
#define cudaMemcpyAsync                      hipMemcpyAsync
#define cudaMemsetAsync                      hipMemsetAsync
#define cudaMemset                           hipMemset
#define cudaMemcpyHostToDevice               hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost               hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice             hipMemcpyDeviceToDevice
#define cudaMemcpyHostToHost                 hipMemcpyHostToHost
#define cudaMemGetInfo                       hipMemGetInfo

#define cudaStream_t                         hipStream_t
#define cudaStreamCreate                     hipStreamCreate
#define cudaStreamDestroy                    hipStreamDestroy
#define cudaStreamSynchronize                hipStreamSynchronize
#define cudaStreamWaitEvent                  hipStreamWaitEvent
#define cudaStreamNonBlocking                hipStreamNonBlocking
#define cudaStreamCreateWithPriority         hipStreamCreateWithPriority

#define cudaEvent_t                          hipEvent_t
#define cudaEventCreate                      hipEventCreate
#define cudaEventDestroy                     hipEventDestroy
#define cudaEventRecord                      hipEventRecord
#define cudaEventSynchronize                 hipEventSynchronize
#define cudaEventElapsedTime                 hipEventElapsedTime

#define cudaDeviceProp                       hipDeviceProp_t
#define cudaGetDeviceProperties              hipGetDeviceProperties
#define cudaSetDevice                        hipSetDevice
#define cudaGetDevice                        hipGetDevice
#define cudaGetDeviceCount                   hipGetDeviceCount
#define cudaDeviceSynchronize                hipDeviceSynchronize
#define cudaDeviceGetAttribute               hipDeviceGetAttribute
#define cudaFuncSetAttribute                 hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// hipProfilerStart/Stop are deprecated and return hipErrorNotSupported unless a
// profiler is attached, which would trip the cudaCheck wrapper. These are
// profiling-only hooks with no effect on compute, so make them succeed no-ops.
#define cudaProfilerStart()                  hipSuccess
#define cudaProfilerStop()                   hipSuccess

// ----------------------------------------------------------------------------
// bf16 / fp16 types and intrinsics
#define __nv_bfloat16                        __hip_bfloat16
#define nv_bfloat16                          __hip_bfloat16
#define __nv_bfloat162                       __hip_bfloat162
#define nv_bfloat162                         __hip_bfloat162
#define __nv_bfloat16_raw                    __hip_bfloat16_raw
#define __nv_bfloat162_raw                   __hip_bfloat162_raw
// HIP's __float2bfloat16 already rounds to nearest; CUDA spells it _rn.
#define __float2bfloat16_rn                  __float2bfloat16

// ----------------------------------------------------------------------------
// cuBLAS / cuBLASLt -> hipBLAS / hipBLASLt
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

// scalar data types (cudaDataType / cudaDataType_t -> hipDataType from <hip/library_types.h>)
#define cudaDataType                         hipDataType
#define cudaDataType_t                       hipDataType
#define cublasDataType_t                     hipDataType
#define CUDA_R_32F                           HIP_R_32F
#define CUDA_R_16F                           HIP_R_16F
#define CUDA_R_16BF                          HIP_R_16BF

// handles / status / ops
#define cublasStatus_t                       hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS                HIPBLAS_STATUS_SUCCESS
#define cublasOperation_t                    hipblasOperation_t
#define CUBLAS_OP_N                          HIPBLAS_OP_N
#define CUBLAS_OP_T                          HIPBLAS_OP_T
#define cublasComputeType_t                  hipblasComputeType_t
#define CUBLAS_COMPUTE_32F                   HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_16F                   HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F_FAST_TF32         HIPBLAS_COMPUTE_32F_FAST_TF32

// plain cuBLAS (the FP32 driver uses cublasSgemm / cublasSgemmStridedBatched)
#define cublasHandle_t                       hipblasHandle_t
#define cublasCreate                         hipblasCreate
#define cublasDestroy                        hipblasDestroy
#define cublasSetStream                      hipblasSetStream
#define cublasSgemm                          hipblasSgemm
#define cublasSgemmStridedBatched            hipblasSgemmStridedBatched
#define cublasMath_t                         hipblasMath_t
#define cublasSetMathMode                    hipblasSetMathMode
#define CUBLAS_DEFAULT_MATH                  HIPBLAS_DEFAULT_MATH
#define CUBLAS_TF32_TENSOR_OP_MATH           HIPBLAS_TF32_TENSOR_OP_MATH

// cuBLASLt object types
#define cublasLtHandle_t                     hipblasLtHandle_t
#define cublasLtCreate                       hipblasLtCreate
#define cublasLtDestroy                      hipblasLtDestroy
#define cublasLtMatmul                       hipblasLtMatmul
#define cublasLtMatmulDesc_t                 hipblasLtMatmulDesc_t
#define cublasLtMatmulDescCreate             hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy            hipblasLtMatmulDescDestroy
#define cublasLtMatmulDescSetAttribute       hipblasLtMatmulDescSetAttribute
#define cublasLtMatrixLayout_t               hipblasLtMatrixLayout_t
#define cublasLtMatrixLayoutCreate           hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy          hipblasLtMatrixLayoutDestroy
#define cublasLtMatrixLayoutSetAttribute     hipblasLtMatrixLayoutSetAttribute
#define cublasLtMatmulPreference_t           hipblasLtMatmulPreference_t
#define cublasLtMatmulPreferenceCreate       hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy      hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulHeuristicResult_t      hipblasLtMatmulHeuristicResult_t
#define cublasLtMatmulAlgoGetHeuristic       hipblasLtMatmulAlgoGetHeuristic
#define cublasLtEpilogue_t                   hipblasLtEpilogue_t

// cuBLASLt attribute / epilogue enums
#define CUBLASLT_MATMUL_DESC_TRANSA              HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB              HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE            HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER        HIPBLASLT_MATMUL_DESC_BIAS_POINTER
#define CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE      HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER
#define CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD     HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT       HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#define CUBLASLT_EPILOGUE_DEFAULT                HIPBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_EPILOGUE_GELU_AUX               HIPBLASLT_EPILOGUE_GELU_AUX
#define CUBLASLT_EPILOGUE_GELU_AUX_BIAS          HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
#define CUBLASLT_EPILOGUE_DGELU                  HIPBLASLT_EPILOGUE_DGELU
#define CUBLASLT_EPILOGUE_BIAS                   HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_BGRADB                 HIPBLASLT_EPILOGUE_BGRADB

// ----------------------------------------------------------------------------
// NVTX profiling markers -> no-ops (roctx is optional and profiling-only)
#define nvtxRangePush(x)                     ((void)0)
#define nvtxRangePop()                       ((void)0)
#define nvtxNameCudaStreamA(stream, name)    ((void)0)

// ----------------------------------------------------------------------------
// Warp size and full-warp mask (fault classes: warp size 32-vs-64, lane masks)
//
// HIP has two compile passes per .cu: a host pass and a device pass. Arch
// macros like __GFX9__ are defined ONLY in the device pass, so they cannot
// drive a constant that the host launch code (dim3(WARP_SIZE, ...)) also reads
// -- host and device would then disagree and the launch geometry would be
// wrong. llm.c uses WARP_SIZE both as a host launch dimension AND as a device
// constant, so it must be one value per build. The Makefile derives
// LLMC_WARP_SIZE from the single target arch (gfx90a/gfx94x => 64, RDNA => 32)
// and defines it for both passes, keeping host and device identical.
#ifndef LLMC_WARP_SIZE
#error "LLMC_WARP_SIZE must be set by the build (derived from the target arch)"
#endif

// HIP's __shfl*_sync require a 64-bit mask regardless of the active wave width
// (ROCm static_asserts sizeof(MaskT)==8); the CUDA 0xFFFFFFFF literal will not
// compile. This is NOT keyed on wave width, only on the toolchain.
#define LLMC_FULL_WARP_MASK 0xffffffffffffffffULL

// ----------------------------------------------------------------------------
// Streaming cache-hint load/store: HIP lacks __stcs/__stcg and only provides
// __ldcs for __half/__half2 (<hip/hip_fp16.h>). The non-template half/half2
// __ldcs overloads win over this unconstrained template, which covers the
// remaining int4 / float / floatX / unsigned short call sites. The hints are
// advisory; a plain load/store is semantically identical.
template <class T> __device__ __forceinline__ T    __ldcs(const T* ptr)       { return *ptr; }
template <class T> __device__ __forceinline__ void __stcs(T* ptr, T value)    { *ptr = value; }
template <class T> __device__ __forceinline__ void __stcg(T* ptr, T value)    { *ptr = value; }

// ----------------------------------------------------------------------------
// Cooperative-groups reduce: HIP's CG has no cg::reduce / cg::plus / cg::greater
// (used by train_gpt2_fp32.cu over tiled_partition<32>). Provide a butterfly
// reduction over the tile; width = tile.size() stays within the 32-lane tile on
// a 64-lane wavefront, so it is correct on both wave32 and wave64.
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_version.h>
namespace cooperative_groups {
// ROCm 7.2.x's cooperative_groups had neither cg::plus/cg::greater nor cg::reduce.
// Newer ROCm (>= 7.13) ships cg::plus/cg::greater natively (redefining them is an
// error there) but STILL has no cg::reduce -- so version-guard only the functors,
// and always supply reduce. Butterfly reduction over the tile; width = tile.size()
// stays within the 32-lane tile on a 64-lane wavefront (correct on wave32/wave64).
#if HIP_VERSION < 71300000
template <class T> struct plus    { __device__ T operator()(T a, T b) const { return a + b; } };
template <class T> struct greater { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
#endif
template <class Group, class T, class Op>
__device__ __forceinline__ T reduce(const Group& g, T val, Op op) {
    for (int offset = g.size() / 2; offset > 0; offset >>= 1) {
        val = op(val, g.shfl_xor(val, offset));
    }
    return val;
}
} // namespace cooperative_groups

#else // NVIDIA: no-op include of the CUDA runtime; the CUDA path is unchanged.
#include <cuda_runtime.h>
#endif // USE_HIP

#endif // LLMC_CUDA_TO_HIP_H
