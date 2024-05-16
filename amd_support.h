/*

Goal: unobtrusively provide support for AMD devices with minimal changes to the main CUDA code

Example (assuming ROCm 6.1.1 installed in /opt/rocm, or ROCM_PATH environment variable is set):

$ make train_gpt2amd
[...]
$ ./train_gpt2amd
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| input dataset prefix  | data/tiny_shakespeare                              |
| output log file       | NULL                                               |
| batch size B          | 4                                                  |
| sequence length T     | 1024                                               |
| learning rate         | 3.000000e-04                                       |
| max_steps             | -1                                                 |
| val_loss_every        | 20                                                 |
| val_max_batches       | 20                                                 |
| sample_every          | 20                                                 |
| genT                  | 64                                                 |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
+-----------------------+----------------------------------------------------+
| device                | Radeon RX 7900 XTX                                 |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
[...]
step    1/74: train loss 4.341256 (acc 4.341256) (173.484512 ms, 23610.175781 tok/s)
step    2/74: train loss 4.520747 (acc 4.520747) (84.588364 ms, 48422.726562 tok/s)
step    3/74: train loss 4.451268 (acc 4.451268) (82.094826 ms, 49176.976562 tok/s)
step    4/74: train loss 3.973935 (acc 3.973935) (81.191582 ms, 49622.757812 tok/s)
step    5/74: train loss 3.607177 (acc 3.607177) (80.539436 ms, 49955.472656 tok/s)
step    6/74: train loss 3.799689 (acc 3.799689) (79.265457 ms, 50335.402344 tok/s)
step    7/74: train loss 3.582036 (acc 3.582036) (79.510399 ms, 50558.101562 tok/s)
step    8/74: train loss 3.721121 (acc 3.721121) (81.067596 ms, 50552.734375 tok/s)
step    9/74: train loss 3.335243 (acc 3.335243) (79.859634 ms, 50662.250000 tok/s)
step   10/74: train loss 3.452644 (acc 3.452644) (80.037292 ms, 50731.734375 tok/s)
[...]
$ mpirun -np 4 ./train_gpt2amd -i data/TinyStories -b 16
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| input dataset prefix  | data/TinyStories                                   |
| output log file       | NULL                                               |
| batch size B          | 16                                                 |
| sequence length T     | 1024                                               |
| learning rate         | 3.000000e-04                                       |
| max_steps             | -1                                                 |
| val_loss_every        | 20                                                 |
| val_max_batches       | 20                                                 |
| sample_every          | 20                                                 |
| genT                  | 64                                                 |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
+-----------------------+----------------------------------------------------+
| device                | Radeon RX 7900 XTX                                 |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
[...]
step    1/14124: train loss 2.412066 (acc 2.374867) (407.426697 ms, 160853.468750 tok/s)
step    2/14124: train loss 3.324505 (acc 3.278165) (309.399994 ms, 211816.375000 tok/s)
step    3/14124: train loss 2.373110 (acc 2.390506) (311.514404 ms, 211079.062500 tok/s)
step    4/14124: train loss 2.189030 (acc 2.220984) (311.049500 ms, 210943.765625 tok/s)
step    5/14124: train loss 2.183907 (acc 2.187199) (310.421326 ms, 210991.140625 tok/s)
step    6/14124: train loss 2.131746 (acc 2.131070) (310.315399 ms, 211035.421875 tok/s)
step    7/14124: train loss 2.054895 (acc 2.078254) (311.606201 ms, 210899.796875 tok/s)
step    8/14124: train loss 2.019783 (acc 2.047217) (311.635284 ms, 210799.890625 tok/s)
step    9/14124: train loss 2.060214 (acc 2.049447) (310.942780 ms, 210794.750000 tok/s)
step   10/14124: train loss 1.991402 (acc 1.970346) (312.162140 ms, 210679.437500 tok/s)
[...]

*/

#pragma once

#ifdef MULTI_GPU
#include <mpi.h>
#include <rccl/rccl.h>
#endif

#include <hip/hip_bfloat16.h>

#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/ck.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// cublaslt does not have kernels for gfx11, so best alternative in terms of perf/effort seems to be composite_kernels
// somewhat janky to invoke with all of the templating, but works..
static inline void matmul_forward_gfx11(hip_bfloat16* out,
                   const hip_bfloat16* inp, const hip_bfloat16* weight, const hip_bfloat16* bias,
                   int B, int T, int C, int OC) {
    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementOp = ck::tensor_operation::element_wise::Add;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};
    auto cde_element_op = CDEElementOp{};

    if (bias == NULL) {
        auto device_op = ck::tensor_operation::device::DeviceGemmWmma_CShuffle <
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::ColumnMajor,
            ck::tensor_layout::gemm::RowMajor,
            ck::bhalf_t,
            ck::bhalf_t,
            ck::bhalf_t,
            float,
            ck::bhalf_t,
            AElementOp,
            BElementOp,
            CElementOp,
            GemmSpec,
            256,
            128,
            256,
            8,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8,
            1>{};
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(inp)),
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(weight)),
            reinterpret_cast<ck::bhalf_t*>(out),
            B*T,
            OC,
            C,
            C,
            C,
            OC,
            a_element_op,
            b_element_op,
            c_element_op);
        invoker.Run(argument);
    } else {
        auto device_op = ck::tensor_operation::device::DeviceGemmMultipleD_Wmma_CShuffle <
            ck::tensor_layout::gemm::RowMajor,
            ck::tensor_layout::gemm::ColumnMajor,
            ck::Tuple<ck::tensor_layout::gemm::RowMajor>,
            ck::tensor_layout::gemm::RowMajor,
            ck::bhalf_t,
            ck::bhalf_t,
            ck::Tuple<ck::bhalf_t>,
            ck::bhalf_t,
            float,
            ck::bhalf_t,
            AElementOp,
            BElementOp,
            CDEElementOp,
            GemmSpec,
            256,
            128,
            256,
            8,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8>{};
        auto invoker = device_op.MakeInvoker();
        auto argument = device_op.MakeArgument(
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(inp)),
            reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(weight)),
            std::array<const void*, 1>{reinterpret_cast<ck::bhalf_t*>(const_cast<hip_bfloat16 *>(bias))},
            reinterpret_cast<ck::bhalf_t*>(out),
            B*T,
            OC,
            C,
            C,
            C,
            std::array<ck::index_t, 1>{0},
            OC,
            a_element_op,
            b_element_op,
            cde_element_op);
        invoker.Run(argument);
    }
}

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>

// macros below handle mostly cublaslt stuff not handled by hipify (yet)
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulDescSetAttribute hipblasLtMatmulDescSetAttribute
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatmulAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define CUBLASLT_EPILOGUE_GELU_BIAS HIPBLASLT_EPILOGUE_GELU_BIAS
#define CUBLASLT_EPILOGUE_GELU HIPBLASLT_EPILOGUE_GELU
#define CUBLASLT_EPILOGUE_BIAS HIPBLASLT_EPILOGUE_BIAS
#define CUBLASLT_EPILOGUE_DEFAULT HIPBLASLT_EPILOGUE_DEFAULT
#define cublasLtEpilogue_t hipblasLtEpilogue_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatmul hipblasLtMatmul
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_EPILOGUE HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER HIPBLASLT_MATMUL_DESC_BIAS_POINTER
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatrixLayoutCreate hipblasLtMatrixLayoutCreate
#define cublasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define cublasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define hipblasSetMathMode(handle, mode) HIPBLAS_STATUS_SUCCESS
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define cublasMath_t hipblasMath_t
#define CUBLAS_TF32_TENSOR_OP_MATH HIPBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define hipFuncSetAttribute(x,y,z) 0
#define hipProfilerStart(x) hipSuccess
#define hipProfilerStop(x) hipSuccess
#define nvtxRangePush(x) {}
#define nvtxRangePop(x) {}

static __device__ __forceinline__ hip_bfloat16 __float2bfloat16_rn(float f) {
    return hip_bfloat16::round_to_bfloat16(f);
}

static __device__ __forceinline__ float __bfloat162float(hip_bfloat16 f) {
    return static_cast<float>(f);
}

template <typename T>
static __device__ __forceinline__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_xor(var, laneMask, width);
}

template <typename T>
static __device__ __forceinline__ T __shfl_down_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_down(var, laneMask, width);
}

// provide cache hints where possible
#define __stcs(ptr, val) patched_stcs(ptr, val)
#define __ldcs(ptr) patched_ldcs(ptr)
#define __stcg(ptr, val) {*(ptr) = val;}
static __device__ __forceinline__ void patched_stcs(float *addr, float val) {
    __builtin_nontemporal_store(val, addr);
}
static __device__ __forceinline__ void patched_stcs(hip_bfloat16 *addr, hip_bfloat16 val) {
    *addr = val;
}
static __device__ __forceinline__ void patched_stcs(int4 *addr, int4 val) {
    int *a = (int *)addr;
    __builtin_nontemporal_store(val.x, a);
    __builtin_nontemporal_store(val.y, a+1);
    __builtin_nontemporal_store(val.z, a+2);
    __builtin_nontemporal_store(val.w, a+3);
}
static __device__ __forceinline__ float patched_ldcs(const float *addr) {
    return __builtin_nontemporal_load(addr);
}
static __device__ __forceinline__ int4 patched_ldcs(const int4 *addr) {
    const int *a = (const int *) addr;
    return make_int4(__builtin_nontemporal_load(a),
        __builtin_nontemporal_load(a+1),
        __builtin_nontemporal_load(a+2),
        __builtin_nontemporal_load(a+3));
}
static __device__ __forceinline__ hip_bfloat16 patched_ldcs(const hip_bfloat16 *addr) {
    return *addr;
}

// emulate CG for old train_gpt2_fp32:
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    asm volatile ("ds_swizzle_b32 v1, %0 offset:swizzle(SWAP,16) \n"\
                  "s_waitcnt lgkmcnt(0) \n"\
                  "v_add_f32_e32 %0, %0, v1 \n"
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:4 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:2 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_add_f32_dpp %0, %0, %0 row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"
                  : "+v"(x) : : "v1");
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
    asm volatile ("s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:8 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:4 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:2 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "v_max_f32_dpp %0, %0, %0 row_ror:1 row_mask:0xf bank_mask:0xf bound_ctrl:1 \n"\
                  "ds_swizzle_b32 v1, %0 offset:swizzle(SWAP,16) \n"\
                  "s_waitcnt lgkmcnt(0) \n"\
                  "v_max_f32_e32 %0, %0, v1 \n"
                  : "+v"(x) : : "v1");
    return x;
}

namespace cooperative_groups {
template <typename T>
struct reduce_operator {
    static __device__ __forceinline__ T reduce(const T a, const T b) { return a+b; };
};

template <typename T>
struct plus : public reduce_operator<T> {
    static __device__ __forceinline__ T reduce(const T a, const T b) {
        return a + b;
    }
};

template <typename T>
struct greater : public reduce_operator<T> {
    static __device__ __forceinline__ T reduce(const T a, const T b) {
        return fmaxf(a, b);
    }
};

template <typename T>
static __device__ __forceinline__ float reduce(const thread_block_tile<32>& warp, float x, const plus<T>& op) {
    return warp_reduce_sum(x);
}

template <typename T>
static __device__ __forceinline__ float reduce(const thread_block_tile<32>& warp, float x, const greater<T>& op) {
    return warp_reduce_max(x);
}

template struct plus<float>;
template struct greater<float>;
}
