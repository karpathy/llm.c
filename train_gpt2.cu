/*
GPT-2 Transformer Neural Net trained in raw CUDA
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define ENABLE_ACTIVATION_COMPRESSION
//#define MASK_ONE_BYTE_COMPRESSION
//#define MASK_TWO_BYTES_COMPRESSION
//#define MASK_THREE_BYTES_COMPRESSION
//#define MASK_ALL_BYTES_COMPRESSION
size_t activation_size = 0; // We need to keep track of the activation size for freeing the memory *sigh*

// ----------------------------------------------------------------------------
// CUDA utils & global variables
// TODO A lot of global variables sadly required, could refactor into struct?

// TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
#define ENABLE_TF32
#ifdef ENABLE_TF32
static const cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
static const cublasMath_t cublas_math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
#else
static const cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
static const cublasMath_t cublas_math_mode = CUBLAS_DEFAULT_MATH;
#endif

// CUDA stream & graph (stream is global as it needs to be used for all kernels)
static cudaStream_t stream = 0; // graph generation cannot use default stream sadly
static cudaGraph_t graph;
static cudaGraphExec_t instance;
static bool graph_created = false;
static int graph_created_B = 0, graph_created_T = 0;

// cuBLAS & cuBLASLt handles + workspace
static cublasLtHandle_t cublaslt_handle = NULL;
static cublasHandle_t cublas_handle = NULL;
static size_t cublaslt_workspace_size = 64 * 1024 * 1024; // TODO what's a good size?
static void* cublaslt_workspace = NULL;

// CUDA & cuBLAS error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
#define cudaCheckErrors() (cudaCheck(cudaPeekAtLastError())) // peek = graph friendly
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__device__ float mask_lsb_byte(float in) {
#if defined(MASK_ALL_BYTES_COMPRESSION)
    return 0.0f;
#elif defined(MASK_THREE_BYTES_COMPRESSION)
    return __uint_as_float(__float_as_uint(in) & 0xFF000000);
#elif defined(MASK_TWO_BYTES_COMPRESSION)
    return __uint_as_float(__float_as_uint(in) & 0xFFFF0000);
#elif defined(MASK_ONE_BYTE_COMPRESSION)
    return __uint_as_float(__float_as_uint(in) & 0xFFFFFF00);
#else
    return in;
#endif
}

cudaError_t setProp(CUmemAllocationProp *prop, bool UseCompressibleMemory)
{
    CUdevice currentDevice;
    if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = currentDevice;

    if (UseCompressibleMemory)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

    return cudaSuccess;
}

cudaError_t allocateCompressible(void **adr, size_t size, bool UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;

    CUdeviceptr dptr;
    if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemGenericAllocationHandle allocationHandle;
    if (cuMemCreate(&allocationHandle, size, &prop, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    // Check if cuMemCreate was able to allocate compressible memory.
    if (UseCompressibleMemory) {
        CUmemAllocationProp allocationProp = {};
        cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);
        if (allocationProp.allocFlags.compressionType != CU_MEM_ALLOCATION_COMP_GENERIC) {
            printf("Could not allocate compressible memory... so waiving execution\n");
            exit(EXIT_FAILURE);
        }
    }

    if (cuMemMap(dptr, size, 0, allocationHandle, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    if (cuMemRelease(allocationHandle) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    *adr = (void *)dptr;
    return cudaSuccess;
}

cudaError_t freeCompressible(void *ptr, size_t size, bool UseCompressibleMemory)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop, UseCompressibleMemory);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == NULL)
        return cudaSuccess;
    if (cuMemUnmap((CUdeviceptr)ptr, size) != CUDA_SUCCESS ||
        cuMemAddressFree((CUdeviceptr)ptr, size) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}

// ----------------------------------------------------------------------------
// all the kernels

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    #pragma unroll // unrolled for higher performance
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for summing values
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void encoder_forward_kernel2(float* out,
                               int* inp, float* wte, float* wpe,
                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;

    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        float* out_btc = out + b * T * C + t * C + c;
        float* wte_ix = wte + ix * C + c;
        float* wpe_tc = wpe + t * C + c;
        *out_btc = mask_lsb_byte(*wte_ix + *wpe_tc);
    }
}

__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if(warp.thread_rank() == 0 && mean != nullptr) {
        //uint mean_bits = ;
        __stcs(mean + idx, mask_lsb_byte(m));
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, mask_lsb_byte(s));
    }

    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, mask_lsb_byte(n * weight[c] + bias[c]));
    }
}


__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = mask_lsb_byte(inp[inp_idx]);
        k[idx] = mask_lsb_byte(inp[inp_idx + NH * d]);
        v[idx] = mask_lsb_byte(inp[inp_idx + 2 * (NH * d)]);
    }
}

__global__ void unpermute_kernel(float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = mask_lsb_byte(inp[idx]);
    }
}

__device__ bool scale_is_autoregressive(int idx, int NH, int T) {
    int rest = idx % (NH * T * T);
    rest = rest % (T * T);
    int t2 = rest / T;
    int t = rest % T;
    return t > t2;
}

__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        if (scale_is_autoregressive(idx, NH, T)) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

__global__ void softmax_forward_kernel5_scale(float* out, float* inp, float scale, int B, int NH, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but assumes blocks of 512 threads (correct but slower with other sizes)
    // supports fused scaling & autoregressive masking with no additional writes to global memory

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    // this is now hardcoded to 16 (=512 threads) so some loops below can be fully unrolled
    int warpsPerBlock = 16; // blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // sanity checks to keep this function fully generic when called with an incorrect block size
    // do we actually need this - probably overkill?
    if (warpId > warpsPerBlock) {
        return;
    }
    if ((blockDim.x/32) < warpsPerBlock) {
        for (int i = warpId; i < warpsPerBlock; i += blockDim.x/32) {
            maxvals[i] = -INFINITY;
            sumvals[i] = 0.0f;
        }
    }
    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    if (scale != 1.0f) {
        for (int i = tid; i < C; i += blockDim.x) {
            float value;
            if (scale_is_autoregressive(idx * C + i, NH, C)) {
                value = -INFINITY;
            } else {
                value = x[i] * scale;
            }
            maxval = fmaxf(maxval, value);
        }
    } else {
        for (int i = tid; i < C; i += blockDim.x) {
            maxval = fmaxf(maxval, x[i]);
        }
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < 16; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum (to avoid rereading x[i] again)
    float sumval = 0.0f;
    if (scale != 1.0f) {
        for (int i = tid; i < C; i += blockDim.x) {
            float value;
            if (scale_is_autoregressive(idx * C + i, NH, C)) {
                value = -INFINITY;
            } else {
                // __ldcs to reduce cache persistence (not read again)
                value = __ldcs(&x[i]) * scale;
            }
            float output = expf(value - offset);
            out[idx * C + i] = mask_lsb_byte(output);
            sumval += output; // combined into the same loop unlike kernel3
        }
    } else {
        for (int i = tid; i < C; i += blockDim.x) {
            float output = expf(__ldcs(&x[i]) - offset);
            out[idx * C + i] = output;
            sumval += output;
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < 16; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = mask_lsb_byte(out[idx * C + i] / sum);
    }
}

__global__ void softmax_forward_kernel6_largeC(float* out, float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Cs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(C-1, idx)
    // so we just do some unnecessary reads (obviously bad for small C)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing
    const int UNROLL_FACTOR = 8;
    const int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];
    
    if (tid >= C) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    float* x = inp + idx * C; // input
    float* y = out + idx * C; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            maxval = fmaxf(maxval, x[min(C - 1, i + u*blockDim.x)]);
        }
    }

    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = __ldcs(&x[min(C - 1, i + u*blockDim.x)]);
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                float output = expf(reg_array[u] - offset);
                y[min(C - 1, i + u*blockDim.x)] = output; // compiler likes redundant min()?!
                sumval += output; // combined into the same loop unlike kernel3
            }
        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        #pragma unroll
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            reg_array[u] = y[min(C - 1, i + u*blockDim.x)];
        }
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            if (i + u*blockDim.x < C) {
                y[i + u*blockDim.x] = reg_array[u] / sum;
            }
        }
    }
}


__global__ void residual_forward_kernel(float* out, float* inp1, float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // __ldcs to reduce cache persistence (not read again after this)
        out[idx] = mask_lsb_byte(__ldcs(&inp1[idx]) + __ldcs(&inp2[idx]));
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

__global__ void crossentropy_forward_kernel1(float* losses,
                            float* probs, int* targets,
                            int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = mask_lsb_byte(-logf(probs_bt[ix]));
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(float* out,
                     int* inp, float* wte, float* wpe,
                     int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheckErrors();
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheckErrors();
}

void matmul_forward(float* out, float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC, int has_bias, int has_gelu) {
    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;
    }

    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc,
        CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, 
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // Define matrix layouts
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // Create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // Find a suitable algorithm
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));

    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, stream));

    // Cleanup
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    const int block_size = 512;
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    // TODO investigate cuBLASLt options for this
    cublasStatus_t stat;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    stat = cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed\n");
        exit(1);
    }
    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    // + fused elementwise scale and autoregressive masking
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 512;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel5_scale<<<grid_size, softmax_block_size, shared_mem_size, stream>>>(att, preatt, scale, B, NH, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    stat = cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed\n");
        exit(1);
    }

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
}

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2, N);
    cudaCheckErrors();
}


void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, N);
    cudaCheckErrors();
}

void softmax_forward(float* out, float* inp, int N, int C) {
    const int block_size = 512;
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    if (C >= 4096) { // TODO find optimal heuristic
        softmax_forward_kernel6_largeC<<<grid_size, block_size, shared_mem_size, stream>>>(out, inp, N, C);
    } else {
        softmax_forward_kernel5_scale<<<grid_size, block_size, shared_mem_size, stream>>>(out, inp, 1.0f, 0, 0, C);
    }
}

void crossentropy_forward(float* losses,
                            float* probs, int* targets,
                            int B, int T, int V) {
    const int block_size = 128;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size, 0, stream>>>(losses, probs, targets, B, T, V);
    cudaCheckErrors();
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;


// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, int on_device) {
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    float* params_memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&params_memory, num_parameters * sizeof(float)));
        cudaCheckErrors();
    } else {
        params_memory = (float*)malloc(num_parameters * sizeof(float));
    }
    // assign all the tensors their place in the array
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }

    // Get size of nv_bfloat16
    size_t nv_bfloat16_size = sizeof(nv_bfloat16);
    // printf it
    printf("nv_bfloat16 size: %zu\n", nv_bfloat16_size);

    return params_memory;
}


#define NUM_ACTIVATION_TENSORS 25
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float* qkvr; // (L, B, T, 3*C)
    float* v_accum; // (L, B, T, C)
} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory;

#if defined(ENABLE_ACTIVATION_COMPRESSION)
    activation_size = num_activations * sizeof(float);
    allocateCompressible((void**)&acts_memory, activation_size, true);
#else
    cudaCheck(cudaMalloc((void**)&acts_memory, num_activations * sizeof(float)));
#endif

    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses,
        &acts->qkvr, &acts->v_accum
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    int num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    int num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;


void gpt2_build_from_checkpoint(GPT2 *model, char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) { printf("Error opening model file\n"); exit(1); }
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file"); exit(1); }
    if (model_header[1] != 1) { printf("Bad version in model file"); exit(1); }

    // read in hyperparameters
    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    printf("[GPT-2]\n");
    printf("max_seq_len: %d\n", maxT);
    printf("vocab_size: %d\n", V);
    printf("num_layers: %d\n", L);
    printf("num_heads: %d\n", NH);
    printf("channels: %d\n", C);

    // allocate space for all the parameters and read them in
    model->param_sizes[0] = V * C; // wte
    model->param_sizes[1] = maxT * C; // wpe
    model->param_sizes[2] = L * C; // ln1w
    model->param_sizes[3] = L * C; // ln1b
    model->param_sizes[4] = L * (3 * C) * C; // qkvw
    model->param_sizes[5] = L * (3 * C); // qkvb
    model->param_sizes[6] = L * C * C; // attprojw
    model->param_sizes[7] = L * C; // attprojb
    model->param_sizes[8] = L * C; // ln2w
    model->param_sizes[9] = L * C; // ln2b
    model->param_sizes[10] = L * (4 * C) * C; // fcw
    model->param_sizes[11] = L * (4 * C); // fcb
    model->param_sizes[12] = L * C * (4 * C); // fcprojw
    model->param_sizes[13] = L * C; // fcprojb
    model->param_sizes[14] = C; // lnfw
    model->param_sizes[15] = C; // lnfb

    // cound the number of paramaters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)malloc(num_parameters * sizeof(float));
    fread(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fclose(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C; // encoded
        model->act_sizes[1] = L * B * T * C; // ln1
        model->act_sizes[2] = L * B * T; // ln1_mean
        model->act_sizes[3] = L * B * T; // ln1_rstd
        model->act_sizes[4] = L * B * T * 3*C; // qkv
        model->act_sizes[5] = L * B * T * C; // atty
        model->act_sizes[6] = L * B * NH * T * T; // preatt
        model->act_sizes[7] = L * B * NH * T * T; // att
        model->act_sizes[8] = L * B * T * C; // attproj
        model->act_sizes[9] = L * B * T * C; // residual2
        model->act_sizes[10] = L * B * T * C; // ln2
        model->act_sizes[11] = L * B * T; // ln2_mean
        model->act_sizes[12] = L * B * T; // ln2_rstd
        model->act_sizes[13] = L * B * T * 4*C; // fch
        model->act_sizes[14] = L * B * T * 4*C; // fch_gelu
        model->act_sizes[15] = L * B * T * C; // fcproj
        model->act_sizes[16] = L * B * T * C; // residual3
        model->act_sizes[17] = B * T * C; // lnf
        model->act_sizes[18] = B * T; // lnf_mean
        model->act_sizes[19] = B * T; // lnf_rstd
        model->act_sizes[20] = B * T * V; // logits
        model->act_sizes[21] = B * T * V; // probs
        model->act_sizes[22] = B * T; // losses
        model->act_sizes[23] = L * B * T * 3*C; // qkvr
        model->act_sizes[24] = L * B * T * C; // v_accum
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    } else {
        // validate B,T is no larger than what was previously allocated
        // in principle, we could re-allocate a larger chunk of memory, for now we just error out
        if (B > model->batch_size || T > model->seq_len) {
            printf("Error: batch size or sequence length is inadequately large\n");
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(1);
        }
    }
    
    // TODO lazy initialisation, is there any better place to do this?
    if (stream == 0) {
        cudaCheck(cudaStreamCreate(&stream));
    }
    if (!cublaslt_workspace) {
        cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size);
    }
    if (cublas_handle == NULL) {
        cublasCheck(cublasLtCreate(&cublaslt_handle));
        cublasCheck(cublasCreate(&cublas_handle));
        cublasCheck(cublasSetStream(cublas_handle, stream));
        cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    }

    // copy inputs/targets to the model
    // asynchronous on our non-default CUDA stream, avoiding unnecessary CPU/GPU sync points
    cudaCheck(cudaMemcpyAsync(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice, stream));
    if (targets != NULL) {
        cudaCheck(cudaMemcpyAsync(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    // the CUDA graph is only valid for given B/T values, we must recreate it if they change for any reason
    if (graph_created && (B != graph_created_B || T != graph_created_T)) {
        cudaGraphDestroy(graph);
        cudaGraphExecDestroy(instance);
        graph_created = false;
    }

    // create a CUDA graph on the first run, then launch it for higher performance
    // (nothing is actually executed on the GPU during stream capture)
    if (!graph_created) {
        cudaCheckErrors();
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        for (int l = 0; l < L; l++) {

            residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = params.ln1w + l * C;
            float* l_ln1b = params.ln1b + l * C;
            float* l_qkvw = params.qkvw + l * 3*C * C;
            float* l_qkvb = params.qkvb + l * 3*C;
            float* l_attprojw = params.attprojw + l * C * C;
            float* l_attprojb = params.attprojb + l * C;
            float* l_ln2w = params.ln2w + l * C;
            float* l_ln2b = params.ln2b + l * C;
            float* l_fcw = params.fcw + l * 4*C * C;
            float* l_fcb = params.fcb + l * 4*C;
            float* l_fcprojw = params.fcprojw + l * C * 4*C;
            float* l_fcprojb = params.fcprojb + l * C;

            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkv = acts.qkv + l * B * T * 3*C;
            float* l_qkvr = acts.qkvr + l * B * T * 3*C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_preatt = acts.preatt + l * B * NH * T * T;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_v_accum = acts.v_accum + l * B * T * C;
            float* l_attproj = acts.attproj + l * B * T * C;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4*C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
            float* l_fcproj = acts.fcproj + l * B * T * C;
            float* l_residual3 = acts.residual3 + l * B * T * C;

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, true, false);
            attention_forward(l_atty, l_v_accum, l_qkvr, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C, true, false);
            residual_forward(l_residual2, residual, l_attproj, B*T*C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, true, true);
            //gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, true, false);
            residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        }
        
        residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V, false, false);
        softmax_forward(acts.probs, acts.logits, B*T, V);
        
        graph_created = true;
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    }
    // launch the graph but don't wait on it until needed for the cudaMemcpyAsync below
    // this allows us to run keep the GPU running without waiting on the CPU
    cudaGraphLaunch(instance, stream);
    cudaCheckErrors();

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        // crossentropy_forward is not the graph (in case targets is not always provided)
        crossentropy_forward(acts.losses, acts.probs, model->targets, B, T, V);

        // for convenience also evaluate the mean loss
        // move the (B,T) losses to CPU
        // TODO get rid of inline mallocs
        float* cpu_losses = (float*)malloc(B * T * sizeof(float));
        cudaCheck(cudaMemcpyAsync(cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost, 0));
        cudaStreamSynchronize(stream);
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += cpu_losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
        free(cpu_losses);

    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
        cudaStreamSynchronize(stream);
    }
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));

#if defined(ENABLE_ACTIVATION_COMPRESSION)
    freeCompressible((void**)&model->acts_memory, activation_size, true);
#else
    cudaCheck(cudaFree(model->acts_memory));
#endif

    // free the cublas handles and CUDA stream
    if (cublaslt_handle != NULL) {
        cublasCheck(cublasLtDestroy(cublaslt_handle));
        cublasCheck(cublasDestroy(cublas_handle));
        cudaStreamDestroy(stream);
    }

    // free the cublaslt workspace memory
    if (cublaslt_workspace) {
        cudaFree(cublaslt_workspace);
    }

    // destroy the graph
    if (graph_created) {
        cudaGraphDestroy(graph);
        cudaGraphExecDestroy(instance);
        graph_created = false;
    }
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
    // hyperparameters
    int B;
    int T;
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL) {
        printf("Error opening tokens file\n");
        exit(1);
    }

    // determine the file size
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    loader->current_position = 0; // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->batch = (int*) malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    int B = loader->B;
    int T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (B*T+1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T integers
    loader->current_position += B*T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fclose(loader->tokens_file);
    free(loader->batch);
}


// ----------------------------------------------------------------------------
// sampler

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {
    cudaCheck(cudaSetDevice(0));

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    char* tiny_stories_train = "data/TinyStories_train.bin";
    char* tiny_stories_val = "data/TinyStories_val.bin";
    char* tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    char* tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
    char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4;
    int T = 1024;
    DataLoader train_loader;
    dataloader_init(&train_loader, train_tokens, B, T);
    printf("train dataset num_batches: %d\n", train_loader.num_batches);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);
    int val_num_batches = 10;
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);
    printf("val_num_batches: %d\n", val_num_batches);

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    const int gen_max_length = 64;
    int gen_tokens[gen_max_length];
    float* cpu_probs = (float*)malloc(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
            for (int t = 1; t < gen_max_length; t++) {
                // note that inference is wasteful here because
                // for each t, we re-compute all activations between 0 and t
                // leaving this alone because you want separate code for inference anyway
                // the inference here is just for sanity checking purposes
                gpt2_forward(&model, gen_tokens, NULL, 1, t);
                float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
                float coin = random_f32(&rng_state);
                // move probs back to CPU and sample
                cudaCheck(cudaMemcpy(cpu_probs, probs, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                int next_token = sample_mult(cpu_probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
            }
            printf("generated: ");
            for (int t = 0; t < gen_max_length; t++) {
                printf("%d ", gen_tokens[t]);
            }
            printf("\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        // these are still TODO
        // gpt2_zero_grad(&model);
        // gpt2_backward(&model);
        // gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    gpt2_free(&model);
    free(cpu_probs);
    return 0;
}
#endif