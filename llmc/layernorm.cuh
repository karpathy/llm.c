/*
LayerNorm CUDA kernel, and also Residual, because sometimes they are fused

Note in llm.c we try to be clever in the backward pass to conserve memory.
All parameters use a += in the backward pass, so we can do gradient accumulation.
But all activations have = instead of += because these are faster (just read, no write).
This is okay for all activations except for those in the residual stream, where the
gradients have to add. We make sure that we do a += as necessary.
E.g., the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "tensor.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template <typename T=floatX>
__global__ void layernorm_forward_kernel6(TensorGPU<T> out, tensor32 mean, tensor32 rstd,
                                          tensorX inp, tensorX weight,
                                          tensorX bias, int N, int C) {
    // Note that blockDim.x must be WARP_SIZE=32 but we don't want to pay the cost of assert() here
    int idx = blockIdx.x * blockDim.y + threadIdx.y; // non-standard: threadIdx.x is used for c
    if(idx >= N) { return; }

    // load/store128 sometimes generated multiple instructions with floatX, so keep it as x128
    extern __shared__ char* params[];
    x128* s_in = reinterpret_cast<x128*>(params) + (threadIdx.y * C / x128::size);

    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        auto inp128 = load_tensor128(inp, idx * C + c, true);
        for(int k = 0; k < x128::size; ++k) {
            sum += inp128.get(k);
        }
        s_in[c / x128::size] = inp128.get128();
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    const float eps = 1e-5f; // todo - is this optimal / theoretically justified?
    float s = rsqrtf(v + eps);

    auto out128 = new_tensor128(out);
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        auto w128 = load_tensor128(weight, c);
        auto b128 = load_tensor128(bias, c);
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m); // normalized output
            float o = n * w128.get(k) + b128.get(k); // scale and shift it
            out128.set(k, o);
        }
        out128.template store_same_length<floatX>(idx * C + c);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) { // todo - add a way to pass equivalent of null for mean/rstd to avoid store
        __stcs(mean + idx, m);
        __stcs(rstd + idx, s);
    }
    // update absmax
    out128.update_absmax(2);
}

template <typename Tout=float8, typename Tin = Tout>
__global__ void fused_residual_forward_kernel5(tensorX residual, TensorGPU<Tout> normed, tensor32 mean, tensor32 rstd,
                                               const tensorX inp1, const TensorGPU<Tin> inp2,
                                               const tensorX weight, const tensorX bias,
                                               int N, int C) {
    // Note that blockDim.x must be WARP_SIZE=32 but we don't want to pay the cost of assert() here
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx > N) return;

    // load/store128 sometimes generated multiple instructions with floatX, so keep it as x128
    extern __shared__ char* params[];
    x128* s_res = reinterpret_cast<x128*>(params) + (threadIdx.y * C / x128::size);

    auto residual128 = new_tensor128(residual);
    auto normed128 = new_tensor128(normed);

    const float eps = 1e-5f;
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        auto inp1_128 = load_tensor128(inp1, idx * C + c, true);
        auto inp2_128 = load_tensor128(inp2, idx * C + c, true);
        for(int k = 0; k < x128::size; ++k) {
            float out = inp1_128.get(k) + inp2_128.get(k);
            residual128.set(k, out);
            sum += residual128.get(k);
        }
        residual128.store(idx * C + c, false);
        s_res[c / x128::size] = residual128.get128();
    }

    sum = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)res[k] - m) * ((float)res[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 res = s_res[c / x128::size];
        auto w128 = load_tensor128(weight, c);
        auto b128 = load_tensor128(bias, c);
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)res[k] - m); // normalized output
            float o = n * w128.get(k) + b128.get(k); // scale and shift it
            normed128.set(k, o);
        }
        normed128.template store_same_length<floatX>(idx * C + c, false);
    }
    // cache the mean and rstd for the backward pass later
    if(threadIdx.x == 0) {
        __stcs(mean + idx, m);
        __stcs(rstd + idx, s);
    }

    // Update absmax for residual and normed tensors (typically it will skip residual as it is not FP8)
    residual128.update_absmax(2);
    normed128.update_absmax(2);
}

template <bool zero_dinp_old=false, typename T=float8e5>
__global__ void __launch_bounds__(512, 2) // todo - any warnings on Turing with only 1024 threads?
    layernorm_backward_kernel10(tensorX dinp_new, tensorX dinp_old, tensorX dweight, tensorX dbias, tensor32 scratch_,
                                TensorGPU<T> dout, tensorX inp, tensorX weight, tensor32 mean, tensor32 rstd,
                                int BT, int C) {
    int BLOCK_SIZE = blockDim.x; // todo - does it make any difference if this is hardcoded here?
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    extern __shared__ float shared[];

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    auto dinp_new128 = new_tensor128(dinp_new);

    for (int bt = baseIdx; bt < BT; bt += warpsInGrid) {
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            auto dout128_i = load_tensor128(dout, bt * C + i);
            auto inp128_i = load_tensor128(inp, bt * C + i);
            auto weight128_i = load_tensor128(weight, i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = weight128_i.get(k) * dout128_i.get(k);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * inp128_i.get(k);
            }
        }

        const float mean_bt = mean[bt];
        const float rstd_bt = rstd[bt];
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            tensor128<T> dout128;
            tensor128<floatX> inp128;
            tensor128<floatX> weight128;
            tensor128<floatX> dinp128;

            if(global_index < C) {
                dout128 = load_tensor128(dout, bt * C + global_index, true);
                inp128 = load_tensor128(inp, bt * C + global_index, true);
                weight128 = load_tensor128(weight, global_index);
                if constexpr (!zero_dinp_old) {
                    dinp128 = load_tensor128(dinp_old, bt * C + global_index);
                }
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = dout128.get(x);
                    float norm_bti = (inp128.get(x) - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += weight128.get(x) * dout128.get(x); // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp_new128.set(x, dinp128.get(x) + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                dinp_new128.store_same_length<floatX>(bt * C + global_index, false);
            }
        }
    }

    // if we did actually update the absmax (returns true), we already did __syncthreads() here
    if (!dinp_new128.update_absmax(1)) {
        __syncthreads();
    }

    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    float* scratch = (float*)scratch_;
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2*C*blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2*C*blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        auto dbias128_out = new_tensor128(dbias);
        auto dweight128_out = new_tensor128(dweight);
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            auto dbias128 = load_tensor128(dbias, global_index);
            auto dweight128 = load_tensor128(dweight, global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128_out.set(x, s_db[i] + dbias128.get(x));
                    dweight128_out.set(x, s_dw[i] + dweight128.get(x));
                }
            }
            dbias128_out.store_same_length<floatX>(global_index);
            dweight128_out.store_same_length<floatX>(global_index);
        }
        dbias128_out.update_absmax(1);
        dweight128_out.update_absmax(1);
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// Helper function to set the block size based on available shared memory and launch the kernel
template<typename KernelFunc, typename... Args>
void launch_layernorm_kernel(KernelFunc kernel, int N, int C, cudaStream_t stream, Args... args) {
    int block_size = 256;
    int block_y = block_size / WARP_SIZE;
    size_t smem = block_y * C * sizeof(floatX);
    auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    // if we don't have enough shared memory, try smaller block sizes down to 32 threads
    // should fit on practically every modern GPU even for very large numbers of channels
    // todo - do we want to manually set the shared memory vs L1 carveout as well?
    while (status != cudaSuccess) {
        if (block_y == 1) {
            printf("ERROR: not enough shared memory for kernel\n");
            exit(EXIT_FAILURE);
        }
        block_y /= 2, block_size /= 2;
        smem = (2 + block_y) * C * sizeof(floatX);
        status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    int grid_size = CEIL_DIV(N, block_y);
    kernel<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(args..., N, C);
    cudaCheck(cudaGetLastError());
}

template <typename T=floatX>
void layernorm_forward(TensorGPU<T> out, tensor32 mean, tensor32 rstd,
                       tensorX inp, const tensorX weight, const tensorX bias,
                       int N, int C, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    launch_layernorm_kernel(layernorm_forward_kernel6<T>, N, C, stream, out, mean, rstd, inp, weight, bias);
}

template <typename Tout=float8, typename Tin = Tout>
void fused_residual_forward5(tensorX residual, TensorGPU<Tout> normed, tensor32 mean, tensor32 rstd,
                             tensorX inp1, TensorGPU<Tin> inp2, tensorX weight, tensorX bias,
                             int N, int C, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    launch_layernorm_kernel(fused_residual_forward_kernel5<Tout, Tin>, N, C, stream, residual, normed, mean, rstd, inp1, inp2, weight, bias);
}

template <typename Tdout=float8e5>
void layernorm_backward(tensorX dinp_new, tensorX dinp_old, tensorX dweight, tensorX dbias, tensor32 scratch,
                        const TensorGPU<Tdout> dout, const tensorX inp, const tensorX weight, tensor32 mean, tensor32 rstd,
                        int BT, int C, cudaStream_t stream=main_stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    const int blocks_per_sm = 2; // supported on every architecture and less cache thrashing than 3
    const int grid_size = blocks_per_sm * deviceProp.multiProcessorCount;
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    cudaCheck(cudaMemsetAsync(scratch, 0, 1 * sizeof(float), stream)); // only need to reset the flag to 0
    if (dinp_old.is_null()) {
        layernorm_backward_kernel10<true><<<grid_size, block_size, shared_mem_size, stream>>>(dinp_new, dinp_old, dweight, dbias, scratch, dout, inp, weight, mean, rstd, BT, C);
    } else {
        layernorm_backward_kernel10<false><<<grid_size, block_size, shared_mem_size, stream>>>(dinp_new, dinp_old, dweight, dbias, scratch, dout, inp, weight, mean, rstd, BT, C);
    }
    cudaCheck(cudaGetLastError());
}
