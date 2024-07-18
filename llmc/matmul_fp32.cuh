/*
Matrix Multiplication in FP32/TF32 without any help from cuBLAS(Lt)
Only used in train_gpt2fp32.cu (rather than the main train_gpt2.cu)
*/
#include <mma.h>
#include <cuda/pipeline>

// use "./train_gpt2fp32cu -c 1" for the custom TF32 kernel and "-c 3" for the custom FP32 kernel
// c=0/1 will also force all non-forward matmul kernels to be TF32, while c=2/3 will force FP32
// for test_gpt2fp32cu, cuBLAS FP32 will always be used (no command line options)
// if you want to force cuBLAS or custom TF32/FP32 kernels, change the values below
constexpr bool FORCE_FORWARD_MATMUL_CUBLAS = false;
constexpr bool FORCE_FORWARD_MATMUL_TF32 = false; // incompatible with pre-A100 GPUs
constexpr bool FORCE_FORWARD_MATMUL_FP32= false;

// -----------------------------------------------------------------------------------
// FP32 non-Tensor Core Kernel (baseline)

// note that this kernel is effectively row-major while cuBLAS is column-major
// so we need to pass "A as B" and "B as A" in order to get the same C/D layout as cuBLAS
__global__ void __launch_bounds__(16*16, 2) matmul_forward_kernel4(float* out,
                                                                   const float* inp, const float* weight, const float* bias,
                                                                   int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    int si_start = 4*(16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so += 32) {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4) {
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji) {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}

// -----------------------------------------------------------------------------------
// TF32 WMMA Tensor Core Kernel
// this is a port of NVIDIA's CUDA 12.4 sample "tf32TensorCoreGemm" for llm.c
// https://github.com/NVIDIA/cuda-samples/tree/v12.4/Samples/3_CUDA_Features/tf32TensorCoreGemm

// keeping similar define names except M/N/K which cannot exist at global scope to avoid conflicts
// various defines have also been replaced by variables derived from function parameters m/n/k
#define WMMA_WARPS_PER_BLOCK 8
#define WMMA_THREADS_PER_BLOCK (WARP_SIZE * WMMA_WARPS_PER_BLOCK)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define CHUNK_K 8
#define CHUNK_LINE_BYTES (CHUNK_K * WMMA_K * sizeof(float))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)
#define SHMEM_STRIDE (WMMA_N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (WMMA_N * WARP_ROW_TILES)
#define SKEW_FLOAT 8

// shared memory size required for each block by the TF32 WMMA kernel
// 70KiB+ for default settings => 2 blocks on A100 but only 1 on GA102/AD102 unfortunately...
constexpr int WMMA_SHMEM_SIZE1 = (BLOCK_COL_TILES * WMMA_M) * (CHUNK_K * WMMA_K + SKEW_FLOAT) * 2;
constexpr int WMMA_SHMEM_SIZE2 = (WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES));
constexpr int WMMA_SHMEM_SIZE = (WMMA_SHMEM_SIZE1 > WMMA_SHMEM_SIZE2 ? WMMA_SHMEM_SIZE1 : WMMA_SHMEM_SIZE2) * sizeof(float);

// transpose for WMMA (equivalent to cuBLAS_OP_T etc.)
// note that this kernel is effectively row-major while cuBLAS is column-major
// so we need to pass "A as B" and "B as A" in order to get the same C/D layout as cuBLAS
// but the transposes should be the same (cuBLAS: A=inp^T / B=weight - WMMA: A=weight^T / B=inp)
using WMMA_T =  nvcuda::wmma::row_major;
using WMMA_NT = nvcuda::wmma::col_major;

template <typename A_major=WMMA_T, typename B_major=WMMA_NT>
__global__ void compute_tf32gemm_async_copy(float *D,
                                            const float *A, const float *B, const float *C, const float *bias,
                                            const float alpha, float beta, int m, int n, int k) {
#if __CUDA_ARCH__ >= 800
    assert(m % WMMA_M == 0 && n % WMMA_N == 0 && k % WMMA_K == 0);

    using namespace nvcuda;
    constexpr int M = WMMA_M;
    constexpr int N = WMMA_N;
    constexpr int K = WMMA_K;
    const int K_GLOBAL = k;
    const int M_TILES = m / WMMA_M;
    const int N_TILES = n / WMMA_N;
    const int K_TILES = k / WMMA_K;
    const int GLOBAL_MEM_STRIDE = n;

    extern __shared__ float shmem[][CHUNK_K * K + SKEW_FLOAT];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId / BLOCK_ROW_WARPS) * SHMEM_STRIDE * N * BLOCK_ROW_WARPS + (warpId % BLOCK_ROW_WARPS) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * N;

    // Offset in shared memory from which the B matrix is stored.
    constexpr size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may result
    // in a loss of precision). Zero still needs to be specially handled though.
    beta /= alpha;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    constexpr int loadStride = 2; // load 4 floats, so left-shift by 2.

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
    // right and down, and selects the next tile to compute. Once there's no such tile,
    // all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory (llm.c: if beta is not 0)
        if (beta != 0.0f) {
            #pragma unroll
            for (int i = 0; i < N; i++) {
                pipe.producer_acquire();
                cuda::memcpy_async(&shmem_warp_stream_ptr[(SHMEM_STRIDE * i) + (laneId << loadStride)],
                                   &src_gmem_warp_stream_ptr[(GLOBAL_MEM_STRIDE * i) + (laneId << loadStride)],
                                   shape4, pipe);
                pipe.producer_commit();
            }
            // Now wait for all the above issued 8 batches to complete.
            cuda::pipeline_consumer_wait_prior<0>(pipe);
            __syncthreads();
        }

        // These fragments will accumulate the result of A and B matrix fragment multiplications
        // along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
        // llm.c: if beta is 0, then don't load C, just initialize the accumulator to 0.
        if (beta == 0.0f) {
            #pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                    for (int t = 0; t < c[i][j].num_elements; t++) {
                        c[i][j].x[t] = 0.0f;
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                    const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;

                    wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, wmma::mem_row_major);
                    // Scale the C matrix.
                    #pragma unroll
                    for (int t = 0; t < c[i][j].num_elements; t++) {
                        c[i][j].x[t] *= beta;
                    }
                }
            }
        }
        pipe.consumer_release();

        // sync here so that shared memory can then be used for loading A & B matrices.
        __syncthreads();

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const float *warp_ptr = (warpId < (WMMA_WARPS_PER_BLOCK / 2)) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % (WMMA_WARPS_PER_BLOCK / 2)) * 2) : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % (WMMA_WARPS_PER_BLOCK / 2)) * 2);

        constexpr int chunksPerLane = ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
        const int laneLoadElem = (laneId % CHUNK_COPY_LINE_LANES) << loadStride;
        const int stridePerLaneCopy = (laneId / CHUNK_COPY_LINE_LANES);
        // Go through the global K dimension by a fixed step at a time.
        #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            // As for tf32 MMA  M == N we use M for warp 4-7 + shmem_idx_b_off.
            size_t shmem_idx = (M * (warpId % (WMMA_WARPS_PER_BLOCK / 2)) * 2) + ((warpId / (WMMA_WARPS_PER_BLOCK / 2)) * shmem_idx_b_off);
            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            const float *lane_ptr = (warp_ptr + tile_k * K + stridePerLaneCopy * K_GLOBAL + laneLoadElem);

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += stridePerLaneCopy;

            #pragma unroll
            for (int i = 0; i < chunksPerLane; i++) {
                // Copy 16 bytes at once in each lane.
                pipe.producer_acquire();
                cuda::memcpy_async(&shmem[shmem_idx][laneLoadElem], lane_ptr, shape4, pipe);
                pipe.producer_commit();

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP;
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            cuda::pipeline_consumer_wait_prior<0>(pipe);
            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, A_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, B_major> b[WARP_ROW_TILES];

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / BLOCK_ROW_WARPS) * M * BLOCK_ROW_WARPS + (i * M);
                    const float *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_FLOAT);

                    #pragma unroll
                    for (int t = 0; t < a[i].num_elements; t++) {
                        a[i].x[t] = wmma::__float_to_tf32(a[i].x[t]);
                    }
                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be reused
                            // against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const float *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_FLOAT);
                            #pragma unroll
                            for (int t = 0; t < b[j].num_elements; t++) {
                                b[j].x[t] = wmma::__float_to_tf32(b[j].x[t]);
                            }
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            pipe.consumer_release();
            __syncthreads();
        }

        // Store the D fragments to shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                #pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                // warp are well-defined even though element indices within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= alpha;
                }
                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * N + j * N;
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

        #pragma unroll
        for (int i = 0; i < N; i++) {
            // todo - adding bias in a hacky way at the last step because it made indexing easier
            float4 *out_ptr = (float4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId;
            float4 shmem_data = *((float4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
            if (bias != NULL) {
                ptrdiff_t column = (ptrdiff_t)((float *)out_ptr - D) % n;
                float4 bias4 = ld_vec(bias + column);
                *out_ptr = add_float4(shmem_data, bias4);
                // todo: __stcs doesn't seem to help in practice on H100, unsure about other GPUs?
                //__stcs(out_ptr, add_float4(shmem_data, bias4));
            } else {
                *out_ptr = shmem_data;
                //__stcs(out_ptr, shmem_data);
            }
        }
        __syncthreads();
    }
#endif
}

// -----------------------------------------------------------------------------------
// cuBLAS baseline for forward pass with separate bias kernel
// cuBLASLt allows merging this which would significantly improve performance

__global__ void add_bias(float* out, float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B*T*OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// -----------------------------------------------------------------------------------
// launchers

void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)

    if (!FORCE_FORWARD_MATMUL_FP32 && !FORCE_FORWARD_MATMUL_TF32) {
        if (custom_matmul_kernel == 0 || custom_matmul_kernel == 2 || FORCE_FORWARD_MATMUL_CUBLAS) {
            // cuBLAS is column-major like FORTRAN, while our kernels are row-major, so A/B are reversed
            const float one = 1.0f, zero = 0.0f;
            cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &one, weight, C, inp, C, &zero, out, OC));
            if (bias != NULL) {
                add_bias<<<min(4096, CEIL_DIV(B*T*OC, 256)), 256>>>(out, bias, B, T, OC);
                cudaCheck(cudaGetLastError());
            }
            return;
        }
    }

    if (!FORCE_FORWARD_MATMUL_FP32) {
        // TF32 WMMA and async require A100+ GPUs -> fallback to FP32 kernel if not available
        if ((custom_matmul_kernel == 1 || FORCE_FORWARD_MATMUL_TF32) && deviceProp.major >= 8)
        {
            cudaCheck(cudaFuncSetAttribute(compute_tf32gemm_async_copy<WMMA_T,WMMA_NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, WMMA_SHMEM_SIZE));
            compute_tf32gemm_async_copy<WMMA_T,WMMA_NT><<<deviceProp.multiProcessorCount*2, WMMA_THREADS_PER_BLOCK, WMMA_SHMEM_SIZE>>>
                                                    (out, inp, weight, out, bias, 1.0f, 0.0f, B*T, OC, C);
            cudaCheck(cudaGetLastError());
            return;
        }
    }

    // safe fallback: custom FP32 kernel
    int sqrt_block_size = 16;
    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);

    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}
