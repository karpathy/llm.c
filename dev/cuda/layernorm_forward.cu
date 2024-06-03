/*
Kernels for layernorm forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o
layernorm_forward

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops
over C
./layernorm_forward 1

version 2 parallelizes over all of B,T,C
./layernorm_forward 2

version 3 uses cooperative groups to parallelize over all of B,T,C
./layernorm_forward 3

version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) -
mean(x)**2 (allowing us to do a single pass over x on load)
./layernorm_forward 4

verstion 5 allocates blocks per row instead of warps per row, same alg as 4
otherwise
./layernorm_forward 5
*/
#include "common.h"
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float *out, float *mean, float *rstd,
                           const float *inp, const float *weight,
                           const float *bias, int B, int T, int C) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < T; t++) {
      // seek to the input position inp[b,t,:]
      const float *x = inp + b * T * C + t * C;
      // calculate the mean
      float m = 0.0f;
      for (int i = 0; i < C; i++) {
        m += x[i];
      }
      m = m / C;
      // calculate the variance (without any bias correction)
      float v = 0.0f;
      for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C;
      // calculate the rstd
      float s = 1.0f / sqrtf(v + eps);
      // seek to the output position in out[b,t,:]
      float *out_bt = out + b * T * C + t * C;
      for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));        // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_bt[i] = o;                     // write
      }
      // cache the mean and rstd for the backward pass later
      mean[b * T + t] = m;
      rstd[b * T + t] = s;
    }
  }
}

// --------------------------------------------------------------------------
// Memory Management

// allocate & set host & device memory, pinned memory allocation if necessary
// pinned memcpy is interleaved with kernel invocation (cudaMemcpyAsync)
// in order to achieve copy compute overlap
void prepareMemory(bool pinned, float **out, float **mean, float **rstd,
                   float **inp, float **weight, float **bias, float **d_out,
                   float **d_mean, float **d_rstd, float **d_inp,
                   float **d_weight, float **d_bias, int B, int T, int C) {

  cudaCheck(cudaMalloc(d_out, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(d_mean, B * T * sizeof(float)));
  cudaCheck(cudaMalloc(d_rstd, B * T * sizeof(float)));
  cudaCheck(cudaMalloc(d_inp, B * T * C * sizeof(float)));
  cudaCheck(cudaMalloc(d_weight, C * sizeof(float)));
  cudaCheck(cudaMalloc(d_bias, C * sizeof(float)));

  if (pinned) {
    cudaCheck(
        cudaHostAlloc(out, B * T * C * sizeof(float), cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(mean, B * T * sizeof(float), cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(rstd, B * T * sizeof(float), cudaHostAllocDefault));

    *inp = make_random_float_pinned(B * T * C);
    *weight = make_random_float_pinned(C);
    *bias = make_random_float_pinned(C);
  } else {
    *out = (float *)malloc(B * T * C * sizeof(float));
    *mean = (float *)malloc(B * T * sizeof(float));
    *rstd = (float *)malloc(B * T * sizeof(float));

    *inp = make_random_float(B * T * C);
    *weight = make_random_float(C);
    *bias = make_random_float(C);

    cudaCheck(cudaMemcpy(*d_inp, *inp, B * T * C * sizeof(float),
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(*d_weight, *weight, C * sizeof(float),
                         cudaMemcpyHostToDevice));
    cudaCheck(
        cudaMemcpy(*d_bias, *bias, C * sizeof(float), cudaMemcpyHostToDevice));
  }
}

// free memory, pinned memory requires cudaFreeHost
void resetMemory(bool pinned, float **out, float **mean, float **rstd,
                 float **inp, float **weight, float **bias, float **d_out,
                 float **d_mean, float **d_rstd, float **d_inp,
                 float **d_weight, float **d_bias) {
  cudaCheck(cudaFree(*d_out));
  cudaCheck(cudaFree(*d_mean));
  cudaCheck(cudaFree(*d_rstd));
  cudaCheck(cudaFree(*d_inp));
  cudaCheck(cudaFree(*d_weight));
  cudaCheck(cudaFree(*d_bias));
  if (pinned) {
    cudaCheck(cudaFreeHost(*out));
    cudaCheck(cudaFreeHost(*mean));
    cudaCheck(cudaFreeHost(*rstd));
    cudaCheck(cudaFreeHost(*inp));
    cudaCheck(cudaFreeHost(*weight));
    cudaCheck(cudaFreeHost(*bias));
  } else {
    free(*out);
    free(*mean);
    free(*rstd);
    free(*inp);
    free(*weight);
    free(*bias);
  }
}

bool isPinnedMemory(int pinnedMemoryKernels[], int kernelNum, size_t N) {
  for (int i = 0; i < N; i++) {
    if (kernelNum == pinnedMemoryKernels[i])
      return true;
  }

  return false;
}
// ----------------------------------------------------------------------------
// GPU kernels

// naive drag and drop implementation into kernel, parallelize over B,T, loop
// over C
__global__ void layernorm_forward_kernel1(float *out, float *mean, float *rstd,
                                          const float *inp, const float *weight,
                                          const float *bias, int N, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1e-5f;

  if (idx < N) {
    // seek to the input position inp[idx,:]
    const float *x = inp + idx * C;
    // calculate the mean
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
      m += x[i];
    }
    m = m / C;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
      float xshift = x[i] - m;
      v += xshift * xshift;
    }
    v = v / C;
    // calculate the rstd
    float s = 1.0f / sqrtf(v + eps);
    // seek to the output position in out[idx,:]
    float *out_idx = out + idx * C;
    for (int i = 0; i < C; i++) {
      float n = (s * (x[i] - m));        // normalized output
      float o = n * weight[i] + bias[i]; // scale and shift it
      out_idx[i] = o;                    // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = m;
    rstd[idx] = s;
  }
}

__global__ void mean_kernel(float *mean, const float *inp, int N, int C,
                            int block_size) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;  // range [0, B*T)
  int tid = threadIdx.x; // range [0, block_size)
  const float *x = inp + idx * C;
  // thread coarsening
  float sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sum += x[i];
  }
  shared[tid] = sum;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
  }
  // write the final result (at thread 0) to global memory
  if (tid == 0) {
    mean[idx] = shared[0] / C;
  }
}

__global__ void rstd_kernel(float *rstd, const float *inp, const float *mean,
                            int N, int C, int block_size) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;  // range [0, B*T)
  int tid = threadIdx.x; // range [0, block_size)
  const float *x = inp + idx * C;
  float m = mean[idx];
  // thread coarsening
  float sum = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  shared[tid] = sum;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
  }
  // write the final result (at thread 0) to global memory
  if (tid == 0) {
    rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
  }
}

__global__ void normalization_kernel(float *out, const float *inp, float *mean,
                                     float *rstd, const float *weight,
                                     const float *bias, int B, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int bt = idx / C;
  int c = idx % C;

  float m = mean[bt];
  float s = rstd[bt];
  float xi = inp[idx];
  float n = s * (xi - m);
  float o = n * weight[c] + bias[c];

  out[idx] = o;
}

__global__ void layernorm_forward_kernel3(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  // meta_group_size is the number of warps in a block, and meta_group_rank is
  // the warp index
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (idx >= N) {
    return;
  }

  // the row of input that this group of threads is responsible for
  const float *x = inp + idx * C;

  // mean
  float sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    sum += x[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float m = sum / C;
  if (warp.thread_rank() == 0 && mean != nullptr) {
    __stcs(mean + idx, m);
  }

  // rstd
  sum = 0.0f;
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float diff = x[i] - m;
    sum += diff * diff;
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});
  float s = rsqrtf(sum / C + 1e-5f);
  if (warp.thread_rank() == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }

  // final normalization and scaling by weight/bias
  float *o = out + idx * C;
  for (int c = warp.thread_rank(); c < C; c += warp.size()) {
    // load and store using the .cs "streaming" hint to the compiler,
    // indicating that this data will not be reused soon, and can be streamed
    // through the caches this allows the threads to get more cache-hits for the
    // (shared) weight and bias parameters
    float n = s * (__ldcs(x + c) - m);
    __stcs(o + c, n * weight[c] + bias[c]);
  }
}

// same as kernel 3 but uses var(x) == mean(x**2) - mean(x)**2
__global__ void layernorm_forward_kernel4(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (idx >= N) {
    return;
  }

  // the row of input that this group of threads is responsible for
  const float *x = inp + idx * C;

  // thread coarsening through the row, reduce the sum in series
  float sum = 0.0;  // stores sum(x)
  float sum2 = 0.0; // stores sum(x**2)
  for (int i = warp.thread_rank(); i < C; i += warp.size()) {
    float xi = x[i];
    sum += xi;
    sum2 += xi * xi;
  }
  // warp-level reduction at the end
  sum = cg::reduce(warp, sum, cg::plus<float>{});   // sum(x)
  sum2 = cg::reduce(warp, sum2, cg::plus<float>{}); // sum(x**2)
  sum /= C;                                         // mean(x)
  sum2 /= C;                                        // mean(x**2)

  // mean, var, rstd
  float m = sum;
  float var = sum2 - sum * sum;
  float s = rsqrtf(var + 1e-5f);

  // store the mean, no need to cache it
  if (warp.thread_rank() == 0 && mean != nullptr) {
    __stcs(mean + idx, m);
  }
  // store the rstd, no need to cache it
  if (warp.thread_rank() == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }
  // final normalization and scaling by weight/bias
  float *o = out + idx * C;
  for (int c = warp.thread_rank(); c < C; c += warp.size()) {
    float n = s * (__ldcs(x + c) - m);
    __stcs(o + c, n * weight[c] + bias[c]);
  }
}

// like 4, but in kernel 5 we have each block doing one row, not just a single
// warp
__global__ void layernorm_forward_kernel5(
    float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
    const float *__restrict__ inp, const float *__restrict__ weight,
    const float *__restrict__ bias, int N, int C) {
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  __shared__ float shared_sum[32];  // block_size max is 1024 = 32 * 32 warps
  __shared__ float shared_sum2[32]; // warps will be writing into shared memeory
                                    // after warp-reduce
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int idx = blockIdx.x; // simpoy one block per row
  // the row of input that this group of threads is responsible for
  const float *x = inp + idx * C;
  // thread coarsening through the row, reduce the sum in series
  float thread_sum = 0.0;  // stores sum(x)
  float thread_sum2 = 0.0; // stores sum(x**2)
  // for (int i = C + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x) {
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float xi = x[i];
    thread_sum += xi;
    thread_sum2 += xi * xi;
  }
  // warp-level reduction
  float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{}); // sum(x)
  float warp_sum2 =
      cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x**2)
  // store the warp-level reduction in shared memory (we could have lane_id == 0
  // guard but not needed)
  shared_sum[warp_id] = warp_sum;
  shared_sum2[warp_id] = warp_sum2;
  __syncthreads();
  // load results from shared memory to threads, pad with zeros for threads that
  // are out of bounds
  warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
  warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
  // now reduce the warp-level reductions
  float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{}); // sum(x)
  float block_sum2 =
      cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
  // mean, var, rstd
  block_sum /= C;  // mean(x)
  block_sum2 /= C; // mean(x**2)
  float m = block_sum;
  float var = block_sum2 - m * m;
  float s = rsqrtf(var + 1e-5f);
  // store the mean, no need to cache it
  if (threadIdx.x == 0 && mean != nullptr) {
    __stcs(mean + idx, m);
  }
  // store the rstd, no need to cache it
  if (threadIdx.x == 0 && rstd != nullptr) {
    __stcs(rstd + idx, s);
  }
  // final normalization and scaling by weight/bias
  float *o = out + idx * C;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    float n = s * (__ldcs(x + i) - m);
    __stcs(o + i, n * weight[i] + bias[i]);
  }
}

// ----------------------------------------------------------------------------
// kernel launcher
void layernorm_forward1(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  const int N = B * T;
  const int grid_size = ceil_div(N, block_size);
  layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp,
                                                       weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward6(float *d_out, float *d_mean, float *d_rstd,
                        const float *d_inp, const float *d_weight,
                        const float *d_bias, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  const int nStreams = 8;
  const int nChunk = 64;
  const int N = nChunk * T;
  size_t sData = T * C * sizeof(float);
  size_t sChunk = nChunk * sData;
  const int grid_size = ceil_div(N, block_size);

  cudaStream_t streams[nStreams];
  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }
  cudaCheck(cudaGetLastError());

  for (int b = 0, sNum = 0; b < B; b += nChunk, sNum = (sNum + 1) % nStreams) {
    cudaCheck(cudaMemcpyAsync((void *)d_inp, (void *)inp, sChunk,
                              cudaMemcpyHostToDevice, streams[sNum]));
    cudaCheck(cudaMemcpyAsync((void *)d_weight, (void *)weight, sData,
                              cudaMemcpyHostToDevice, streams[sNum]));
    cudaCheck(cudaMemcpyAsync((void *)d_bias, (void *)bias, sData,
                              cudaMemcpyHostToDevice, streams[sNum]));

    layernorm_forward_kernel1<<<grid_size, block_size, 0, streams[sNum]>>>(
        d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, N, C);

    d_out = d_out + nChunk * T * C;
    d_mean = d_mean + nChunk * T;
    d_rstd = d_rstd + nChunk * T;
    d_inp = d_inp + nChunk * T * C;
  }

  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());

  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
}

void layernorm_forward2(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  int N = B * T;
  // in mean and rstd, threads cooperate within blocks via reductions
  mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(mean, inp, N,
                                                                 C, block_size);
  cudaCheck(cudaGetLastError());
  rstd_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(
      rstd, inp, mean, N, C, block_size);
  cudaCheck(cudaGetLastError());
  // in the normalization, everything just gets flattened out
  const int block_size2 = 256;
  const int grid_size = ceil_div(B * T * C, block_size2);
  normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight,
                                                   bias, B, T, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward3(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  assert(block_size % 32 == 0);
  const int N = B * T;
  const int grid_size = ceil_div(N * 32, block_size);
  layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp,
                                                       weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward4(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  assert(block_size % 32 == 0);
  const int N = B * T;
  const int grid_size = ceil_div(N * 32, block_size);
  layernorm_forward_kernel4<<<grid_size, block_size>>>(out, mean, rstd, inp,
                                                       weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

void layernorm_forward5(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int B, int T,
                        int C, const int block_size) {
  assert(block_size % 32 == 0);
  const int N = B * T;
  const int grid_size = N;
  layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp,
                                                       weight, bias, N, C);
  cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void layernorm_forward(int kernel_num, float **d_out, float **d_mean,
                       float **d_rstd, float **d_inp, float **d_weight,
                       float **d_bias, float **out, float **mean, float **rstd,
                       float **inp, float **weight, float **bias, int B, int T,
                       int C, const int block_size) {
  switch (kernel_num) {
  case 1:
    prepareMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward1(*d_out, *d_mean, *d_rstd, *d_inp, *weight, *bias, B, T,
                       C, block_size);
    resetMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                d_rstd, d_inp, d_weight, d_bias);
    break;
  case 2:
    prepareMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward2(*d_out, *d_mean, *d_rstd, *d_inp, *weight, *bias, B, T,
                       C, block_size);
    resetMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                d_rstd, d_inp, d_weight, d_bias);
    break;
  case 3:
    prepareMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward3(*d_out, *d_mean, *d_rstd, *d_inp, *weight, *bias, B, T,
                       C, block_size);
    resetMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                d_rstd, d_inp, d_weight, d_bias);
    break;
  case 4:
    prepareMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward4(*d_out, *d_mean, *d_rstd, *d_inp, *weight, *bias, B, T,
                       C, block_size);
    resetMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                d_rstd, d_inp, d_weight, d_bias);
    break;
  case 5:
    prepareMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward5(*d_out, *d_mean, *d_rstd, *d_inp, *weight, *bias, B, T,
                       C, block_size);
    resetMemory(false, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                d_rstd, d_inp, d_weight, d_bias);
    break;
  case 6:
    prepareMemory(true, out, mean, rstd, inp, weight, bias, d_out, d_mean,
                  d_rstd, d_inp, d_weight, d_bias, B, T, C);
    layernorm_forward6(*d_out, *d_mean, *d_rstd, *d_inp, *d_weight, *d_bias,
                       *inp, *weight, *bias, B, T, C, block_size);
    resetMemory(true, out, mean, rstd, inp, weight, bias, d_out, d_mean, d_rstd,
                d_inp, d_weight, d_bias);
    break;
  default:
    printf("Invalid kernel number\n");
    exit(1);
  }
}

int main(int argc, char **argv) {
  srand(0);

  int B = 256;
  int T = 1024;
  int C = 768;

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  // host pointers
  float *out = NULL;
  float *mean = NULL;
  float *rstd = NULL;
  float *inp = NULL;
  float *weight = NULL;
  float *bias = NULL;

  // device pointers
  float *d_out = NULL;
  float *d_mean = NULL;
  float *d_rstd = NULL;
  float *d_inp = NULL;
  float *d_weight = NULL;
  float *d_bias = NULL;

  // read kernel_num from command line
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  printf("Using kernel %d\n", kernel_num);

  // int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  int block_sizes[] = {256};
  int pinned_memory_kernels[1] = {6};
  bool pinned;
  layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

  // check the correctness of the kernel at all block sizes
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);

    layernorm_forward(kernel_num, &d_out, &d_mean, &d_rstd, &d_inp, &d_weight,
                      &d_bias, &out, &mean, &rstd, &inp, &weight, &bias, B, T,
                      C, block_size);

    pinned = isPinnedMemory(pinned_memory_kernels, kernel_num,
                            sizeof(pinned_memory_kernels));
    resetMemory(pinned, &out, &mean, &rstd, &inp, &weight, &bias, &d_out,
                &d_mean, &d_rstd, &d_inp, &d_weight, &d_bias);

    validate_result(d_out, out, "out", B * T * C, 1e-5f);
    validate_result(d_mean, mean, "mean", B * T, 1e-5f);
    validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
    int block_size = block_sizes[j];

    int repeat_times = 2000;
    float elapsed_time = benchmark_kernel(
        repeat_times, layernorm_forward, kernel_num, &d_out, &d_mean, &d_rstd,
        &d_inp, &d_weight, &d_bias, &out, &mean, &rstd, &inp, &weight, &bias, B,
        T, C, block_size);

    pinned = isPinnedMemory(pinned_memory_kernels, kernel_num,
                            sizeof(pinned_memory_kernels));
    resetMemory(pinned, &out, &mean, &rstd, &inp, &weight, &bias, &d_out,
                &d_mean, &d_rstd, &d_inp, &d_weight, &d_bias);

    // napkin math: estimate the memory bandwidth achieved
    // e.g. A100 40GB PCIe is advertised at 1,555GB/s
    long memory_ops = (2 * B * T * C) * 4; // *4 for float
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size,
           elapsed_time, memory_bandwidth);
  }

  return 0;
}
