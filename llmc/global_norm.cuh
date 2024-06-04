/*
Global norm, used in gradient clipping
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

template<class T>
__global__ void global_norm_squared_kernel(float* out, const T* data, size_t count) {
    // we want as few atomics as possible, so each block tries to do
    // the maximum amount of work (so no fixed chunk, but instead iterating
    // until we run out of data), and then we reduce inside the block
    // and finally have just one atomic per block.
    // out will be updated atomically from all thread blocks. It is a float, so the
    // atomic op is unproblematic
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // warp-level reduce
    float block_sum = blockReduce<warpReduceSum>(accumulator);
    if(threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

template<typename T>
void global_norm_squared(float* out, const T* values, size_t count, cudaStream_t stream) {
    const int block_size = 512;
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    // initialize out with zero
    cudaCheck(cudaMemsetAsync(out, 0, sizeof(float), stream));
    global_norm_squared_kernel<<<grid_size, block_size, 0, stream>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

