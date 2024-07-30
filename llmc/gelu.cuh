/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

__global__ void swiglu_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx); // load and do not keep in cache
    x128 packed_inp2 = load128cs(inp2 + idx);
    for(int k = 0; k < packed_inp1.size; ++k) {
        float x1 = (float)packed_inp1[k];
        float x2 = (float)packed_inp2[k];
        // swish(x1) = x1 * sigmoid(x1) = x1 / (1.0 + exp(-x1))
        // swiglu(x1, x2) = swish(x1) * x2
        packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x1)));
    }
    store128(out + idx, packed_out);
}

__global__ void swiglu_backward_inplace_kernel(floatX* dinp_out1, floatX* dinp2, const floatX* inp1, const floatX* inp2) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp1;
    x128 packed_dinp2;
    x128 packed_inp1 = load128cs(inp1 + idx);
    x128 packed_inp2 = load128cs(inp2 + idx);
    x128 packed_dinp_out1 = load128(dinp_out1 + idx);
    for (int k = 0; k < packed_inp1.size; ++k) {
        float x1 = (float)packed_inp1[k];
        float x2 = (float)packed_inp2[k];
        float sig_x1 = 1.0f / (1.0f + expf(-x1));
        // swiglu(x1, x2) = swish(x1) * x2
        // -> dout/dx1 = x2 * sigmoid(x1) + x2 * x1 * sigmoid(x1) * (1 - sigmoid(x1))
        // ---> dout/dx1 = x2 * sigmoid(x1) * (1 + x1 * (1 - sigmoid(x1)))
        // -> dout/dx2 = swish(x1) = x1 * sigmoid(x1)
        float local_grad1 = x2 * sig_x1 * (1.0f + x1 * (1.0f - sig_x1));
        float local_grad2 = x1 * sig_x1;
        packed_dinp1[k] = (floatX)(local_grad1 * (float)packed_dinp_out1[k]);
        packed_dinp2[k] = (floatX)(local_grad2 * (float)packed_dinp_out1[k]);
    }
    store128(dinp_out1 + idx, packed_dinp1);
    store128(dinp2 + idx, packed_dinp2);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}

void swiglu_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    swiglu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2);
    cudaCheck(cudaGetLastError());
}

void swiglu_backward_inplace(floatX* dinp_out1, floatX* dinp2, const floatX* inp1, const floatX* inp2, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    swiglu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(dinp_out1, dinp2, inp1, inp2);
    cudaCheck(cudaGetLastError());
}
