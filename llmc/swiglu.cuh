/*
SwiGLU activation function
Unlike GeLU, SwiGLU is a bit more tricky because there are two separate linear layers.
In PyTorch we have:

self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)

and then:

x1 = self.c_fc(x)
x2 = self.c_fc2(x)
x2 = F.silu(x2)
x = x1 * x2

But in our implementation to minimize the amount of changes, we have the weights of
the two linear layers concatenated together. So in this non-linearity, we receive
as input the conctatenation of [x1, x2], and our job is just to apply silu and
elementwise multiply. And we have to be careful because the output size is half
the input size!
*/

#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void swiglu_forward_kernel1(floatX* out, const floatX* inp, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    floatX* out_ptr = out + idx;
    // b,t,c in the output
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;

    int C2 = C * 2;
    const floatX* inp1_ptr = inp + (b * T * C2 + t * C2 + c);
    const floatX* inp2_ptr = inp1_ptr + C;

    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1_ptr); // fc1
    x128 packed_inp2 = load128cs(inp2_ptr); // fc2
    for(int k = 0; k < packed_inp1.size; ++k) {
        float x1 = (float)packed_inp1[k];
        float x2 = (float)packed_inp2[k];
        packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
    }
    store128(out_ptr, packed_out);
}

__global__ void swiglu_forward_kernel2(floatX* out, const floatX* inp, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // derive the b,t,c from idx
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;
    int C2 = C * 2;
    float x1 = (float) inp[b * T * C2 + t * C2 + c];
    float x2 = (float) inp[b * T * C2 + t * C2 + c + C];
    out[idx] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
}

__global__ void swiglu_backward_kernel1(floatX* dinp, const floatX* dout, const floatX* inp, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    const floatX* dout_ptr = dout + idx;
    // b,t,c in the output
    int b = idx / (T * C);
    int t = (idx / C) % T;
    int c = idx % C;
    // coords in input
    int C2 = C * 2;
    const floatX* inp1_ptr = inp + (b * T * C2 + t * C2 + c);
    const floatX* inp2_ptr = inp1_ptr + C;
    floatX* dinp1_ptr = dinp + (b * T * C2 + t * C2 + c);
    floatX* dinp2_ptr = dinp1_ptr + C;
    // backward
    x128 dinp1;
    x128 dinp2;
    x128 packed_dout = load128cs(dout_ptr);
    x128 packed_inp1 = load128cs(inp1_ptr); // fc1
    x128 packed_inp2 = load128cs(inp2_ptr); // fc2
    for(int k = 0; k < packed_inp1.size; ++k) {
        float x1 = (float)packed_inp1[k];
        float x2 = (float)packed_inp2[k];
        float dout = (float)packed_dout[k];

        float sx2 = 1.0f / (1.0f + expf(-x2)); // sigmoid of x2
        float dx1 = dout * x2 * sx2;
        float dx2 = dout * x1 * sx2 * (1.0f + x2 * (1.0f - sx2));

        dinp1[k] = (floatX)dx1;
        dinp2[k] = (floatX)dx2;
    }
    store128(dinp1_ptr, dinp1);
    store128(dinp2_ptr, dinp2);
}

// ----------------------------------------------------------------------------
// kernel launchers

void swiglu_forward(floatX* out, const floatX* inp, int B, int T, int C, cudaStream_t stream) {
    // input is (B, T, 2C), output is (B, T, C)
    // we have that inp[b, t, :] = [fc1, fc2] (i.e. they are concatenated in each C-fiber)
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert((B*T*C) % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(B*T*C, block_size * x128::size);
    swiglu_forward_kernel1<<<grid_size, block_size, 0, stream>>>(out, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void swiglu_forward_naive(floatX* out, const floatX* inp, int B, int T, int C, cudaStream_t stream) {
    // same as above but no x128 packing to be SAFE
    const int block_size = 128;
    const int grid_size = CEIL_DIV(B*T*C, block_size);
    swiglu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

void swiglu_backward(floatX* dinp, const floatX* dout, const floatX* inp, int B, int T, int C, cudaStream_t stream) {
    // input is (B, T, 2C), output is (B, T, C)
    // we have that inp[b, t, :] = [fc1, fc2] (i.e. they are concatenated in each C-fiber)
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert((B*T*C) % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(B*T*C, block_size * x128::size);
    swiglu_backward_kernel1<<<grid_size, block_size, 0, stream>>>(dinp, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}
