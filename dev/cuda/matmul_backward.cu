/*
Kernels for matmul backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// naive kernel to backpropagate only the bias, it's just a sum :'(
__global__ void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < OC) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

// use shared memory and coarsening + reductions
__global__ void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC)
    int tid = threadIdx.x; // range [0, block_size)
    int block_size = blockDim.x;
    const float* x = dout + o;
    // thread coarsening
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];
    }
    shared[tid] = (float) sum;
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
        dbias[o] = shared[0];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// version1: simple cuBLAS calls
void matmul_backward1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // backward to input
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));
    // backward to weight
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &alpha, inp, C, dout, OC, &beta, dweight, C));
    // backward to bias, if given
    if (dbias != NULL) {

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // bit faster
        const int block_size=512;
        dim3 block_dim(block_size);
        dim3 grid_dim(OC);
        size_t shared_mem_size = block_size * sizeof(float);
        matmul_backward_bias_kernel_faster<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, B, T, OC);
    }
}

void matmul_backward(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and its mathmodes, ensure fp32
    int enable_tf32 = 0; // use fp32 to get accurate results for checking w.r.t. CPU
    cublasCheck(cublasCreate(&cublas_handle));
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // create host memory of random numbers
    float* dinp = make_zeros_float(B * T * C);
    float* dweight = make_zeros_float(OC * C);
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_ones, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dweight, dweight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dbias, dbias, OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_ones, ones, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    printf("time %.4f ms\n", elapsed_time);

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    free(ones);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_ones));
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}