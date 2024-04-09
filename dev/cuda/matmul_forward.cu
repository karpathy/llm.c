/*
Kernels for matmul forward pass.
It's advised to use OpenMP here because the CPU implementation is fairly slow otherwise

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas

version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 parallelizes over all of B,T,C
OMP_NUM_THREADS=32 ./matmul_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

// ----------------------------------------------------------------------------
// CUDA utils

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// error checking
void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// kernel 1: naive kernel, every thread handles one output element, direct global memory access
__global__ void matmul_forward_kernel1(float* out,
                                       float* inp, float* weight, float* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        int b = bt / BT;
        int t = bt % BT;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        float* wrow = weight + oc*C;
        float* inp_bt = inp + b * BT * C + t * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out[bt * OC + oc] = val;
    }
}

// is there no better way other than just adding bias with a whole separate kernel?
// this is a highly memory-bound operation, should be fused into the matmul kernel
// but i can't seem to find a cuBLAS function that does this
__global__ void add_bias(float* out, float* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B*T*OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     float* inp, float* weight, float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    dim3 gridDim(CEIL_DIV(B * T, sqrt_block_size), CEIL_DIV(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

// kernel 2 calls cuBLAS, which should be very efficient
void matmul_forward2(float* out,
                     float* inp, float* weight, float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    cublasHandle_t handle; // cuBLAS context
    cublasStatus_t stat = cublasCreate(&handle); // initialize CUBLAS context
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out^T. Mathematically, this is:
    // out^T = weight @ inp^T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out^T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed\n");
        exit(1);
    }
    // and now we still have to add the bias... (ew)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = CEIL_DIV(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
    cublasDestroy(handle);
}

// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------
// random utils

float* make_random_float(int N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
    matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, 32);

    float* out_gpu = (float*)malloc(B * T * OC * sizeof(float));
    cudaCheck(cudaMemcpy(out_gpu, d_out, B * T * OC * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B * T * OC; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", out[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (i >= 5 && fabs(out[i] - out_gpu[i]) > 1e-4) {
            printf("Mismatch at %d: %f vs %f\n", i, out[i], out_gpu[i]);
            exit(1);
        }
    }
    printf("Results match at block_size=256!\n");

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 10;
        cudaEvent_t start, stop;
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&stop));
        cudaCheck(cudaEventRecord(start, 0));
        for (int i = 0; i < repeat_times; i++) {
            matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        }
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float elapsed_time;
        cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 * repeat_times / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %f ms | tflops %f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    free(out_gpu);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}