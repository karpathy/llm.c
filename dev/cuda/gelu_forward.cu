/*
Kernels for gelu forward pass.

Compile example:
nvcc -O3 --use_fast_math gelu_forward.cu -o gelu_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_forward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// OK, so this part requires a bit of explanation. Our end goal here is that we get a convenient interface that
// allows us to interact with vectorized loads that read 128 bits of aligned memory in a uniform way, independent
// of the underlying datatype, and the whims of the nvcc compiler.

// as a first step, we define some constants that indicate which type of memory operation we intend to do.
// these correspond to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
enum class ELoadMode {
    CA, CG, CS, LU, CV
};

enum class EStoreMode {
    WB, CG, CS, WT
};


// in order to enable dispatch _at compile time_, we need to encode the load/store mode into types (unless we want
// to have ugly template syntax at the call sites, like load<ELoadMode::CG>(address).
// Therefore, we wrap all these into compile-time integers, and provide global objects that can be passed to
// select the correct function overload.
template<ELoadMode V>
using load_mode_t = std::integral_constant<ELoadMode, V>;

template<EStoreMode V>
using store_mode_t = std::integral_constant<EStoreMode, V>;

constexpr load_mode_t<ELoadMode::CA> LdCA;
constexpr load_mode_t<ELoadMode::CG> LdCG;
constexpr load_mode_t<ELoadMode::CS> LdCS;
constexpr load_mode_t<ELoadMode::LU> LdLU;
constexpr load_mode_t<ELoadMode::CV> LdCV;

constexpr store_mode_t<EStoreMode::WB> StWB;
constexpr store_mode_t<EStoreMode::CG> StCG;
constexpr store_mode_t<EStoreMode::CS> StCS;
constexpr store_mode_t<EStoreMode::WT> StWT;

// Finally, we define the dispatch mechanism itself. Really just a long if-else chain, except
// that all of this needs to be decided at compile time (hence constexpr if)
template<ELoadMode Mode, class T>
__device__ T generic_load(const T* address, load_mode_t<Mode>) {
    if constexpr (Mode == ELoadMode::CA) {
        return __ldca(address);
    } else if constexpr (Mode == ELoadMode::CG) {
        return __ldcg(address);
    } else if constexpr (Mode == ELoadMode::CS) {
        return __ldcs(address);
    } else if constexpr (Mode == ELoadMode::LU) {
        return __ldlu(address);
    } else if constexpr (Mode == ELoadMode::CV) {
        return __ldcv(address);
    } else {
        __builtin_unreachable();
    }
}

template<EStoreMode Mode, class T>
__device__ void generic_store(T* address, const T& value, store_mode_t<Mode>) {
    if constexpr (Mode == EStoreMode::WB) {
        return __stwb(address, value);
    } else if constexpr (Mode == EStoreMode::CG) {
        return __stcg(address, value);
    } else if constexpr (Mode == EStoreMode::CS) {
        return __stcs(address, value);
    } else if constexpr (Mode == EStoreMode::WT) {
        return __stwt(address, value);
    }  else {
        __builtin_unreachable();
    }
}

// Finally, we define a wrapper type that contains 128 bits of whatever underlying type we want
// we store the actual data in an int4 vector, and reinterpret its bits, because int4 gets nvcc to
// reliably produce 128-bit instructions
// TODO do we really need this here, or can we get away with the int4 trick just inside the  load/store functions
// we allow individual element access with [], and provide a convenience accessor to get the data converted to
// a regular float for mixed-precision operations.
template<class ElementType>
struct alignas(16) Packed128 {
    __device__ ElementType& operator[](int index) {
        return reinterpret_cast<ElementType*>(&payload)[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return reinterpret_cast<const ElementType*>(&payload)[index];
    }
    __device__ float fp32(int index) {
        return static_cast<float>(reinterpret_cast<ElementType*>(&payload)[index]);
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);

    int4 payload;
};

// use this function to load a Packet128 from an aligned memory address
template<class ElementType, ELoadMode Mode=ELoadMode::CA>
__device__ Packed128<std::remove_const_t<ElementType>> load_aligned(ElementType* address, load_mode_t<Mode> mode = {}) {
    return {generic_load(reinterpret_cast<const int4*>(address), mode)};
}

// use this function to store a Packet128 to an aligned memory address
template<class ElementType, EStoreMode Mode=EStoreMode::WB>
__device__ void store_aligned(ElementType* target, Packed128<ElementType> value, store_mode_t<Mode> mode = {}) {
    generic_store(reinterpret_cast<int4*>(target), value.payload, mode);
}

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// elementwise ops are nice and ez
__global__ void gelu_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

// elementwise ops are nice and ez
__global__ void gelu_kernel2(float* out, const float* inp, int N) {
    using packet_t = Packed128<float>;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * packet_t::size;
    if (i < N) {
        packet_t packet_out;
        packet_t packet_in = load_aligned(inp + i, LdCS);
        for(int k = 0; k < packet_in.size; ++k) {
            float xi = packet_in[k];
            float cube = 0.044715f * xi * xi * xi;
            packet_out[k] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
        }
        store_aligned(out + i, packet_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(float* out, const float* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward2(float* out, const float* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, 4 * block_size);
    gelu_kernel2<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_forward(int kernel_num,
                  float* out,
                  const float* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
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

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // move to GPU
    float* d_out;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);


    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_size);
        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));

    return 0;
}