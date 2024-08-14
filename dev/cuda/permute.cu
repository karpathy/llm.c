/*
Kernels to demonstrate permute operation.

Compile example:
nvcc -O3 permute.cu -o permute

The goal is to permute a 4D matrix from its original shape (dim1, dim2, dim3, dim4) to a new shape (dim4, dim3, dim1, dim2).

Before permutation, we need to understand how to access elements in a flattened (linear) form of the matrix.

Given:

dim1 = size of the 1st dimension
dim2 = size of the 2nd dimension
dim3 = size of the 3rd dimension
dim4 = size of the 4th dimension

For any element in a 4D matrix at position (i1, i2, i3, i4), where:

i1 is the index in dimension 1
i2 is the index in dimension 2
i3 is the index in dimension 3
i4 is the index in dimension 4

If you find it challenging to calculate the indices i1, i2, i3, and i4, observe the pattern in the index calculations.
Initially, it might take some time to grasp, but with practice, you'll develop a mental model for it.

To calculate the indices, use the following formulas:

i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
i2 = (idx / (dim3 * dim4)) % dim2;
i3 = (idx / dim4) % dim3;
i4 = idx % dim4;

Pattern Explanation:
To find the index for any dimension, divide the thread ID (idx) by the product of all subsequent dimensions.
Then, perform modulo operation with the current dimension.



The linear index in a flattened 1D array is calculated as:
linear_idx = i1 × ( dim2 × dim3 × dim4 ) + i2 × ( dim3 × dim4 ) + i3 × dim4 + i4
This linear index uniquely identifies the position of the element in the 1D array.

To permute the matrix, we need to rearrange the indices according to the new shape.
In this case, we are permuting from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2).

The new dimension post permutation will be as follows:

dim1 becomes the new 3rd dimension.
dim2 becomes the new 4th dimension.
dim3 becomes the new 2nd dimension.
dim4 becomes the new 1st dimension.

permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;

Here's how this works:

i4 * (dim3 * dim1 * dim2): This accounts for how many complete dim3 × dim1 × dim2 blocks fit before the current i4 block.
i3 * (dim1 * dim2): This accounts for the offset within the current i4 block, specifying which i3 block we are in.
i1 * dim2: This accounts for the offset within the current i3 block, specifying which i1 block we are in.
i2: This gives the offset within the current i1 block.

Lastly at the end we store the current value at idx index of the original value to the permuted index in the permuted_matrix.


--------------------------------------------------------------------------------------------------------------------------------------------------------

Similarly we can follow the above approach to permute matrices of any dimensions.

*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "common.h"

// CPU function to permute a 4D matrix
void permute_cpu(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int total_threads = dim1 * dim2 * dim3 * dim4;

    for (int idx = 0; idx < total_threads; idx++) {
        // Calculate the 4D indices from the linear index
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // Compute the new index for the permuted matrix
        // Transpose from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        out_matrix[permuted_idx] = matrix[idx];
    }
}

// CUDA kernel to permute a 4D matrix
__global__ void permute_kernel(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure index is within bounds
    if (idx < dim1 * dim2 * dim3 * dim4) {
        // Calculate the 4D indices from the linear index
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // Compute the new index for the permuted matrix
        // Transpose from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;
        out_matrix[permuted_idx] = matrix[idx];
    }
}


int main() {
    int dim_1 = 24;
    int dim_2 = 42;
    int dim_3 = 20;
    int dim_4 = 32;

    // Set up the device
    int deviceIdx = 0;
    cudaSetDevice(deviceIdx);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // Allocate host memory
    float* matrix = make_random_float(dim_1 * dim_2 * dim_3 * dim_4);
    float* permuted_matrix = (float*)malloc(dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // Initialize the matrix with random values

    // Allocate device memory
    float *d_matrix, *d_permuted_matrix;
    cudaMalloc(&d_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));
    cudaMalloc(&d_permuted_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform permutation on CPU
    clock_t start = clock();
    permute_cpu(matrix, permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    clock_t end = clock();
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // Define block and grid sizes
    dim3 blockSize(256);
    int totalThreads = dim_1 * dim_2 * dim_3 * dim_4;
    int gridSize = (totalThreads + blockSize.x - 1) / blockSize.x; // Compute grid size

    // Launch CUDA kernel to perform permutation
    permute_kernel<<<gridSize, blockSize>>>(d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Verify results
    printf("Checking correctness...\n");
    validate_result(d_permuted_matrix, permuted_matrix, "permuted_matrix", dim_1 * dim_2 * dim_3 * dim_4, 1e-5f);

    printf("All results match.\n\n");
    // benchmark kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, permute_kernel,
                                          d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4
    );
    printf("time gpu %.4f ms\n", elapsed_time);
    printf("time cpu %.4f ms\n", elapsed_time_cpu);

    // Free allocated memory
    free(matrix);
    free(permuted_matrix);
    cudaFree(d_matrix);
    cudaFree(d_permuted_matrix);

    return 0;
}
