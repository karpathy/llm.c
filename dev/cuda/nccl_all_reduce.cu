/*

A simple test of NCCL capabilities.
Fills a vector with 1s on the first GPU, 2s on the second, etc.
Then aggregates the values in the resulting vectors.

Compile example:
nvcc -lnccl -lcublas -lcublasLt nccl_all_reduce.cu -o nccl_all_reduce

Run on 2 local GPUs (set -np to a different value to change GPU count):
mpirun -np 2 bash -c './nccl_all_reduce $OMPI_COMM_WORLD_RANK'

*/

#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void nccl_check(ncclResult_t status, const char *file, int line) {
  if (status != ncclSuccess) {
    printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line,
           ncclGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

// Parameters specific to training on multiple GPUs.
typedef struct {
  int process_rank;      // Rank of this process among all processes on all hosts. 0 if no multi-GPU.
  int num_processes;     // Total number of processes on all hosts. 1 if no multi-GPU.
  int device_idx;        // This process GPU index on current machine. 0 if no multi-GPU.
  ncclComm_t nccl_comm;  // NCCL communication primitive, used for collective mutli-GPU work.
} MultiGpuConfig;

MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node, char *dfs_path) {
    MultiGpuConfig result;
    ncclUniqueId nccl_id;

    result.process_rank = process_rank;
    result.num_processes = num_processes;
    result.device_idx = process_rank % gpus_per_node;

    FILE* idFile;
    static char filename[256];
    snprintf(filename, sizeof(filename), "%s/ncclUniqueId.dat", dfs_path);

    if (result.process_rank == 0) { // Generate the NCCL unique ID at rank 0 and write it to a file
        ncclCheck(ncclGetUniqueId(&nccl_id));
        idFile = fopen(filename, "wb");
        assert(idFile != NULL);
        fwrite(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);            
    } else {                        // Other ranks wait until the file is available and read the unique ID
        do {
            usleep(1000000);
            idFile = fopen(filename, "rb");
            if (idFile != NULL) break;
        } while (idFile == NULL);
        fread(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);
    }

    printf("ProcessID:%d, NumProcess::%d, DeviceId:%d\n", result.process_rank, result.num_processes, result.device_idx);
    cudaCheck(cudaSetDevice(result.device_idx));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    return result;
}

void multi_gpu_config_free(const MultiGpuConfig* multi_gpu_config) {
    ncclCommDestroy(multi_gpu_config->nccl_comm);
}

float get_mean(float *arr, size_t size, int process_rank) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += arr[i];
  }
  return sum / size;
}

// CUDA kernel to set each element of the array to a specific value
__global__ void set_vector(float *array, float value, size_t num_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        array[idx] = value;
    }
}

int main(int argc, char *argv[]) {
  // Some constants
  const size_t all_reduce_buffer_size = 32 * 1024 * 1024;
  const size_t threads_per_block = 1024;

  MultiGpuConfig multi_gpu_config = multi_gpu_config_init(2, atoi(argv[1]), 8, ".");

  // Allocating buffers on each of the devices.
  float *all_reduce_buffer;
  cudaCheck(
      cudaMalloc(&all_reduce_buffer, all_reduce_buffer_size * sizeof(float)));

  int n_blocks = cdiv(all_reduce_buffer_size, threads_per_block);
  // Set the allocated memory to a defined value.
  set_vector<<<n_blocks, threads_per_block>>>(
      all_reduce_buffer, all_reduce_buffer_size,
      (float)(multi_gpu_config.process_rank + 1));
  cudaCheck(cudaGetLastError());

  float *all_reduce_buffer_host =
      (float *)malloc(all_reduce_buffer_size * sizeof(float));

  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  printf("[Process rank %d] average value before all reduce is %.6f\n", multi_gpu_config.process_rank,
         get_mean(all_reduce_buffer_host, all_reduce_buffer_size,
                  multi_gpu_config.process_rank));

  float *all_reduce_buffer_recv;
  cudaCheck(cudaMalloc(&all_reduce_buffer_recv,
                       all_reduce_buffer_size * sizeof(float)));

  ncclCheck(ncclAllReduce(
      (const void *)all_reduce_buffer, (void *)all_reduce_buffer_recv,
      all_reduce_buffer_size, ncclFloat, ncclSum, multi_gpu_config.nccl_comm, 0));


  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer_recv,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  float all_reduce_mean_value = get_mean(all_reduce_buffer_host, all_reduce_buffer_size, multi_gpu_config.process_rank);

  printf("[Process rank %d] average value after all reduce is %.6f\n", multi_gpu_config.process_rank, all_reduce_mean_value);

  float expected_all_reduce_mean_value = 0.0;
  for (int i = 0; i != multi_gpu_config.num_processes; ++i) {
    expected_all_reduce_mean_value += i + 1;
  }
  if (abs(expected_all_reduce_mean_value - all_reduce_mean_value) > 1e-5) {
    printf("[Process rank %d] ERROR: Unexpected all reduce value: %.8f, expected %.8f\n", multi_gpu_config.process_rank, all_reduce_mean_value, expected_all_reduce_mean_value);
  } else {
    printf("[Process rank %d] Checked against expected mean value. All good!\n", multi_gpu_config.process_rank);
  }

  free(all_reduce_buffer_host);
  cudaCheck(cudaFree(all_reduce_buffer));
  multi_gpu_config_free(&multi_gpu_config);
}
