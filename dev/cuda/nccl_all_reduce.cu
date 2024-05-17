/*

A simple test of NCCL capabilities.
Fills a vector with 1s on the first GPU, 2s on the second, etc.
Then aggregates the values in the resulting vectors.

Compile example:
nvcc -lmpi -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ -lcublas -lcublasLt nccl_all_reduce.cu -o nccl_all_reduce

Run on 2 local GPUs (set -np to a different value to change GPU count):
mpirun -np 2 ./nccl_all_reduce

*/

#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
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

void mpi_check(int status, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    char mpi_error[4096];
    int mpi_error_len = 0;
    assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) ==
           MPI_SUCCESS);
    printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len,
           mpi_error);
    exit(EXIT_FAILURE);
  }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

// Sets a vector to a predefined value
__global__ void set_vector(float *data, int N, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check for out-of-bounds access
  if (i < N) {
    data[i] = value;
  }
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

// Parameters specific to training on multiple GPUs.
typedef struct {
  int process_rank;      // Rank of this process among all MPI processes on all hosts. 0 if no multi-GPU.
  int num_processes;     // Total number of processes on all hosts. 1 if no multi-GPU.
  int local_device_idx;  // This process GPU index on current machine. 0 if no multi-GPU.
  ncclComm_t nccl_comm;  // NCCL communication primitive, used for collective mutli-GPU work.
} MultiGpuConfig;

// Determine which GPU this process should use.
// Processes on the same machines use different GPU indicies. Processes on other machines don't.
// Copied from NCCL examples: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread
int multi_gpu_get_local_device_idx(int process_rank, int num_processes) {
  char hostname[1024];
  hostname[1023] = '\0';
  // All processes on the same machine will share the same hostname.
  gethostname(hostname, 1023);
  for (int i=0; i < 1024; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        break;
    }
  }
  uint64_t hostname_hash = 5381;
  for (int c = 0; hostname[c] != '\0'; c++){ hostname_hash = ((hostname_hash << 5) + hostname_hash) ^ hostname[c]; }

  // Distribute all hostname hashes to all processes.
  uint64_t* all_hostsname_hashes = (uint64_t*)malloc(num_processes * sizeof(uint64_t));
  all_hostsname_hashes[process_rank] = hostname_hash;
  mpiCheck(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_hostsname_hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

  // Identify which GPU we need to use.
  int local_device_idx = 0;
  for (int current_process = 0; current_process < num_processes; ++current_process) {
     if (current_process == process_rank) {
      // Found my gpu, local_device_idx now has my target GPU index.
      break;
     }
     if (all_hostsname_hashes[current_process] == all_hostsname_hashes[process_rank]) {
      // This process ID runs on the same machine, but it's not me, skip this GPU
      local_device_idx++;
     }
  }

  free(all_hostsname_hashes);
  return local_device_idx;
}

MultiGpuConfig multi_gpu_config_init(int *argc, char ***argv) {
    // Initialize MPI.
    MultiGpuConfig result;
    mpiCheck(MPI_Init(argc, argv));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
    result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
    printf("[Process rank %d] Using GPU %d\n", result.process_rank, result.local_device_idx);
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclUniqueId nccl_id;
    if (result.process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }
    mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    return result;
}

void multi_gpu_config_free(const MultiGpuConfig* multi_gpu_config) {
    ncclCommDestroy(multi_gpu_config->nccl_comm);
    mpiCheck(MPI_Finalize());
}

float get_mean(float *arr, size_t size, int process_rank) {
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += arr[i];
  }
  return sum / size;
}

int main(int argc, char **argv) {
  // Some constants
  const size_t all_reduce_buffer_size = 32 * 1024 * 1024;
  const size_t threads_per_block = 1024;

  MultiGpuConfig multi_gpu_config = multi_gpu_config_init(&argc, &argv);

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
