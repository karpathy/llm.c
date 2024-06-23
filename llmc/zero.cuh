/*
Utilities for ZeRO sharding
*/

#ifndef LLMC_ZERO_CUH
#define LLMC_ZERO_CUH

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#ifdef MULTI_GPU
#include <mpi.h>
#include <nccl.h>
#endif

// ----------------------------------------------------------------------------
// Multi-GPU related
#ifdef MULTI_GPU

#if defined(ENABLE_FP32)
const ncclDataType_t ncclFloatX = ncclFloat;
#elif defined(ENABLE_FP16)
const ncclDataType_t ncclFloatX = ncclHalf;
#else // Default to bfloat16
const ncclDataType_t ncclFloatX = ncclBfloat16;
#endif

void nccl_check(ncclResult_t status, const char *file, int line) {
    if (status != ncclSuccess) {
        printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line, ncclGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS);
        printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len, mpi_error);
        exit(EXIT_FAILURE);
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

#endif // MULTI_GPU

// ----------------------------------------------------------------------------
// Parameters specific to training on multiple GPUs.
typedef struct {
    int process_rank;      // Rank of this process among all processes. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int local_device_idx;  // This process GPU index on current machine. 0 if no multi-GPU.

    // Zero Redundancy Optimizer stage - https://fairscale.readthedocs.io/en/stable/deep_dive/oss_sdp_fsdp.html
    // 0-Disabled
    // 1-Optimizer State Sharding (OSS)
    // 2-Optimizer + Gradient State Sharding (SDP)
    // 3-Optimizer + Gradient + Horizontal Model Sharding (FSDP)
    int zero_stage;
    size_t shard_num_parameters;
#ifdef MULTI_GPU
    ncclComm_t nccl_comm;       // NCCL communication primitive, used for collective multi-GPU work.
    cudaStream_t nccl_stream;   // CUDA Stream to perform NCCL operations.
    cudaEvent_t compute_nccl_sync; // Event used to synchronize NCCL with the compute
    float* unified_buffer;
#endif
} MultiGpuConfig;

#ifdef MULTI_GPU
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
    uint64_t hostname_hash = 5381u;
    for (int c = 0; hostname[c] != '\0'; c++){ hostname_hash = ((hostname_hash << 5u) + hostname_hash) ^ hostname[c]; }

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
#endif

MultiGpuConfig multi_gpu_config_init(int *argc, char ***argv) {
#ifdef MULTI_GPU
    // Initialize MPI.
    MultiGpuConfig result;
    mpiCheck(MPI_Init(argc, argv));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
    result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclUniqueId nccl_id;
    if (result.process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }
    mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    cudaCheck(cudaStreamCreate(&result.nccl_stream));
    // event without timing for maximum performance
    cudaCheck(cudaEventCreate(&result.compute_nccl_sync, cudaEventDisableTiming));
    nvtxNameCudaStreamA(result.nccl_stream, "nccl stream");
    nvtxNameCudaEventA(result.compute_nccl_sync, "nccl compute sync");
    cudaCheck(cudaMallocManaged(&result.unified_buffer, sizeof(float)));
    return result;
#else
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    cudaCheck(cudaSetDevice(0));
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    return result;
#endif
}

void multi_gpu_config_free(MultiGpuConfig* multi_gpu_config) {
#ifdef MULTI_GPU
    ncclCheck(ncclCommDestroy(multi_gpu_config->nccl_comm));
    cudaCheck(cudaStreamDestroy(multi_gpu_config->nccl_stream));
    cudaCheck(cudaEventDestroy(multi_gpu_config->compute_nccl_sync));
    mpiCheck(MPI_Finalize());
#endif
}

void multi_gpu_barrier(const MultiGpuConfig* multi_gpu_config) {
#ifdef MULTI_GPU
    if (multi_gpu_config->num_processes > 1) {
        ncclCheck(ncclAllReduce(multi_gpu_config->unified_buffer, multi_gpu_config->unified_buffer, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream));
    }
    cudaCheck(cudaDeviceSynchronize());
#endif
}

// Offset and size of a tensor shard
typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;

// Get info about sharding for a tensor of elements many numbers
ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* multi_gpu_config, int shard_at_stage) {
    const int nproc = multi_gpu_config->num_processes;
    if(multi_gpu_config->zero_stage >= shard_at_stage) {
        if (elements % nproc != 0) {
            fprintf(stderr, "Number of elements %zu must be a multiple of the number of processes %d\n", elements, nproc);
            exit(EXIT_FAILURE);
        }
        return {(ptrdiff_t) (multi_gpu_config->process_rank * (elements / nproc)), elements / nproc};
    } else {
        return {0, elements};
    }
}

// Block NCCL stream until computations on compute_stream are done, then aggregate multiple pointers in an NCCL group.
// This can work either as an all-reduce (i.e., no ZeRo), or a reduce-scatter (ZeRO 1).
// The awkward `(&pointers)[N]` syntax ensures we are capturing the parameters as sized arrays, so that it becomes impossible
// to call this function if pointers and pointers_sizes do not match.
template<int N>
void multi_gpu_async_reduce_gradient(
    floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
    MultiGpuConfig* multi_gpu_config, cudaStream_t compute_stream) {
    if (multi_gpu_config->num_processes == 1) {
        return; // no multi-GPU, just exit.
    }

#ifdef MULTI_GPU
    NVTX_RANGE_FN();
    // mark an event on the compute stream, and immediately wait on this in the nccl stream
    // this means that the nccl stream won't start executing before all compute kernels that
    // have been submitted before this point have finished.
    // by using an event instead of cudaSyncStream, we avoid having to synchronize the host, and
    // can enqueue new work to the GPU right away.
    cudaCheck(cudaEventRecord(multi_gpu_config->compute_nccl_sync, compute_stream));
    cudaCheck(cudaStreamWaitEvent(multi_gpu_config->nccl_stream, multi_gpu_config->compute_nccl_sync));
    ncclCheck(ncclGroupStart()); // NCCL group: aggregate all pointers in a single NCCL GPU kernel.
    for (int i = 0; i < N; ++i) {
        if(multi_gpu_config->zero_stage == 0) {
            ncclCheck(ncclAllReduce(
                pointers[i], pointers[i],
                pointers_sizes[i],
                ncclFloatX, ncclAvg,
                multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream
            ));
        } else if(multi_gpu_config->zero_stage == 1) {
            assert(pointers_sizes[i] % multi_gpu_config->num_processes == 0);
            size_t shard_size = pointers_sizes[i] / multi_gpu_config->num_processes;
            ptrdiff_t shard_offset = (ptrdiff_t)shard_size * multi_gpu_config->process_rank;
            ncclCheck(ncclReduceScatter(
                pointers[i], pointers[i] + shard_offset,
                shard_size,
                ncclFloatX, ncclAvg,
                multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream
            ));
        }
    }
    ncclCheck(ncclGroupEnd());
#endif
}

#endif

