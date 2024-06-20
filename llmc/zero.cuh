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

#endif // MULTI_GPU

// ----------------------------------------------------------------------------
// Parameters specific to training on multiple GPUs.
typedef struct {
    int process_rank;      // Rank of this process among all processes launched. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int device_idx;        // This process GPU index on current machine. 0 if no multi-GPU.

    // Zero Redundancy Optimizer stage - https://fairscale.readthedocs.io/en/stable/deep_dive/oss_sdp_fsdp.html
    int zero_stage;        // 0-Disabled, 1-OSS, 2-SDP, 3-FSDP
    size_t shard_num_parameters;
    size_t shard_offset;
#ifdef MULTI_GPU
    ncclComm_t nccl_comm;  // NCCL communication primitive, used for collective multi-GPU work.
    cudaStream_t nccl_stream;   // CUDA Stream to perform NCCL operations.
    cudaEvent_t compute_nccl_sync; // Event used to synchronize NCCL with the compute
#endif
} MultiGpuConfig;

MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node, char *dfs_path) {
#ifdef MULTI_GPU
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
    cudaCheck(cudaStreamCreate(&result.nccl_stream));
    // event without timing for maximum performance
    cudaCheck(cudaEventCreate(&result.compute_nccl_sync, cudaEventDisableTiming));
    nvtxNameCudaStreamA(result.nccl_stream, "nccl stream");
    nvtxNameCudaEventA(result.compute_nccl_sync, "nccl compute sync");
    return result;
#else
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    cudaCheck(cudaSetDevice(0));
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.device_idx = 0;
    return result;
#endif
}

void multi_gpu_config_free(MultiGpuConfig* multi_gpu_config) {
#ifdef MULTI_GPU
    ncclCheck(ncclCommDestroy(multi_gpu_config->nccl_comm));
    cudaCheck(cudaStreamDestroy(multi_gpu_config->nccl_stream));
    cudaCheck(cudaEventDestroy(multi_gpu_config->compute_nccl_sync));
#endif
}

void multi_gpu_barrier(const MultiGpuConfig* multi_gpu_config, float *unified_buffer) {
#ifdef MULTI_GPU
    if (multi_gpu_config->num_processes > 1) {
        if (unified_buffer == NULL) cudaCheck(cudaMallocManaged(&unified_buffer, sizeof(float)));
        ncclCheck(ncclAllReduce(unified_buffer, unified_buffer, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, 0));
    }
#endif
    cudaCheck(cudaDeviceSynchronize());
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

