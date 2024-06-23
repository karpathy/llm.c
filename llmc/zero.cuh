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

MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank) {
#ifdef MULTI_GPU
    MultiGpuConfig result;
    result.process_rank = process_rank;
    result.num_processes = num_processes;
    result.local_device_idx = process_rank % 8;
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclUniqueId nccl_id;

    FILE* idFile;
    char dfs_path[256] = "/ephemeral/data/tokenizers";
    static char filename[256];
    snprintf(filename, sizeof(filename), "%s/ncclUniqueId.dat", dfs_path);

    if (result.process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
        idFile = fopen(filename, "wb");
        assert(idFile != NULL);
        fwrite(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);
        // Construct the scp command
        char command[1024];
        snprintf(command, sizeof(command), "scp %s ubuntu@h100-node-1-1:%s", filename, filename);
        printf("Executing command: %s\n", command);
        // Execute the scp command
        int ret = system(command);
        if (ret != 0) {
            fprintf(stderr, "scp command failed with error code %d\n", ret);
            exit(EXIT_FAILURE);
        }
    } else {                        // Other ranks wait until the file is available and read the unique ID
        do {
            printf("%d: Waiting for the file to be available\n", result.process_rank);
            usleep(1000000);
            idFile = fopen(filename, "rb");
            if (idFile != NULL) break;
        } while (idFile == NULL);
        freadCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);
    }
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
    cudaCheck(cudaFree(multi_gpu_config->unified_buffer));
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

