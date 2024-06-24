/*
Utilities for ZeRO sharding
*/

#ifndef LLMC_ZERO_CUH
#define LLMC_ZERO_CUH

#include <arpa/inet.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#ifdef MULTI_GPU
#include <nccl.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
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

#ifdef USE_MPI
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
#endif

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
void send_nccl_id_to_clients(ncclUniqueId *nccl_id, int client_sockets[], int num_clients) {
    for (int i = 0; i < num_clients; ++i) {
        if (send(client_sockets[i], nccl_id, sizeof(*nccl_id), 0) == -1) {
            printf("Failed to send nccl_id");
            exit(EXIT_FAILURE);
        }
        close(client_sockets[i]);
    }
}

ncclUniqueId get_nccl_id_via_tcp(MultiGpuConfig* result, const char* server_ip) {
    ncclUniqueId nccl_id;
    // hardcode an arbitrary port number between 1024 and 49151 (registered ports)
    int SERVER_PORT = 12345;
    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));

        int MAX_CLIENTS = result->num_processes - 1;
        int client_sockets[MAX_CLIENTS];
        int num_clients = 0;
        int server_socket, new_socket;
        struct sockaddr_in address;
        int addrlen = sizeof(address);
        int opt = 1;

        // Create a TCP socket
        if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket failed");
            exit(EXIT_FAILURE);
        }

        // set socket options
        // SOL_SOCKET - means that option is configured at socket level
        // SO_REUSEADDR - allows to bind to an address which is in a TIME_WAIT state (already used by another socket) - useful when restarting the server
        // SO_REUSEPORT - allows to bind to the same port multiple times
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            printf("Setsockopt failed");
            exit(EXIT_FAILURE);
        }

        address.sin_family = AF_INET;  // IPv4
        address.sin_addr.s_addr = inet_addr(server_ip); // alternatively use INADDR_ANY to bind to all interfaces, currently we only allow ethernet
        address.sin_port = htons(SERVER_PORT);

        // Bind the socket to the address and port
        if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
            printf("Bind failed");
            exit(EXIT_FAILURE);
        }

        // MAX_CLIENTS specifies the maximum number of clients that can be queued for this server
        if (listen(server_socket, MAX_CLIENTS) < 0) {
            printf("Listen failed");
            exit(EXIT_FAILURE);
        }

        printf("Waiting for clients to connect...\n");
        while (num_clients < MAX_CLIENTS) {
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
                printf("Accept failed");
                exit(EXIT_FAILURE);
            }
            client_sockets[num_clients++] = new_socket;
            printf("Client %d connected\n", num_clients);
        }

        send_nccl_id_to_clients(&nccl_id, client_sockets, num_clients);
        printf("NCCL ID sent to all clients\n");

        close(server_socket);
    } else {
        int num_attempts = 5;
        int time_to_sleep = 2;

        int client_socket;
        struct sockaddr_in serv_addr;

        // Create a TCP socket
        if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket creation error");
            exit(EXIT_FAILURE);
        }

        // Set the server address and port
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address or address not supported");
            exit(EXIT_FAILURE);
        }

        // Try to connect to the server - retry if connection fails
        while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            printf("%d Connection failed, retrying in %d seconds\n", result->process_rank, time_to_sleep);
            if (--num_attempts == 0) {
                printf("Failed to connect to the server\n");
                exit(EXIT_FAILURE);
            }
            sleep(time_to_sleep);
        }

        if (recv(client_socket, &nccl_id, sizeof(nccl_id), 0) <= 0) {
            printf("Failed to receive nccl_id");
            exit(EXIT_FAILURE);
        }

        printf("Received NCCL ID\n");
        close(client_socket);
    }

    return nccl_id;
}

ncclUniqueId get_nccl_id_via_fs(MultiGpuConfig* result, char* fs_path) {
    // Works assuming that the filesystem is shared among all processes
    ncclUniqueId nccl_id;
    FILE* idFile;
    static char filename[1024];
    snprintf(filename, sizeof(filename), "%s/ncclUniqueId.sync", fs_path);

    if (result->process_rank != 0) {
        // Wait for the server to write the file
        // This is a naive way to synchronize the processes
        sleep(2);
    }

    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
        idFile = fopen(filename, "wb");
        assert(idFile != NULL);
        fwrite(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);
    } else {
        // Other ranks wait until the file is available and read the unique ID
        do {
            sleep(1);  // 1 second
            idFile = fopen(filename, "rb");
            if (idFile != NULL) break;
        } while (idFile == NULL);
        freadCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fclose(idFile);
    }

    return nccl_id;
}

#ifdef USE_MPI
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

#endif

MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, char* server_ip, char* fs_path, char* init_method) {
#ifdef MULTI_GPU
    MultiGpuConfig result;
    ncclUniqueId nccl_id;
    if (strcmp(init_method, "mpi") != 0) {
        result.process_rank = process_rank;
        result.num_processes = num_processes;
        result.local_device_idx = process_rank % 8;
        if (strcmp(init_method, "tcp") == 0) {
            nccl_id = get_nccl_id_via_tcp(&result, server_ip);
        } else if (strcmp(init_method, "fs") == 0) {
            nccl_id = get_nccl_id_via_fs(&result, fs_path);
        } else {
            printf("Invalid init method\n");
            exit(EXIT_FAILURE);
        }
    } else {
        #ifdef USE_MPI
        mpiCheck(MPI_Init(NULL, NULL));
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
        result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
        if (result.process_rank == 0) {
            ncclCheck(ncclGetUniqueId(&nccl_id));
        }
        mpiCheck(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
        #else
        printf("MPI support is disabled. Please enable MPI support to use MPI init method.\n");
        exit(EXIT_FAILURE);
        #endif
    }
    cudaCheck(cudaSetDevice(result.local_device_idx));
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
    #ifdef USE_MPI
    mpiCheck(MPI_Finalize());
    #endif
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

