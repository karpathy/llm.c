#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "tune.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MSTRINGIFY(...) #__VA_ARGS__

const char *build_options = "-cl-fast-relaxed-math -cl-mad-enable";

const char *KernelSource =
#include "utils.cl"
#include "matmul_forward.cl"
#include "matmul_backward.cl"
;

// structure to house opencl variables
typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_forward;
    cl_kernel matmul_backward1;
    cl_kernel matmul_backward2;
    cl_kernel matmul_backward3;
    cl_mem matmul_A;
    cl_mem matmul_B;
    cl_mem matmul_bias;
    cl_mem matmul_out;
    size_t matmul_tile_size;
    size_t kb3_preferred_wg_size;
} GPT2_CL;

enum {
    ERR_GET_PLATFORM = -1,
    ERR_INVALID_PLATFORM = -2,
    ERR_GET_PLATFORM_INFO = -3,
    ERR_GET_DEVICE = -4,
    ERR_INVALID_DEVICE = -5,
    ERR_GET_DEVICE_INFO = -6,
    ERR_CREATE_CONTEXT = -7,
    ERR_CREATE_QUEUE = -8,
    ERR_CREATE_PROGRAM = -9,
    ERR_BUILD_PROGRAM = -10,
    ERR_CREATE_KERNEL = -11,
    ERR_GET_KERNEL_INFO = -12,
    ERR_INVALID_COMBINATION = -13,
    ERR_CREATE_BUFFER = -14,
    ERR_SET_KERNEL_ARG = -15,
};

// set env CL_PLATFORM_IDX to select a platform
// set env CL_DEVICE_IDX to select a device
// returns non-zero number on error
int cl_init(GPT2_CL *gcl, int B, int T, int C, int V) {
    cl_platform_id platforms[5];
    char platform_name[1024];
    cl_uint num_platforms;
    int selected_platform = 0;
    cl_device_id devices[10];
    char device_name[1024];
    cl_uint num_devices;
    int selected_device = 0;
    char build_options_str[1024];
    int size;
    char *env;
    cl_int err;
    size_t max_kernel_wg_size;
    int matmul_tile_size = MATMUL_TILE_SIZE;
    int matmul_local_mem_padding_size = MATMUL_LOCAL_MEM_PADDING_SIZE;
    int matmul_vload_size = MATMUL_VLOAD_SIZE;
    int matmul_do_preload = MATMUL_DO_PRELOAD;
    int matmul_use_dot_product = MATMUL_USE_DOT_PRODUCT;

    // initialize all variables to NULL
    gcl->context = NULL;
    gcl->queue = NULL;
    gcl->program = NULL;
    gcl->matmul_forward = NULL;
    gcl->matmul_backward1 = NULL;
    gcl->matmul_backward2 = NULL;
    gcl->matmul_backward3 = NULL;
    gcl->matmul_A = NULL;
    gcl->matmul_B = NULL;
    gcl->matmul_bias = NULL;
    gcl->matmul_out = NULL;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    err |= clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl platform: %d\n", err);
        return ERR_GET_PLATFORM;
    }

    // choose cl platfrom from environment
    env = getenv("CL_PLATFORM_IDX");
    if (env != NULL) {
        selected_platform = atoi(env);
    }
    if(selected_platform >= num_platforms) {
        printf("invalid platform index %d\n", selected_platform);
        return ERR_INVALID_PLATFORM;
    }

    err = clGetPlatformInfo(platforms[selected_platform], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl platform info: %d\n", err);
        return ERR_GET_PLATFORM_INFO;
    }
    printf("using opencl platform: %s\n", platform_name);

    err = clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    err |= clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl device: %d\n", err);
        return ERR_GET_DEVICE;
    }

    // choose cl device from environment
    env = getenv("CL_DEVICE_IDX");
    if (env != NULL) {
        selected_device = atoi(env);
    }
    if(selected_device >= num_devices) {
        printf("invalid device index %d\n", selected_device);
        return ERR_INVALID_DEVICE;
    }

    gcl->device = devices[selected_device];
    err = clGetDeviceInfo(gcl->device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl device info: %d\n", err);
        return ERR_GET_DEVICE_INFO;
    }
    printf("using opencl device: %s\n", device_name);

    // create context
    gcl->context = clCreateContext(0, 1, &gcl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl context: %d\n", err);
        return ERR_CREATE_CONTEXT;
    }

    // create command queue
    gcl->queue = clCreateCommandQueue(gcl->context, gcl->device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl command queue: %d\n", err);
        return ERR_CREATE_QUEUE;
    }

    // create program
    gcl->program = clCreateProgramWithSource(gcl->context, 1, (const char**)&KernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl program: %d\n", err);
        return ERR_CREATE_PROGRAM;
    }

    // load tune parameters from env
    env = getenv("MATMUL_TILE_SIZE");
    if (env != NULL) {
        matmul_tile_size = atoi(env);
    }
    env = getenv("MATMUL_LOCAL_MEM_PADDING_SIZE");
    if (env != NULL) {
        matmul_local_mem_padding_size = atoi(env);
    }
    env = getenv("MATMUL_VLOAD_SIZE");
    if (env != NULL) {
        matmul_vload_size = atoi(env);
    }
    env = getenv("MATMUL_DO_PRELOAD");
    if (env != NULL) {
        matmul_do_preload = atoi(env);
    }
    env = getenv("MATMUL_USE_DOT_PRODUCT");
    if (env != NULL) {
        matmul_use_dot_product = atoi(env);
    }
    if(matmul_vload_size && (matmul_tile_size % matmul_vload_size) != 0) {
        printf("error: matmul_tile_size(%d) must be multiple of matmul_vload_size(%d)\n",
                    matmul_tile_size, matmul_vload_size);
        return ERR_INVALID_COMBINATION;
    }
    gcl->matmul_tile_size = matmul_tile_size;

    // build program
    sprintf(build_options_str, "%s -D TILE_SIZE=%d -D LOCAL_MEM_PADDING_SIZE=%d -D VLOAD_SIZE=%d -D DO_PRELOAD=%d -D USE_DOT_PRODUCT=%d",
        build_options, matmul_tile_size, matmul_local_mem_padding_size, matmul_vload_size, matmul_do_preload, matmul_use_dot_product);
    err = clBuildProgram(gcl->program, 1, &gcl->device, build_options_str, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t buf_len = 0;
        char *buffer = NULL;

        printf("error: Failed to build cl program\n");
        clGetProgramBuildInfo(gcl->program, gcl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &buf_len);

        buffer = (char*)malloc(buf_len);
        if(buffer) {
            clGetProgramBuildInfo(gcl->program, gcl->device, CL_PROGRAM_BUILD_LOG, buf_len, buffer, NULL);
            printf("%s\n", buffer);
            free(buffer);
        }
        return ERR_BUILD_PROGRAM;
    }

    // create kernel
    gcl->matmul_forward = clCreateKernel(gcl->program, "matmul_forward", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        return ERR_CREATE_KERNEL;
    }
    err = clGetKernelWorkGroupInfo(gcl->matmul_forward, gcl->device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(max_kernel_wg_size), &max_kernel_wg_size, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl kernel info: %d\n", err);
        return ERR_GET_KERNEL_INFO;
    }
    if((matmul_tile_size * matmul_tile_size) > max_kernel_wg_size) {
        printf("error: matmul_tile_size(%d) * matmul_tile_size(%d) > max_kernel_wg_size(%lu)\n",
                matmul_tile_size, matmul_tile_size, max_kernel_wg_size);
        return ERR_INVALID_COMBINATION;
    }

    gcl->matmul_backward1 = clCreateKernel(gcl->program, "matmul_backward1", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        return ERR_CREATE_KERNEL;
    }
    err = clGetKernelWorkGroupInfo(gcl->matmul_backward1, gcl->device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(max_kernel_wg_size), &max_kernel_wg_size, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl kernel info: %d\n", err);
        return ERR_GET_KERNEL_INFO;
    }
    if((matmul_tile_size * matmul_tile_size) > max_kernel_wg_size) {
        printf("error: matmul_tile_size(%d) * matmul_tile_size(%d) > max_kernel_wg_size(%lu)\n",
                matmul_tile_size, matmul_tile_size, max_kernel_wg_size);
        return ERR_INVALID_COMBINATION;
    }

    gcl->matmul_backward2 = clCreateKernel(gcl->program, "matmul_backward2", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        return ERR_CREATE_KERNEL;
    }
    err = clGetKernelWorkGroupInfo(gcl->matmul_backward2, gcl->device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(max_kernel_wg_size), &max_kernel_wg_size, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl kernel info: %d\n", err);
        return ERR_GET_KERNEL_INFO;
    }
    if((matmul_tile_size * matmul_tile_size) > max_kernel_wg_size) {
        printf("error: matmul_tile_size(%d) * matmul_tile_size(%d) > max_kernel_wg_size(%lu)\n",
                matmul_tile_size, matmul_tile_size, max_kernel_wg_size);
        return ERR_INVALID_COMBINATION;
    }

    gcl->matmul_backward3 = clCreateKernel(gcl->program, "matmul_backward3", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        return ERR_CREATE_KERNEL;
    }
    err = clGetKernelWorkGroupInfo(gcl->matmul_backward3, gcl->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(gcl->kb3_preferred_wg_size), &gcl->kb3_preferred_wg_size, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl kernel info: %d\n", err);
        return ERR_GET_KERNEL_INFO;
    }

    size = MAX(B * T * 4 * C, B * T * V);
    gcl->matmul_A = clCreateBuffer(gcl->context,  CL_MEM_READ_ONLY,  sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        return ERR_CREATE_BUFFER;
    }

    size = MAX(4 * C * C, V * C);
    size = MAX(size, B * T * 4 * C);
    gcl->matmul_B = clCreateBuffer(gcl->context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        return ERR_CREATE_BUFFER;
    }

    size = MAX(4 * C, V);
    gcl->matmul_bias = clCreateBuffer(gcl->context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        return ERR_CREATE_BUFFER;
    }

    size = MAX(B * T * 4 * C, B * T * V);
    size = MAX(size, 4 * C * C);
    size = MAX(size, V * C);
    gcl->matmul_out = clCreateBuffer(gcl->context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        return ERR_CREATE_BUFFER;
    }

    err = 0;
    err  = clSetKernelArg(gcl->matmul_forward, 0, sizeof(cl_mem), &gcl->matmul_out);
    err |= clSetKernelArg(gcl->matmul_forward, 1, sizeof(cl_mem), &gcl->matmul_A);
    err |= clSetKernelArg(gcl->matmul_forward, 2, sizeof(cl_mem), &gcl->matmul_B);
    err |= clSetKernelArg(gcl->matmul_forward, 3, sizeof(cl_mem), &gcl->matmul_bias);
    err |= clSetKernelArg(gcl->matmul_backward1, 0, sizeof(cl_mem), &gcl->matmul_out);
    err |= clSetKernelArg(gcl->matmul_backward1, 1, sizeof(cl_mem), &gcl->matmul_A);
    err |= clSetKernelArg(gcl->matmul_backward1, 2, sizeof(cl_mem), &gcl->matmul_B);
    err |= clSetKernelArg(gcl->matmul_backward2, 0, sizeof(cl_mem), &gcl->matmul_out);
    err |= clSetKernelArg(gcl->matmul_backward2, 1, sizeof(cl_mem), &gcl->matmul_A);
    err |= clSetKernelArg(gcl->matmul_backward2, 2, sizeof(cl_mem), &gcl->matmul_B);
    err |= clSetKernelArg(gcl->matmul_backward3, 0, sizeof(cl_mem), &gcl->matmul_A);
    err |= clSetKernelArg(gcl->matmul_backward3, 1, sizeof(cl_mem), &gcl->matmul_bias);
    if (err != CL_SUCCESS)
    {
        printf("error: Failed to set kernel arguments! %d\n", err);
        return ERR_SET_KERNEL_ARG;
    }

    return 0;
}

void cl_deinit(GPT2_CL *gcl) {
    if(gcl->matmul_A) clReleaseMemObject(gcl->matmul_A);
    if(gcl->matmul_B) clReleaseMemObject(gcl->matmul_B);
    if(gcl->matmul_bias) clReleaseMemObject(gcl->matmul_bias);
    if(gcl->matmul_out) clReleaseMemObject(gcl->matmul_out);
    if(gcl->matmul_forward) clReleaseKernel(gcl->matmul_forward);
    if(gcl->matmul_backward1) clReleaseKernel(gcl->matmul_backward1);
    if(gcl->matmul_backward2) clReleaseKernel(gcl->matmul_backward2);
    if(gcl->matmul_backward3) clReleaseKernel(gcl->matmul_backward3);
    if(gcl->program) clReleaseProgram(gcl->program);
    if(gcl->queue) clReleaseCommandQueue(gcl->queue);
    if(gcl->context) clReleaseContext(gcl->context);
}
