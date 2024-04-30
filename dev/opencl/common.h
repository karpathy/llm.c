#include <stdio.h>
#include <stdlib.h>

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
    size_t max_wg_size;
} GPT2_CL;

// set env CL_PLATFORM_IDX to select a platform
// set env CL_DEVICE_IDX to select a device
void cl_init(GPT2_CL *gcl, int B, int T, int C, int V) {
    cl_platform_id platforms[5];
    char platform_name[1024];
    cl_uint num_platforms;
    int selected_platform = 0;
    cl_device_id devices[5];
    char device_name[1024];
    cl_uint num_devices;
    int selected_device = 0;
    char build_options_str[1024];
    int size;
    char *env;
    cl_int err;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    err |= clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl platform: %d\n", err);
        exit(1);
    }

    // choose cl platfrom from environment
    env = getenv("CL_PLATFORM_IDX");
    if (env != NULL) {
        selected_platform = atoi(env);
    }
    if(selected_platform >= num_platforms) {
        printf("invalid platform index %d\n", selected_platform);
        exit(1);
    }

    err = clGetPlatformInfo(platforms[selected_platform], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl platform name: %d\n", err);
        exit(1);
    }
    printf("using opencl platform: %s\n", platform_name);

    err = clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    err |= clGetDeviceIDs(platforms[selected_platform], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl device: %d\n", err);
        exit(1);
    }

    // choose cl device from environment
    env = getenv("CL_DEVICE_IDX");
    if (env != NULL) {
        selected_device = atoi(env);
    }
    if(selected_device >= num_devices) {
        printf("invalid device index %d\n", selected_device);
        exit(1);
    }

    gcl->device = devices[selected_device];
    err = clGetDeviceInfo(gcl->device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if (err != CL_SUCCESS) {
        printf("error getting opencl device name: %d\n", err);
        exit(1);
    }
    printf("using opencl device: %s\n", device_name);

    // create context
    gcl->context = clCreateContext(0, 1, &gcl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl context: %d\n", err);
        exit(1);
    }

    // create command queue
    gcl->queue = clCreateCommandQueue(gcl->context, gcl->device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl command queue: %d\n", err);
        exit(1);
    }

    // create program
    gcl->program = clCreateProgramWithSource(gcl->context, 1, (const char**)&KernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl program: %d\n", err);
        exit(1);
    }

    // build program
    sprintf(build_options_str, "%s -D TILE_SIZE=%d -D LOCAL_MEM_PADDING_SIZE=%d -D MATMUL_VLOAD_SIZE=%d",
        build_options, MATMUL_TILE_SIZE, MATMUL_LOCAL_MEM_PADDING_SIZE, MATMUL_VLOAD_SIZE);
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
        exit(1);
    }

    // create kernel
    gcl->matmul_forward = clCreateKernel(gcl->program, "matmul_forward", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        exit(1);
    }

    gcl->matmul_backward1 = clCreateKernel(gcl->program, "matmul_backward1", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        exit(1);
    }

    gcl->matmul_backward2 = clCreateKernel(gcl->program, "matmul_backward2", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        exit(1);
    }

    gcl->matmul_backward3 = clCreateKernel(gcl->program, "matmul_backward3", &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl kernel: %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(gcl->matmul_forward, gcl->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(gcl->max_wg_size), &gcl->max_wg_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    size = MAX(B * T * 4 * C, B * T * V);
    gcl->matmul_A = clCreateBuffer(gcl->context,  CL_MEM_READ_ONLY,  sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        exit(1);
    }

    size = MAX(4 * C * C, V * C);
    size = MAX(size, B * T * 4 * C);
    gcl->matmul_B = clCreateBuffer(gcl->context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        exit(1);
    }

    size = MAX(4 * C, V);
    gcl->matmul_bias = clCreateBuffer(gcl->context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        exit(1);
    }

    size = MAX(B * T * 4 * C, B * T * V);
    size = MAX(size, 4 * C * C);
    size = MAX(size, V * C);
    gcl->matmul_out = clCreateBuffer(gcl->context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("error creating opencl buffer: %d\n", err);
        exit(1);
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
        exit(1);
    }
}

void cl_deinit(GPT2_CL *gcl) {
    clReleaseMemObject(gcl->matmul_A);
    clReleaseMemObject(gcl->matmul_B);
    clReleaseMemObject(gcl->matmul_bias);
    clReleaseMemObject(gcl->matmul_out);
    clReleaseKernel(gcl->matmul_forward);
    clReleaseKernel(gcl->matmul_backward1);
    clReleaseKernel(gcl->matmul_backward2);
    clReleaseKernel(gcl->matmul_backward3);
    clReleaseProgram(gcl->program);
    clReleaseCommandQueue(gcl->queue);
    clReleaseContext(gcl->context);
}
