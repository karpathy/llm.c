#include <stdio.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MSTRINGIFY(...) #__VA_ARGS__

const char *build_options = "-Werror -cl-fast-relaxed-math -cl-mad-enable";

const char *KernelSource =
#include "matmul_forward.cl"
;

// structure to house opencl variables
typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel matmul_forward;
    cl_mem matmul_inp;
    cl_mem matmul_out;
    cl_mem matmul_weight;
    cl_mem matmul_bias;
    size_t size_global;
    size_t size_local;
} GPT2_CL;

void cl_matmul_forward(GPT2_CL *gcl, float* out,
                    float* inp, float* weight,
                    int B, int T, int C, int OC) {
    // inp is (B,T,C), weight is (OC, C)
    // out will be (B,T,OC)
    cl_int err = 0;

    err = clEnqueueWriteBuffer(gcl->queue, gcl->matmul_inp, CL_TRUE, 0, sizeof(float) * B * T * C, inp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_weight, CL_TRUE, 0, sizeof(float) * OC * C, weight, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    err = clSetKernelArg(gcl->matmul_forward, 3, sizeof(int), &B);
    err |= clSetKernelArg(gcl->matmul_forward, 4, sizeof(int), &T);
    err |= clSetKernelArg(gcl->matmul_forward, 5, sizeof(int), &C);
    err |= clSetKernelArg(gcl->matmul_forward, 6, sizeof(int), &OC);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    gcl->size_global = B * T * OC;
    err = clEnqueueNDRangeKernel(gcl->queue, gcl->matmul_forward, 1, NULL, &gcl->size_global, &gcl->size_local, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel!\n");
        exit(1);
    }
    clFinish(gcl->queue);

    err = clEnqueueReadBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * B * T * OC, out, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
}

void cl_matmul_forward_bias(GPT2_CL *gcl, float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = bias[o];
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void cl_init(GPT2_CL *gcl, int B, int T, int C, int V) {
    cl_int err;
    int size;

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &gcl->device, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting OpenCL device: %d\n", err);
        exit(1);
    }

    // create context
    gcl->context = clCreateContext(0, 1, &gcl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL context: %d\n", err);
        exit(1);
    }

    // create command queue
    gcl->queue = clCreateCommandQueue(gcl->context, gcl->device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL command queue: %d\n", err);
        exit(1);
    }

    // create program
    gcl->program = clCreateProgramWithSource(gcl->context, 1, (const char**)&KernelSource, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL program: %d\n", err);
        exit(1);
    }

    // build program
    err = clBuildProgram(gcl->program, 1, &gcl->device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build cl program\n");
        clGetProgramBuildInfo(gcl->program, gcl->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // create kernel
    gcl->matmul_forward = clCreateKernel(gcl->program, "matmul_forward", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL kernel: %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(gcl->matmul_forward, gcl->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(gcl->size_local), &gcl->size_local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    gcl->matmul_inp = clCreateBuffer(gcl->context,  CL_MEM_READ_ONLY,  sizeof(float) * B * T * 4 * C, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL buffer: %d\n", err);
        exit(1);
    }

    size = MAX(B * T * 4 * C, B * T * V);
    gcl->matmul_out = clCreateBuffer(gcl->context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL buffer: %d\n", err);
        exit(1);
    }

    size = MAX(4 * C * C, V * C);
    gcl->matmul_weight = clCreateBuffer(gcl->context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL buffer: %d\n", err);
        exit(1);
    }

    gcl->matmul_bias = clCreateBuffer(gcl->context, CL_MEM_READ_ONLY, sizeof(float) * 4 * C, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL buffer: %d\n", err);
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(gcl->matmul_forward, 0, sizeof(cl_mem), &gcl->matmul_out);
    err |= clSetKernelArg(gcl->matmul_forward, 1, sizeof(cl_mem), &gcl->matmul_inp);
    err |= clSetKernelArg(gcl->matmul_forward, 2, sizeof(cl_mem), &gcl->matmul_weight);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
}

void cl_deinit(GPT2_CL *gcl) {
    clReleaseMemObject(gcl->matmul_inp);
    clReleaseMemObject(gcl->matmul_out);
    clReleaseMemObject(gcl->matmul_weight);
    clReleaseMemObject(gcl->matmul_bias);
    clReleaseKernel(gcl->matmul_forward);
    clReleaseProgram(gcl->program);
    clReleaseCommandQueue(gcl->queue);
    clReleaseContext(gcl->context);
}
