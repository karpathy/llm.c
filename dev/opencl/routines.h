#include "common.h"

void cl_matmul_forward(GPT2_CL *gcl, float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    cl_int err = 0;

    err = clEnqueueWriteBuffer(gcl->queue, gcl->matmul_A, CL_TRUE, 0, sizeof(float) * B * T * C, inp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_B, CL_TRUE, 0, sizeof(float) * OC * C, weight, 0, NULL, NULL);
    if(bias != NULL) {
        err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_bias, CL_TRUE, 0, sizeof(float) * OC, bias, 0, NULL, NULL);
    }
    if (err != CL_SUCCESS)
    {
        printf("error: failed to write to source array!\n");
        exit(1);
    }

    cl_int use_bias = bias != NULL;
    err = clSetKernelArg(gcl->matmul_forward, 4, sizeof(int), &B);
    err |= clSetKernelArg(gcl->matmul_forward, 5, sizeof(int), &T);
    err |= clSetKernelArg(gcl->matmul_forward, 6, sizeof(int), &C);
    err |= clSetKernelArg(gcl->matmul_forward, 7, sizeof(int), &OC);
    err |= clSetKernelArg(gcl->matmul_forward, 8, sizeof(cl_int), &use_bias);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    size_t tile_size = MATMUL_TILE_SIZE;
    size_t bt_round = (((B * T) + tile_size - 1) / tile_size) * tile_size;
    size_t oc_round = ((OC + tile_size - 1) / tile_size) * tile_size;
    size_t size_global[2] = {bt_round, oc_round};
    size_t size_local[2] = {tile_size, tile_size};
    err = clEnqueueNDRangeKernel(gcl->queue, gcl->matmul_forward, 2, NULL, size_global, size_local, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to execute kernel! %d\n", err);
        exit(1);
    }
    clFinish(gcl->queue);

    err = clEnqueueReadBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * B * T * OC, out, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("error: failed to read output array! %d\n", err);
        exit(1);
    }
}

void cl_matmul_backward(GPT2_CL *gcl, float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // dout is (B,T,OC), weight is (OC, C)
    // dinp will be (B,T,C)
    // dinp += dout * weight

    cl_int err = 0;

    err = clEnqueueWriteBuffer(gcl->queue, gcl->matmul_A, CL_TRUE, 0, sizeof(float) * B * T * OC, dout, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_B, CL_TRUE, 0, sizeof(float) * OC * C, weight, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * B * T * C, dinp, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to write to source array!\n");
        exit(1);
    }

    err = clSetKernelArg(gcl->matmul_backward1, 3, sizeof(int), &B);
    err |= clSetKernelArg(gcl->matmul_backward1, 4, sizeof(int), &T);
    err |= clSetKernelArg(gcl->matmul_backward1, 5, sizeof(int), &C);
    err |= clSetKernelArg(gcl->matmul_backward1, 6, sizeof(int), &OC);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    size_t tile_size = MATMUL_TILE_SIZE;
    size_t bt_round = (((B * T) + tile_size - 1) / tile_size) * tile_size;
    size_t c_round = ((C + tile_size - 1) / tile_size) * tile_size;
    size_t size_global1[2] = {bt_round, c_round};
    size_t size_local1[2] = {tile_size, tile_size};
    err = clEnqueueNDRangeKernel(gcl->queue, gcl->matmul_backward1, 2, NULL, size_global1, size_local1, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to execute kernel! %d\n", err);
        exit(1);
    }
    clFinish(gcl->queue);

    err = clEnqueueReadBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * B * T * C, dinp, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("error: failed to read output array! %d\n", err);
        exit(1);
    }

    // inp is (B,T,C), dout is (B,T,OC)
    // dweight will be (OC,C)
    // dweight += dout^T * inp

    err = clEnqueueWriteBuffer(gcl->queue, gcl->matmul_A, CL_TRUE, 0, sizeof(float) * B * T * OC, dout, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_B, CL_TRUE, 0, sizeof(float) * B * T * C, inp, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * OC * C, dweight, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to write to source array!\n");
        exit(1);
    }

    err = clSetKernelArg(gcl->matmul_backward2, 3, sizeof(int), &B);
    err |= clSetKernelArg(gcl->matmul_backward2, 4, sizeof(int), &T);
    err |= clSetKernelArg(gcl->matmul_backward2, 5, sizeof(int), &C);
    err |= clSetKernelArg(gcl->matmul_backward2, 6, sizeof(int), &OC);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    size_t oc_round = ((OC + tile_size - 1) / tile_size) * tile_size;
    c_round = ((C + tile_size - 1) / tile_size) * tile_size;
    size_t size_global2[2] = {oc_round, c_round};
    size_t size_local2[2] = {tile_size, tile_size};
    err = clEnqueueNDRangeKernel(gcl->queue, gcl->matmul_backward2, 2, NULL, size_global2, size_local2, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("error: failed to execute kernel! %d\n", err);
        exit(1);
    }
    clFinish(gcl->queue);

    err = clEnqueueReadBuffer(gcl->queue, gcl->matmul_out, CL_TRUE, 0, sizeof(float) * OC * C, dweight, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("error: failed to read output array! %d\n", err);
        exit(1);
    }

    // dbias will be (OC)
    if(dbias != NULL) {
        err = clEnqueueWriteBuffer(gcl->queue, gcl->matmul_A, CL_TRUE, 0, sizeof(float) * B * T * OC, dout, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(gcl->queue, gcl->matmul_bias, CL_TRUE, 0, sizeof(float) * OC, dbias, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("error: failed to write to source array!\n");
            exit(1);
        }

        err = clSetKernelArg(gcl->matmul_backward3, 2, sizeof(int), &B);
        err |= clSetKernelArg(gcl->matmul_backward3, 3, sizeof(int), &T);
        err |= clSetKernelArg(gcl->matmul_backward3, 4, sizeof(int), &C);
        err |= clSetKernelArg(gcl->matmul_backward3, 5, sizeof(int), &OC);
        if (err != CL_SUCCESS)
        {
            printf("error: failed to set kernel arguments! %d\n", err);
            exit(1);
        }

        size_t wg_size = gcl->max_wg_size;
        size_t oc_round = ((OC + wg_size - 1) / wg_size) * wg_size;
        size_t size_global3 = oc_round;
        size_t size_local3 = wg_size;
        err = clEnqueueNDRangeKernel(gcl->queue, gcl->matmul_backward3, 1, NULL, &size_global3, &size_local3, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("error: failed to execute kernel! %d\n", err);
            exit(1);
        }
        clFinish(gcl->queue);

        err = clEnqueueReadBuffer(gcl->queue, gcl->matmul_bias, CL_TRUE, 0, sizeof(float) * OC, dbias, 0, NULL, NULL );
        if (err != CL_SUCCESS)
        {
            printf("error: failed to read output array! %d\n", err);
            exit(1);
        }
    }
}
