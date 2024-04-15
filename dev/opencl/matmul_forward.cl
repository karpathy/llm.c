MSTRINGIFY(

__kernel void matmul_forward(__global float* out, __global float* inp, __global float* weight,
                    int B, int T, int C, int OC)
{
    size_t b = get_global_id(0) / T;
    size_t t = get_global_id(0) - (b * T);

    if (b >= (size_t)B || t >= (size_t)T)
        return;

    __global float* out_bt = out + b * T * OC + t * OC;
    __global float* inp_bt = inp + b * T * C + t * C;

    for (int o = 0; o < OC; o++) {
        float val = 0.0f;
        __global float* wrow = weight + o*C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out_bt[o] = val;
    }
}

)