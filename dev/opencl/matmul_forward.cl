MSTRINGIFY(

__kernel void matmul_forward(__global float* out, __global float* inp, __global float* weight,
                    int B, int T, int C, int OC)
{
    size_t b = get_global_id(0) / (T * OC);
    size_t t = (get_global_id(0) - (b * T * OC)) / OC;
    size_t o = get_global_id(0) - (b * T * OC) - (t * OC);

    if (b >= (size_t)B || t >= (size_t)T || o >= (size_t)OC)
        return;

    __global float* out_bt = out + b * T * OC + t * OC;
    __global float* inp_bt = inp + b * T * C + t * C;
    __global float* wrow = weight + o*C;

    float val = 0.0f;
    for (int i = 0; i < C; i++) {
        val += inp_bt[i] * wrow[i];
    }
    out_bt[o] = val;
}

)