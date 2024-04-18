MSTRINGIFY(

__kernel void matmul_forward(__global float* out, __global float* inp, __global float* weight,
                    int B, int T, int C, int OC)
{
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t global_size1 = get_global_size(1);
    size_t b = global_id0 / T;
    size_t t = global_id0 - (b * T);

    __global float* out_bt = out + b * T * OC + t * OC;
    __global float* inp_bt = inp + b * T * C + t * C;

    for(int o = global_id1; o < OC; o += global_size1) {
        __global float* wrow = weight + o*C;
        float val = 0.0f;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
        out_bt[o] = val;
    }
}

)