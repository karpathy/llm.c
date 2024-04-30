MSTRINGIFY(


__kernel void matmul_backward1(__global float* out, __global float* AMat, __global float* BMat,
                    int B, int T, int C, int OC)
{
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);

    if(global_id0 >= B*C || global_id1 >= C) return;

    float val = 0.0f;
    for(int o=0; o<OC; o++) {
        val += AMat[global_id0 * OC + o] * BMat[o * C + global_id1];
    }

    out[global_id0 * C + global_id1] += val;
}


__kernel void matmul_backward2(__global float* out, __global float* AMat, __global float* BMat,
                    int B, int T, int C, int OC)
{
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);

    if(global_id0 >= OC || global_id1 >= C) return;

    float val = 0.0f;
    for(int bt=0; bt<(B * T); bt++) {
        val += AMat[bt * OC + global_id0] * BMat[bt * C + global_id1];
    }

    out[global_id0 * C + global_id1] += val;
}

__kernel void matmul_backward3(__global float* AMat, __global float* bias,
                    int B, int T, int C, int OC)
{
    size_t global_id0 = get_global_id(0);

    if(global_id0 >= OC) return;

    float val = 0.0f;
    for(int bt=0; bt<(B * T); bt++) {
        val += AMat[bt * OC + global_id0];
    }

    bias[global_id0] += val;
}

)