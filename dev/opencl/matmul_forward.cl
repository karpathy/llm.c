MSTRINGIFY(

__kernel void matmul_forward(__global float* out, __global float* inp, __global float* weight,
                    int B, int T, int C, int OC)
{
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);

    int x_tile_end = (global_id0 * TILE_SIZE) + TILE_SIZE;
    x_tile_end = x_tile_end < (B * T)? x_tile_end: (B * T);
    int y_tile_end = (global_id1 * TILE_SIZE) + TILE_SIZE;
    y_tile_end = y_tile_end < OC? y_tile_end: OC;

    for(int x=global_id0 * TILE_SIZE; x<x_tile_end; x++) {
        for(int y=global_id1 * TILE_SIZE; y<y_tile_end; y++) {
            float val = 0.0f;
            for(int i=0; i<C; i++) {
                val += inp[x * C + i] * weight[y * C + i];
            }
            out[x * OC + y] = val;
        }
    }
}

)