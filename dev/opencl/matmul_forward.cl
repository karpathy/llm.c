MSTRINGIFY(


__kernel void matmul_forward(__global float* out, __global float* AMat,
                    __global float* BMat, __global float* bias,
                    int B, int T, int C, int OC, int use_bias)
{
    // define local memory for AMat and BMat tiles with padding
    __local float inp_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];
    __local float weight_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];

    // get global and local IDs
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);

    // calculate number of tiles
    int num_tiles = (C + TILE_SIZE - 1) / TILE_SIZE;

    // initialize output value
    float val = use_bias? bias[global_id1] : 0.0f;

    // loop over tiles
    for (int t = 0; t < num_tiles; ++t) {
        // load input tile into local memory
        int row = t * TILE_SIZE + local_id0;
        int col = t * TILE_SIZE + local_id1;

        // load input tile
        inp_tile[local_id0][local_id1] = (row < C && global_id0 < B * C) ? AMat[global_id0 * C + col] : 0.0f;

        // transpose BMat tile
        weight_tile[local_id1][local_id0] = (col < C && global_id1 < OC) ? BMat[global_id1 * C + row] : 0.0f;

        // synchronize to make sure all data is loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // compute partial dot product
        #if MATMUL_VLOAD_SIZE == 4
            for (int i = 0; i < TILE_SIZE/4; i++) {
                float4 inp_vec = vload4(i, inp_tile[local_id0]);
                float4 weight_vec = vload4(i, weight_tile[local_id1]);
                val += dot(inp_vec, weight_vec);
            }
        #elif MATMUL_VLOAD_SIZE == 8
            for (int i = 0; i < TILE_SIZE/8; i++) {
                float8 inp_vec = vload8(i, inp_tile[local_id0]);
                float8 weight_vec = vload8(i, weight_tile[local_id1]);
                val += dot(inp_vec.lo, weight_vec.lo);
                val += dot(inp_vec.hi, weight_vec.hi);
            }
        #elif MATMUL_VLOAD_SIZE == 16
            for (int i = 0; i < TILE_SIZE/16; i++) {
                float16 inp_vec = vload16(i, inp_tile[local_id0]);
                float16 weight_vec = vload16(i, weight_tile[local_id1]);
                val += dot(inp_vec.lo.lo, weight_vec.lo.lo);
                val += dot(inp_vec.lo.hi, weight_vec.lo.hi);
                val += dot(inp_vec.hi.lo, weight_vec.hi.lo);
                val += dot(inp_vec.hi.hi, weight_vec.hi.hi);
            }
        #else
            for (int i = 0; i < TILE_SIZE; ++i) {
                val += inp_tile[local_id0][i] * weight_tile[local_id1][i];
            }
        #endif

        // synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result to global memory
    if (global_id0 < B * C && global_id1 < OC) {
        out[global_id0 * OC + global_id1] = val;
    }
}


)