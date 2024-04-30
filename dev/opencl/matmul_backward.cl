MSTRINGIFY(


__kernel void matmul_backward1(__global float* out, __global float* AMat, __global float* BMat,
                    int B, int T, int C, int OC)
{
    // define local memory for AMat and BMat tiles with padding
    __local float A_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];

    // get global and local IDs
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);
    bool is_valid_g0 = global_id0 < B * T;
    bool is_valid_g1 = global_id1 < C;

    // calculate number of tiles
    int num_tiles = (OC + TILE_SIZE - 1) / TILE_SIZE;

    // initialize output value
    float val = 0.0f;

    // loop over tiles
    for (int t = 0; t < num_tiles; ++t) {
        int row = t * TILE_SIZE + local_id0;
        int col = t * TILE_SIZE + local_id1;

        // load AMat tile
        A_tile[local_id0][local_id1] = (col < OC && is_valid_g0) ? AMat[global_id0 * OC + col] : 0.0f;

        // BMat tile
        B_tile[local_id1][local_id0] = (row < OC && is_valid_g1) ? BMat[row * C + global_id1] : 0.0f;

        // synchronize to make sure all data is loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // compute partial dot product
        #if MATMUL_VLOAD_SIZE == 4
            for (int i = 0; i < TILE_SIZE/4; i++) {
                float4 A_vec = vload4(i, A_tile[local_id0]);
                float4 B_vec = vload4(i, B_tile[local_id1]);
                val += dot(A_vec, B_vec);
            }
        #elif MATMUL_VLOAD_SIZE == 8
            for (int i = 0; i < TILE_SIZE/8; i++) {
                float8 A_vec = vload8(i, A_tile[local_id0]);
                float8 B_vec = vload8(i, B_tile[local_id1]);
                val += dot(A_vec.lo, B_vec.lo);
                val += dot(A_vec.hi, B_vec.hi);
            }
        #elif MATMUL_VLOAD_SIZE == 16
            for (int i = 0; i < TILE_SIZE/16; i++) {
                float16 A_vec = vload16(i, A_tile[local_id0]);
                float16 B_vec = vload16(i, B_tile[local_id1]);
                val += dot(A_vec.lo.lo, B_vec.lo.lo);
                val += dot(A_vec.lo.hi, B_vec.lo.hi);
                val += dot(A_vec.hi.lo, B_vec.hi.lo);
                val += dot(A_vec.hi.hi, B_vec.hi.hi);
            }
        #else
            for (int i = 0; i < TILE_SIZE; ++i) {
                val += A_tile[local_id0][i] * B_tile[local_id1][i];
            }
        #endif

        // synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result to global memory
    if (is_valid_g0 && is_valid_g1) {
        out[global_id0 * C + global_id1] += val;
    }
}


__kernel void matmul_backward2(__global float* out, __global float* AMat, __global float* BMat,
                    int B, int T, int C, int OC)
{
    // define local memory for AMat and BMat tiles with padding
    __local float A_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE + LOCAL_MEM_PADDING_SIZE];

    // get global and local IDs
    size_t global_id0 = get_global_id(0);
    size_t global_id1 = get_global_id(1);
    size_t local_id0 = get_local_id(0);
    size_t local_id1 = get_local_id(1);
    bool is_valid_g0 = global_id0 < OC;
    bool is_valid_g1 = global_id1 < C;

    // calculate number of tiles
    int num_tiles = ((B * T) + TILE_SIZE - 1) / TILE_SIZE;

    // initialize output value
    float val = 0.0f;

    // loop over tiles
    for (int t = 0; t < num_tiles; ++t) {
        int row = t * TILE_SIZE + local_id0;
        int col = t * TILE_SIZE + local_id1;

        // load AMat tile
        A_tile[local_id0][local_id1] = (col < (B * T) && is_valid_g0) ? AMat[col * OC + global_id0] : 0.0f;

        // BMat tile
        B_tile[local_id1][local_id0] = (row < (B * T) && is_valid_g1) ? BMat[row * C + global_id1] : 0.0f;

        // synchronize to make sure all data is loaded into local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        // compute partial dot product
        #if MATMUL_VLOAD_SIZE == 4
            for (int i = 0; i < TILE_SIZE/4; i++) {
                float4 A_vec = vload4(i, A_tile[local_id0]);
                float4 B_vec = vload4(i, B_tile[local_id1]);
                val += dot(A_vec, B_vec);
            }
        #elif MATMUL_VLOAD_SIZE == 8
            for (int i = 0; i < TILE_SIZE/8; i++) {
                float8 A_vec = vload8(i, A_tile[local_id0]);
                float8 B_vec = vload8(i, B_tile[local_id1]);
                val += dot(A_vec.lo, B_vec.lo);
                val += dot(A_vec.hi, B_vec.hi);
            }
        #elif MATMUL_VLOAD_SIZE == 16
            for (int i = 0; i < TILE_SIZE/16; i++) {
                float16 A_vec = vload16(i, A_tile[local_id0]);
                float16 B_vec = vload16(i, B_tile[local_id1]);
                val += dot(A_vec.lo.lo, B_vec.lo.lo);
                val += dot(A_vec.lo.hi, B_vec.lo.hi);
                val += dot(A_vec.hi.lo, B_vec.hi.lo);
                val += dot(A_vec.hi.hi, B_vec.hi.hi);
            }
        #else
            for (int i = 0; i < TILE_SIZE; ++i) {
                val += A_tile[local_id0][i] * B_tile[local_id1][i];
            }
        #endif

        // synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result to global memory
    if (is_valid_g0 && is_valid_g1) {
        out[global_id0 * C + global_id1] += val;
    }
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