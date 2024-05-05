MSTRINGIFY(

inline float perform_dot(__local float A_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            __local float B_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            size_t local_id0, size_t local_id1) {
    float val = 0.0f;

    // synchronize to make sure all data is loaded into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

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

    return val;
}

)