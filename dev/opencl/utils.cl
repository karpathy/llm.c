MSTRINGIFY(

float perform_dot(__local float A_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            __local float B_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            size_t local_id0, size_t local_id1) {
    float val = 0.0f;

    // synchronize to make sure all data is loaded into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    \n#if VLOAD_SIZE == 4\n
        for (int i = 0; i < TILE_SIZE/4; i++) {
            float4 A_vec = vload4(i, A_tile[local_id0]);
            float4 B_vec = vload4(i, B_tile[local_id1]);

            \n#if USE_DOT_PRODUCT == 1\n
                val += dot(A_vec, B_vec);
            \n#else\n
                val = mad(A_vec.x, B_vec.x, val);
                val = mad(A_vec.y, B_vec.y, val);
                val = mad(A_vec.z, B_vec.z, val);
                val = mad(A_vec.w, B_vec.w, val);
            \n#endif\n
        }
    \n#elif VLOAD_SIZE == 8\n
        for (int i = 0; i < TILE_SIZE/8; i++) {
            float8 A_vec = vload8(i, A_tile[local_id0]);
            float8 B_vec = vload8(i, B_tile[local_id1]);

            \n#if USE_DOT_PRODUCT == 1\n
                val += dot(A_vec.lo, B_vec.lo);
                val += dot(A_vec.hi, B_vec.hi);
            \n#else\n
                val = mad(A_vec.S0, B_vec.S0, val);
                val = mad(A_vec.S1, B_vec.S1, val);
                val = mad(A_vec.S2, B_vec.S2, val);
                val = mad(A_vec.S3, B_vec.S3, val);
                val = mad(A_vec.S4, B_vec.S4, val);
                val = mad(A_vec.S5, B_vec.S5, val);
                val = mad(A_vec.S6, B_vec.S6, val);
                val = mad(A_vec.S7, B_vec.S7, val);
            \n#endif\n
        }
    \n#elif VLOAD_SIZE == 16\n
        for (int i = 0; i < TILE_SIZE/16; i++) {
            float16 A_vec = vload16(i, A_tile[local_id0]);
            float16 B_vec = vload16(i, B_tile[local_id1]);

            \n#if USE_DOT_PRODUCT == 1\n
                val += dot(A_vec.lo.lo, B_vec.lo.lo);
                val += dot(A_vec.lo.hi, B_vec.lo.hi);
                val += dot(A_vec.hi.lo, B_vec.hi.lo);
                val += dot(A_vec.hi.hi, B_vec.hi.hi);
            \n#else\n
                val = mad(A_vec.S0, B_vec.S0, val);
                val = mad(A_vec.S1, B_vec.S1, val);
                val = mad(A_vec.S2, B_vec.S2, val);
                val = mad(A_vec.S3, B_vec.S3, val);
                val = mad(A_vec.S4, B_vec.S4, val);
                val = mad(A_vec.S5, B_vec.S5, val);
                val = mad(A_vec.S6, B_vec.S6, val);
                val = mad(A_vec.S7, B_vec.S7, val);
                val = mad(A_vec.S8, B_vec.S8, val);
                val = mad(A_vec.S9, B_vec.S9, val);
                val = mad(A_vec.SA, B_vec.SA, val);
                val = mad(A_vec.SB, B_vec.SB, val);
                val = mad(A_vec.SC, B_vec.SC, val);
                val = mad(A_vec.SD, B_vec.SD, val);
                val = mad(A_vec.SE, B_vec.SE, val);
                val = mad(A_vec.SF, B_vec.SF, val);
            \n#endif\n
        }
    \n#else\n
        for (int i = 0; i < TILE_SIZE; i++) {
            \n#if USE_DOT_PRODUCT == 1\n
                val += dot(A_tile[local_id0][i], B_tile[local_id1][i]);
            \n#else\n
                val = mad(A_tile[local_id0][i], B_tile[local_id1][i], val);
            \n#endif\n
        }
    \n#endif\n

    // synchronize before loading next tile
    barrier(CLK_LOCAL_MEM_FENCE);

    return val;
}

)