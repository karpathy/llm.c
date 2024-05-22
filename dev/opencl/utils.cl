MSTRINGIFY(

float perform_dot(__local float A_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            __local float B_tile[TILE_SIZE][TILE_SIZE+LOCAL_MEM_PADDING_SIZE],
            size_t local_id0, size_t local_id1) {
    float val = 0.0f;

    // synchronize to make sure all data is loaded into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    \n#if VLOAD_SIZE == 4\n

        float4 t4 = (float4)(0.0);
        for (int i = 0; i < TILE_SIZE/4; i++) {
            float4 A_vec = vload4(i, A_tile[local_id0]);
            float4 B_vec = vload4(i, B_tile[local_id1]);

            \n#if USE_MAD == 1\n
                t4 = mad(A_vec, B_vec, t4);
            \n#else\n
                t4 = (A_vec * B_vec) + t4;
            \n#endif\n
        }
        float2 t2 = t4.hi + t4.lo;
        val = t2.hi + t2.lo;

    \n#elif VLOAD_SIZE == 8\n

        float8 t8 = (float8)(0.0);
        for (int i = 0; i < TILE_SIZE/8; i++) {
            float8 A_vec = vload8(i, A_tile[local_id0]);
            float8 B_vec = vload8(i, B_tile[local_id1]);

            \n#if USE_MAD == 1\n
                t8 = mad(A_vec, B_vec, t8);
            \n#else\n
                t8 = (A_vec * B_vec) + t8;
            \n#endif\n
        }
        float4 t4 = t8.hi + t8.lo;
        float2 t2 = t4.hi + t4.lo;
        val = t2.hi + t2.lo;

    \n#elif VLOAD_SIZE == 16\n

        float16 t16 = (float16)(0.0);
        for (int i = 0; i < TILE_SIZE/16; i++) {
            float16 A_vec = vload16(i, A_tile[local_id0]);
            float16 B_vec = vload16(i, B_tile[local_id1]);

            \n#if USE_MAD == 1\n
                t16 = mad(A_vec, B_vec, t16);
            \n#else\n
                t16 = (A_vec * B_vec) + t16;
            \n#endif\n
        }
        float8 t8 = t16.hi + t16.lo;
        float4 t4 = t8.hi + t8.lo;
        float2 t2 = t4.hi + t4.lo;
        val = t2.hi + t2.lo;

    \n#else\n

        for (int i = 0; i < TILE_SIZE; i++) {
            \n#if USE_MAD == 1\n
                val = mad(A_tile[local_id0][i], B_tile[local_id1][i], val);
            \n#else\n
                val = (A_tile[local_id0][i] * B_tile[local_id1][i]) + val;
            \n#endif\n
        }

    \n#endif\n

    // synchronize before loading next tile
    barrier(CLK_LOCAL_MEM_FENCE);

    return val;
}

)