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
    size_t wg_id0 = get_group_id(0);
    size_t wg_id1 = get_group_id(1);
    bool is_valid_g0 = global_id0 < B * T;
    bool is_valid_g1 = global_id1 < C;
    int t;

    // if wg is in tail areas
    bool is_tail_wg = (((wg_id0 * TILE_SIZE) + TILE_SIZE - 1) >= B * T) ||
                        (((wg_id1 * TILE_SIZE) + TILE_SIZE - 1) >= C);

    // initialize output value
    float val = 0.0f;

    // wgs with most conditionals
    if(is_tail_wg) {
        // loop over tiles
        for (t = 0; (t+TILE_SIZE-1) < OC; t+=TILE_SIZE) {
            // load AMat tile
            A_tile[local_id0][local_id1] = is_valid_g0 ? AMat[global_id0 * OC + t + local_id1] : 0.0f;

            // transpose BMat tile
            B_tile[local_id1][local_id0] = is_valid_g1 ? BMat[(t + local_id0) * C + global_id1] : 0.0f;

            // compute partial dot product
            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
        if(t < OC) {
            int row = t + local_id0;
            int col = t + local_id1;

            A_tile[local_id0][local_id1] = (col < C && is_valid_g0) ? AMat[global_id0 * OC + col] : 0.0f;
            B_tile[local_id1][local_id0] = (row < C && is_valid_g1) ? BMat[row * C + global_id1] : 0.0f;

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
    }
    else {
        __global float *A_ptr = AMat + wg_id0 * TILE_SIZE * OC;
        __global float *B_ptr = BMat + wg_id1 * TILE_SIZE;
        size_t l0cl1 = local_id0 * C + local_id1;
        size_t l0ocl1 = local_id0 * OC + local_id1;

        for (t = 0; (t+TILE_SIZE-1) < OC; t+=TILE_SIZE) {

            #if DO_PRELOAD == 1
                // prefetch next tile
                if(t+TILE_SIZE < OC) {
                    prefetch(A_ptr + TILE_SIZE + l0ocl1, 1);
                    prefetch(B_ptr + (C * TILE_SIZE) + l0cl1, 1);
                }
            #endif

            // load current tile
            A_tile[local_id0][local_id1] = A_ptr[l0ocl1];
            B_tile[local_id1][local_id0] = B_ptr[l0cl1];

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);

            A_ptr += TILE_SIZE;
            B_ptr += C * TILE_SIZE;
        }
        if(t < OC) {
            int row = t + local_id0;
            int col = t + local_id1;

            A_tile[local_id0][local_id1] = (col < C) ? AMat[global_id0 * OC + col] : 0.0f;
            B_tile[local_id1][local_id0] = (row < C) ? BMat[row * C + global_id1] : 0.0f;

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
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
    size_t wg_id0 = get_group_id(0);
    size_t wg_id1 = get_group_id(1);
    bool is_valid_g0 = global_id0 < OC;
    bool is_valid_g1 = global_id1 < C;
    int t, bt = B * T;

    // if wg is in tail areas
    bool is_tail_wg = (((wg_id0 * TILE_SIZE) + TILE_SIZE - 1) >= OC) ||
                        (((wg_id1 * TILE_SIZE) + TILE_SIZE - 1) >= C);

    // initialize output value
    float val = 0.0f;

    // wgs with most conditionals
    if(is_tail_wg) {
        // loop over tiles
        for (t = 0; (t+TILE_SIZE-1) < bt; t+=TILE_SIZE) {
            // load AMat tile
            A_tile[local_id0][local_id1] = is_valid_g0 ? AMat[(t + local_id1) * OC + global_id0] : 0.0f;

            // transpose BMat tile
            B_tile[local_id1][local_id0] = is_valid_g1 ? BMat[(t + local_id0) * C + global_id1] : 0.0f;

            // compute partial dot product
            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
        if(t < bt) {
            int row = t + local_id0;
            int col = t + local_id1;

            A_tile[local_id0][local_id1] = (col < C && is_valid_g0) ? AMat[col * OC + global_id0] : 0.0f;
            B_tile[local_id1][local_id0] = (row < C && is_valid_g1) ? BMat[row * C + global_id1] : 0.0f;

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
    }
    else {
        __global float *A_ptr = AMat + wg_id0 * TILE_SIZE;
        __global float *B_ptr = BMat + wg_id1 * TILE_SIZE;
        size_t l0cl1 = local_id0 * C + local_id1;
        size_t l0ocl1 = local_id0 * OC + local_id1;

        for (t = 0; (t+TILE_SIZE-1) < bt; t+=TILE_SIZE) {

            #if DO_PRELOAD == 1
                // prefetch next tile
                if(t+TILE_SIZE < bt) {
                    prefetch(A_ptr + (OC * TILE_SIZE) + l0ocl1, 1);
                    prefetch(B_ptr + (C * TILE_SIZE) + l0cl1, 1);
                }
            #endif

            // load current tile
            A_tile[local_id1][local_id0] = A_ptr[l0ocl1];
            B_tile[local_id1][local_id0] = B_ptr[l0cl1];

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);

            A_ptr += OC * TILE_SIZE;
            B_ptr += C * TILE_SIZE;
        }
        if(t < bt) {
            int row = t + local_id0;
            int col = t + local_id1;

            A_tile[local_id0][local_id1] = (col < C) ? AMat[col * OC + global_id0] : 0.0f;
            B_tile[local_id1][local_id0] = (row < C) ? BMat[row * C + global_id1] : 0.0f;

            val += perform_dot(A_tile, B_tile, local_id0, local_id1);
        }
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

    __global float *A_ptr = AMat + global_id0;
    float val = 0.0f;
    for(int bt=0; bt<(B * T); bt++) {

        #if DO_PRELOAD == 1
            if((bt + 1) < (B * T)) {
                prefetch(A_ptr + ((bt + 1) * OC), 1);
            }
        #endif

        val += A_ptr[bt * OC];
    }

    bias[global_id0] += val;
}

)