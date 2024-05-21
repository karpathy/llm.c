#define TESTING
#include "train_gpt2cl.c"

#define NUM_STEPS 6
#define SKIP_STEPS 3

int do_run(double *time_taken) {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    GPT2_CL gcl;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); return 1; }
    int state_header[256];
    fread(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); return 1; }
    if (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        return 1;
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)

    // inputs and expected outputs, only used for error checking
    int* x = (int*) malloc(B * T * sizeof(int));
    int* y = (int*) malloc(B * T * sizeof(int));

    // read reference information from Python
    fread(x, sizeof(int), B*T, state_file);
    fread(y, sizeof(int), B*T, state_file);
    fclose(state_file);

    int clret = cl_init(&gcl, B, T, C, Vp);
    if (clret != 0) {
        printf("error initializing opencl\n");
        free(x);
        free(y);
        gpt2_free(&model);
        cl_deinit(&gcl);
        return clret;
    }

    struct timespec start, end;
    double total_time = 0.0;
    for (int step = 0; step < NUM_STEPS; step++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&gcl, &model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&gcl, &model);

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // consider after skip steps for warmup
        if(step >= SKIP_STEPS) {
            total_time += time_elapsed_s;
        }
    }

    *time_taken = total_time * 1000 / (NUM_STEPS - SKIP_STEPS);

    // free everything
    free(x);
    free(y);
    gpt2_free(&model);
    cl_deinit(&gcl);
    return 0;
}

int main(int argc, char *argv[]) {
    double time_taken = 0.0;
    char str[16];

    int tile_size_lst[] = {4, 8, 12, 16, 24, 32, 48, 64};
    int lmp_size_lst[] = {0, 1};
    int vload_size_lst[] = {0, 4, 8, 16};
    int do_preload_lst[] = {0, 1};
    int use_dp_lst[] = {0, 1};

    double best_time_taken = 1e10;
    int best_tile_size = 0;
    int best_lmp_size = 0;
    int best_vload_size = 0;
    int best_do_preload = 0;
    int best_use_dp = 0;

    for(int ti=0; ti<sizeof(tile_size_lst)/sizeof(tile_size_lst[0]); ti++) {
        int tile_size = tile_size_lst[ti];
        snprintf(str, sizeof(str), "%d", tile_size);
        setenv("MATMUL_TILE_SIZE", str, 1);

        for(int lmpi=0; lmpi<sizeof(lmp_size_lst)/sizeof(lmp_size_lst[0]); lmpi++) {
            int lmp_size = lmp_size_lst[lmpi];
            snprintf(str, sizeof(str), "%d", lmp_size);
            setenv("MATMUL_LOCAL_MEM_PADDING_SIZE", str, 1);

            for(int vli=0; vli<sizeof(vload_size_lst)/sizeof(vload_size_lst[0]); vli++) {
                int vload_size = vload_size_lst[vli];
                snprintf(str, sizeof(str), "%d", vload_size);
                setenv("MATMUL_VLOAD_SIZE", str, 1);

                for(int dpli=0; dpli<sizeof(do_preload_lst)/sizeof(do_preload_lst[0]); dpli++) {
                    int do_preload = do_preload_lst[dpli];
                    snprintf(str, sizeof(str), "%d", do_preload);
                    setenv("MATMUL_DO_PRELOAD", str, 1);

                    for(int udpi=0; udpi<sizeof(use_dp_lst)/sizeof(use_dp_lst[0]); udpi++) {
                        int use_dp = use_dp_lst[udpi];
                        snprintf(str, sizeof(str), "%d", use_dp);
                        setenv("MATMUL_USE_DOT_PRODUCT", str, 1);

                        printf("MATMUL_TILE_SIZE=%d MATMUL_LOCAL_MEM_PADDING_SIZE=%d MATMUL_VLOAD_SIZE=%d MATMUL_DO_PRELOAD=%d MATMUL_USE_DOT_PRODUCT=%d\n",
                                tile_size, lmp_size, vload_size, do_preload, use_dp);
                        printf("---------------------------------------------\n");

                        int ret = do_run(&time_taken);
                        if (ret == 0) {
                            printf("---------------------------------------------\n");
                            printf("time taken: %lf ms\n", time_taken);
                            if(time_taken < best_time_taken) {
                                best_time_taken = time_taken;
                                best_tile_size = tile_size;
                                best_lmp_size = lmp_size;
                                best_vload_size = vload_size;
                                best_do_preload = do_preload;
                                best_use_dp = use_dp;
                            }
                        } else {
                            printf("skipping\n");
                        }
                        printf("---------------------------------------------\n");
                    }
                }
            }
        }
    }
    printf("\nbest time taken: %lf ms with combination\n", best_time_taken);
    printf("MATMUL_TILE_SIZE=%d MATMUL_LOCAL_MEM_PADDING_SIZE=%d MATMUL_VLOAD_SIZE=%d MATMUL_DO_PRELOAD=%d MATMUL_USE_DOT_PRODUCT=%d\n",
                best_tile_size, best_lmp_size, best_vload_size, best_do_preload, best_use_dp);
    return 0;
}
