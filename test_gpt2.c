#define TESTING
#include "train_gpt2.c"

#define VERBOSITY_LEVEL_NONE 0 // only print OK or NOT OK
#define VERBOSITY_LEVEL_ERROR 1 // if NOT OK print a few mismataches
#define VERBOSITY_LEVEL_DEBUG 2 // if OK print a few values
#define VERBOSITY_LEVEL VERBOSITY_LEVEL_ERROR

int check_tensor(float* a, float* b, int n, const char* label, float tolerance) {
// poor man's tensor checker. (Note: duplicate of test_gpt2.cu)
#ifndef _NO_COLORS
#define FMT_PARAM_NOT_OK_DIFF_S "%s - \033[41mNOT OK\033[0m w/ tol = \033[33m%e\033[0m, faults = \033[41m%llu\033[0m, maxdiff=\033[33m%e\033[0m\n"
#define FMT_PARAM_OK_DIFF_S "%s - \033[42mOK\033[0m w/ tol = \033[36m%e\033[0m, maxdiff=%e\n"
#define FMT_TENSOR_NOT_OK_S "%s[%d] \033[31m%f8 %f8\033[0m, diff=\033[33m%e\033[0m\n"
#define FMT_TENSOR_OK_S "%s[%d] %f8 %f8\n"
#else
#define FMT_PARAM_NOT_OK_DIFF_S "%s - NOT OK w/ tol = %e, faults = %llu, maxdiff=%e\n"
#define FMT_PARAM_OK_DIFF_S "%s - OK w/ tol = %e, maxdiff=%e\n"
#define FMT_TENSOR_NOT_OK_S "%s[%d] %f8 %f8, diff=%e\n"
#define FMT_TENSOR_OK_S "%s[%d] %f8 %f8\n"
#endif
    unsigned long long faults = 0;
    float maxdiff = 0.0f;
    // check the entire tensor without printing anything
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tolerance) { faults++; }
        if (diff > maxdiff) { maxdiff = diff; }
    }
    const int PRINT_UP_TO = 5;
    int num_printed = 0;
    // print the final OK or NOT OK result
    if (VERBOSITY_LEVEL > VERBOSITY_LEVEL_NONE) {
        if (faults > 0) {
            printf(FMT_PARAM_NOT_OK_DIFF_S, label, tolerance, faults, maxdiff);
        }
        else {
            printf(FMT_PARAM_OK_DIFF_S, label, tolerance, maxdiff);
        }
    }
    // print a few values for visual comparison
    for (int i = 0; i < n; i++) {
        if (num_printed > PRINT_UP_TO) break;
        float diff = fabsf(a[i] - b[i]);
        if (diff > tolerance && VERBOSITY_LEVEL >= VERBOSITY_LEVEL_ERROR) {
            printf(FMT_TENSOR_NOT_OK_S, label, i, a[i], b[i], diff);
            num_printed++;
        }
        else if (faults == 0 && VERBOSITY_LEVEL >= VERBOSITY_LEVEL_DEBUG) {
            printf(FMT_TENSOR_OK_S, label, i, a[i], b[i]);
            num_printed++;
        }
    }
    return faults == 0;
}

int main(int argc, char *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int C = model.config.channels;
    int V = model.config.vocab_size;
    int maxT = model.config.max_seq_len;
    int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); return 1; }
    int state_header[256];
    fread(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file"); return 1; }
    if (state_header[1] != 1) { printf("Bad version in state file"); return 1; }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    int* x = (int*) malloc(B * T * sizeof(int));
    int* y = (int*) malloc(B * T * sizeof(int));
    float* expected_logits = (float*) malloc(B * T * V * sizeof(float));
    float* expected_loss = (float*) malloc(1 * sizeof(float));

    // read reference information from Python
    fread(x, sizeof(int), B*T, state_file);
    fread(y, sizeof(int), B*T, state_file);
    fread(expected_logits, sizeof(float), B*T*V, state_file);
    fread(expected_loss, sizeof(float), 1, state_file);
    fread(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fclose(state_file);

    // overall OK signal for the test
    int allok = 1;

    // let's do 10 training iterations, following the pytorch code
    float losses[10];
    for (int step = 0; step < 10; step++) {

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step == 0) {
            // error checking at step 0 for reference activations/gradients

            int logits_ok = check_tensor(model.acts.logits, expected_logits, B * T * V, "LOGITS", 8.9e-04f);
            allok = allok & logits_ok;

            // compare the achieved loss
            int loss_ok = check_tensor(&model.mean_loss, expected_loss, 1, "LOSS", 2.6e-05);
            allok = allok & loss_ok;

            // finally check all the gradients
            int gradoks[16];
            ParameterTensors grads = model.grads;
            gradoks[0] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "lnfb", 3.9e-05f);
            gradoks[1] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "lnfw", 2.2e-04f);
            gradoks[2] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "fcprojw", 8e-05f);
            gradoks[3] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L * C, "fcprojb", 2.4e-05f);
            gradoks[4] = check_tensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "fcw", 1.5e-04f);
            gradoks[5] = check_tensor(grads.fcb, expected_grads.fcb, L * 4 * C, "fcb", 3.9e-05f);
            gradoks[6] = check_tensor(grads.ln2w, expected_grads.ln2w, L * C, "ln2w", 1.7e-03f);
            gradoks[7] = check_tensor(grads.ln2b, expected_grads.ln2b, L * C, "ln2b", 1.4e-04f);
            gradoks[8] = check_tensor(grads.attprojw, expected_grads.attprojw, L * C * C, "attprojw", 6.3e-05f);
            gradoks[9] = check_tensor(grads.attprojb, expected_grads.attprojb, L * C, "attprojb", 5.1e-05f);
            gradoks[10] = check_tensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "qkvw", 9.5e-05);
            gradoks[11] = check_tensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "qkvb", 5.9e-05);
            gradoks[12] = check_tensor(grads.ln1w, expected_grads.ln1w, L * C, "ln1w", 6e-04f);
            gradoks[13] = check_tensor(grads.ln1b, expected_grads.ln1b, L * C, "ln1b", 2.6e-04f);
            gradoks[14] = check_tensor(grads.wte, expected_grads.wte, V * C, "wte", 3.6e-04f);
            gradoks[15] = check_tensor(grads.wpe, expected_grads.wpe, maxT * C, "wpe", 9.9e-06);
            for (int i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
            // compare the gradients on the parameters all at once
            if (!check_tensor(model.grads_memory, expected_grads_memory, model.num_parameters, "all grads", 1.7e-03)) {
                allok = 0;
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        losses[step] = model.mean_loss;
    }

    float expected_losses[10] = {
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615
    };

    int losses_ok = check_tensor(&losses[0], &expected_losses[0], 10, "LOSS", 3.9e-04);
    allok = allok & losses_ok;

    // final approval
    if (allok) {
        printf("overall okay: \033[42mOK\033[0m\n");
    } else {
        printf("overall okay: \033[41mNOT OK\033[0m\n");
    }

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    gpt2_free(&model);
    return 0;
}
