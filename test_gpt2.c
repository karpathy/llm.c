#define TESTING
#include "train_gpt2.c"

// poor man's tensor checker
int check_tensor(float* a, float* b, int n, const char* label, float tolerance) {
#define VERBOSITY_STATUS_ONLY 1
#define VERBOSITY_ERRORS 2
#define VERBOSITY_TRACE 3

    int verbosity = VERBOSITY_ERRORS;

    unsigned long long faults = 0;
    float maxdiff = 0.0f;
    // check the entire tensor without printing anything
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > tolerance) {
            faults++;
            if (diff > maxdiff) { maxdiff = diff; }
        }
    }

    const int PRINT_UP_TO = 5;

    int num_printed = 0;

    // print the final result

    if (verbosity >= VERBOSITY_STATUS_ONLY) {
        if (faults > 0) {
            printf("%s - \033[41mERROR\033[0m, faults = \033[41m%llu\033[0m @ \033[33m%e\033[0m, maxdiff=\033[33m%e\033[0m\n", label, faults, tolerance, maxdiff);
        }
        else {
            printf("%s - \033[32mOK\033[0m, maxdiff=%e\n", label, maxdiff);
        }
    }
    for (int i = 0; i < n; i++) {
        if (num_printed > PRINT_UP_TO) break;
        float diff = fabsf(a[i] - b[i]);
        if (diff > tolerance && verbosity >= VERBOSITY_ERRORS) {
            printf("NOT OK \033[31m%f8 %f8\033[0m, diff=\033[33m%e\033[0m\n", a[i], b[i], diff);
            num_printed++;
        }
        else if (faults == 0 && verbosity >= VERBOSITY_TRACE) {
            printf("OK %f8 %f8\n", a[i], b[i]);
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

            // at this point, target should be equal to expected_logits, let's compare
            int logits_ok = 1;
            for (int i=0; i<B*T*V; i++) {
                if(i < 3) {
                    printf("%f %f\n", expected_logits[i], model.acts.logits[i]);
                }
                if (fabsf(expected_logits[i] - model.acts.logits[i]) >= 1e-2) {
                    printf("MISMATCH AT INDEX %d: ", i);
                    printf("%f %f\n", expected_logits[i],model.acts.logits[i]);
                    logits_ok = 0;
                    break;
                }
            }
            if(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS)\n");
            allok = allok && logits_ok;

            // compare the achieved loss
            if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } else {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            int gradoks[16];
            ParameterTensors grads = model.grads;
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V*C, "dwte", 1e-6);
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe", 1e-6);
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w", 1e-6);
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b", 1e-6);
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw", 1e-6);
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb", 1e-6);
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw", 1e-6);
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb", 1e-6);
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w", 1e-6);
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b", 1e-6);
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw", 1e-6);
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb", 1e-6);
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw", 1e-6);
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb", 1e-6);
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw", 1e-6);
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb", 1e-6);
            for (int i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        float expected_loss = expected_losses[step];
        float actual_loss = model.mean_loss;
        float step_loss_diff = fabsf(expected_loss - actual_loss);
        int step_loss_ok = step_loss_diff < 1e-6;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms)", step, model.mean_loss, time_elapsed_s * 1000);
        if (!step_loss_ok) {
            printf(", diff=\033[41m%e\033[0m\n", step_loss_diff);
        }
        else {
            printf(", OK\n");
        }
    }

    // final judgement
    if (!allok) {
        printf("overall okay: \033[41mERROR!\033[0m\n");
    } else {
        printf("overall okay: OK\n");
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
