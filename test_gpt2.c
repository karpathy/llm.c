#define TESTING
#include "train_gpt2.c"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label) {
    int print_upto = 5;
    int ok = 1;
    float maxdiff = 0.0f;
    float tol = 2e-2;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        float diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        if (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if (i < print_upto) {
            if (diff <= tol) {
                if (i < print_upto) { printf("OK "); }
            } else {
                if (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    if (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } else {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    return ok;
}

struct timespec clock_gettime_monotonic()
{ 
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

double timespec_difference_ms(struct timespec start, struct timespec end)
{
    double ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) * 0.000001;
    return ms;
}

int main(int argc, char *argv[]) {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
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

        struct timespec t0 = clock_gettime_monotonic();
        gpt2_forward(&model, x, y, B, T);
        struct timespec t1 = clock_gettime_monotonic();
        gpt2_zero_grad(&model);
        struct timespec t2 = clock_gettime_monotonic();
        gpt2_backward(&model);
        struct timespec t3 = clock_gettime_monotonic();

        if (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            int logits_ok = 1;
            float* calculated_logits = model.acts.logits;
            float max_diff = 0.0f;
            for (int bt = 0; bt < B*T; bt++) {
                for (int v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    int i = bt * Vp + v; // linearized index, using Vp
                    if (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    float diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    if (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        break;
                    }
                }
            }
            if(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
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
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V*C, "dwte");
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe");
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w");
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b");
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw");
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb");
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw");
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb");
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w");
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b");
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw");
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb");
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw");
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb");
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
            for (int i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        struct timespec t4 = clock_gettime_monotonic();
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);
        struct timespec t5 = clock_gettime_monotonic();

        double forward_ms = timespec_difference_ms(t0, t1);
        double zero_ms = timespec_difference_ms(t1, t2);
        double backward_ms = timespec_difference_ms(t2, t3);
        double update_ms = timespec_difference_ms(t4, t5);
        double total_ms = forward_ms + zero_ms + backward_ms + update_ms;

        // compare the losses
        float expected_loss = expected_losses[step];
        float actual_loss = model.mean_loss;
        int step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f OK = %d (%5.0f ms = forward %4.0f ms zero_grad %2.0f ms backward %4.0f ms update %4.0f ms)\n", 
            step, model.mean_loss, step_loss_ok, 
            total_ms, forward_ms, zero_ms, backward_ms, update_ms);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    gpt2_free(&model);
    return 0;
}
