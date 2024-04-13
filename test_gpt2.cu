#define TESTING
#include <string.h>
#include "train_gpt2.cu"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, char* label) {
    int print_upto = 5;
    int ok = 1;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) <= 1e-2) {
            if (i < print_upto) { printf("OK "); }
        } else {
            if (i < print_upto) { printf("NOT OK "); }
            ok = 0;
        }
        if (i < print_upto) { printf("%f %f\n", a[i], b[i]); }
    }
    // print the final result
    if (ok) {
        printf("TENSOR OK\n");
    } else {
        printf("TENSOR NOT OK\n");
    }
    return ok;
}

int check_decoder() {
    char decoder[GPT2_NUM_TOKENS][GPT2_MAX_TOKEN_LEN];
    gpt2_load_decoder("data/decode_gpt2.txt", decoder);

    static int tokens[] = {50256, 16773, 18162, 21986, 11, 198, 13681, 263, 23875, 198, 3152, 262, 11773, 2910, 198, 1169, 6002, 6386, 2583, 286, 262, 11858, 198, 20424, 428, 3135, 7596, 995, 3675, 13, 198, 40, 481, 407, 736, 17903, 11, 329, 703, 6029, 706, 4082, 198, 42826, 1028, 1128, 633, 263, 11, 198, 10594, 407, 198, 2704, 454, 680, 1028, 262, 1027, 28860, 286, 198, 3237, 323};
    static char* expected[] = {"<|endoftext|>", "Come", " Running", " Away", ",", "\n", "Great", "er", " conquer", "\n", "With", " the", " Imperial", " blood", "\n", "the", " heav", "iest", " host", " of", " the", " gods", "\n", "into", " this", " wond", "rous", " world", " beyond", ".", "\n", "I", " will", " not", " back", " thee", ",", " for", " how", " sweet", " after", " birth", "\n", "Netflix", " against", " rep", "ound", "er", ",", "\n", "will", " not", "\n", "fl", "our", "ish", " against", " the", " ear", "locks", " of", "\n", "All", "ay"};
    int num = sizeof(tokens) / sizeof(tokens[0]);

    int ok = 1;
    for (int i = 0; i < num; ++i) {
        if (strcmp(decoder[tokens[i]], expected[i]) != 0) {
            printf("MISMATCH AT INDEX %d: %s %s\n", i, decoder[tokens[i]], expected[i]);
            ok = 0;
        }
    }
    if (ok) {
        printf("Decoder OK\n");
    } else {
        printf("Decoder NOT OK\n");
    }
    return ok;
}

int main(int argc, char *argv[]) {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("[System]\n");
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    enable_tf32 = 0; // NOTE: disable TF32 for testing!!!
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    int decoder_ok = check_decoder();
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    int C = model.config.channels;
    int V = model.config.vocab_size;
    int maxT = model.config.max_seq_len;
    int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); exit(1); }
    int state_header[256];
    fread(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file"); exit(1); }
    if (state_header[1] != 1) { printf("Bad version in state file"); exit(1); }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes, 0);

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
    int allok = decoder_ok;

    // let's do 10 training iterations, following the pytorch code
    float losses[10];
    for (int step = 0; step < 10; step++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        gpt2_forward(&model, x, y, B, T);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step == 0) {
            // error checking at step 0 for reference activations

            // at this point, target should be equal to expected_logits, let's compare
            // copy logits to CPU so we can compare them
            float* logits_cpu = (float*) malloc(B * T * V * sizeof(float));
            cudaMemcpy(logits_cpu, model.acts.logits, B * T * V * sizeof(float), cudaMemcpyDeviceToHost);
            int logits_ok = 1;
            for (int i=0; i<B*T*V; i++) {
                if(i < 3) {
                    printf("%f %f\n", expected_logits[i], logits_cpu[i]);
                }
                if (fabsf(expected_logits[i] - logits_cpu[i]) >= 1e-2) {
                    printf("MISMATCH AT INDEX %d: ", i);
                    printf("%f %f\n", expected_logits[i],logits_cpu[i]);
                    logits_ok = 0;
                    break;
                }
            }
            if(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS)\n");
            allok = allok && logits_ok;
            free(logits_cpu);

            // compare the achieved loss
            if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } else {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }
        }
    }

    printf("overall okay: %d\n", allok);

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    gpt2_free(&model);
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));

    return 0;
}
