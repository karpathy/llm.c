#define TESTING
#include "train_gpt2_fp32.cu"

// poor man's tensor checker
int check_tensor(float *a, float *b, int n, const char* label) {
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
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    enable_tf32 = 0; // NOTE: disable TF32 for testing!!!
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    // int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    assert(0 <= T && T <= maxT);
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads; // will be read from file (from PyTorch)
    ParameterTensors calculated_grads; // will be calculated by us
    float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes, 0);
    float* calculated_grads_memory = malloc_and_point_parameters(&calculated_grads, model.param_sizes, 0);

    // inputs and expected outputs, only used for error checking
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    float* expected_logits = (float*) mallocCheck(B * T * V * sizeof(float));
    float* expected_loss = (float*) mallocCheck(1 * sizeof(float));

    // read reference information from Python
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    freadCheck(expected_loss, sizeof(float), 1, state_file);
    freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    int allok = 1;

    // First, do target-free forward pass to validate logits
    gpt2_forward(&model, x, NULL, B, T);
    // at this point, target should be equal to expected_logits, let's compare
    // copy logits to CPU so we can compare them
    float* logits_cpu = (float*)mallocCheck(B * T * Vp * sizeof(float));
    cudaCheck(cudaMemcpy(logits_cpu, model.acts.output, B * T * Vp * sizeof(float), cudaMemcpyDeviceToHost));

    // compare the output logits from the forward pass
    // also careful that we don't access and compare the padded columns of logits
    int logits_ok = 1;
    float max_diff = 0.0f;
    for (int bt = 0; bt < B*T; bt++) {
        for (int v = 0; v < V; v++) {
            int i = bt * Vp + v; // linearized index
            if (i < 10) {
                printf("%f, %f\n", expected_logits[i], logits_cpu[i]);
            }
            float diff = fabsf(expected_logits[bt*V + v] - logits_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff >= 1e-2f) {
                printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f\n", expected_logits[bt*V + v], logits_cpu[i]);
                logits_ok = 0;
                bt = B*T; // to break out of both loops
                break;
            }
        }
    }
    allok = allok && logits_ok;
    if(!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS)\n");

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
            // error checking at step 0 for reference activations
            free(logits_cpu);

            // compare the achieved loss
            if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } else {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // and now compare the gradients on the parameters
            // cudaMemcpy(calculated_grads.lnfw, model.grads.lnfw, C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.lnfb, model.grads.lnfb, C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.fcprojw, model.grads.fcprojw, L * C * 4*C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.fcprojb, model.grads.fcprojb, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.fcw, model.grads.fcw, L * 4*C * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.fcb, model.grads.fcb, L * 4*C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.ln2w, model.grads.ln2w, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.ln2b, model.grads.ln2b, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.attprojw, model.grads.attprojw, L * C * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.attprojb, model.grads.attprojb, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.qkvw, model.grads.qkvw, L * 3*C * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.qkvb, model.grads.qkvb, L * 3*C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.ln1w, model.grads.ln1w, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.ln1b, model.grads.ln1b, L * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.wte, model.grads.wte, V * C * sizeof(float), cudaMemcpyDeviceToHost);
            // cudaMemcpy(calculated_grads.wpe, model.grads.wpe, maxT * C * sizeof(float), cudaMemcpyDeviceToHost);
            // check_tensor(calculated_grads.lnfb, expected_grads.lnfb, C, "lnfb");
            // check_tensor(calculated_grads.lnfw, expected_grads.lnfw, C, "lnfw");
            // check_tensor(calculated_grads.fcprojw, expected_grads.fcprojw, L * C * 4*C, "fcprojw");
            // check_tensor(calculated_grads.fcprojb, expected_grads.fcprojb, L * C, "fcprojb");
            // check_tensor(calculated_grads.fcw, expected_grads.fcw, L * 4*C * C, "fcw");
            // check_tensor(calculated_grads.fcb, expected_grads.fcb, L * 4*C, "fcb");
            // check_tensor(calculated_grads.ln2w, expected_grads.ln2w, L * C, "ln2w");
            // check_tensor(calculated_grads.ln2b, expected_grads.ln2b, L * C, "ln2b");
            // check_tensor(calculated_grads.attprojw, expected_grads.attprojw, L * C * C, "attprojw");
            // check_tensor(calculated_grads.attprojb, expected_grads.attprojb, L * C, "attprojb");
            // check_tensor(calculated_grads.qkvw, expected_grads.qkvw, L * 3*C * C, "qkvw");
            // check_tensor(calculated_grads.qkvb, expected_grads.qkvb, L * 3*C, "qkvb");
            // check_tensor(calculated_grads.ln1w, expected_grads.ln1w, L * C, "ln1w");
            // check_tensor(calculated_grads.ln1b, expected_grads.ln1b, L * C, "ln1b");
            // check_tensor(calculated_grads.wte, expected_grads.wte, V * C, "wte");
            // check_tensor(calculated_grads.wpe, expected_grads.wpe, maxT * C, "wpe");

            // compare the gradients ona the parameters all at once
            cudaMemcpy(calculated_grads_memory, model.grads_memory, model.num_parameters * sizeof(float), cudaMemcpyDeviceToHost);
            check_tensor(calculated_grads_memory, expected_grads_memory, model.num_parameters, "grads");
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        losses[step] = model.mean_loss;
    }

    // expected losses are as follows, from Python
    float expected_losses[10] = {
        5.270007133483887f,
        4.059706687927246f,
        3.3751230239868164f,
        2.8007826805114746f,
        2.315382242202759f,
        1.8490285873413086f,
        1.3946564197540283f,
        0.9991465210914612f,
        0.6240804195404053f,
        0.37651097774505615f
    };

    // compare
    for (int i = 0; i < 10; i++) {
        if (fabsf(losses[i] - expected_losses[i]) >= 1e-2) {
            printf("LOSS MISMATCH AT STEP %d: %f %f\n", i, losses[i], expected_losses[i]);
            allok = 0;
        } else {
            printf("loss ok at step %d: %f %f\n", i, losses[i], expected_losses[i]);
        }
    }

    // final approval
    printf("overall okay: %d\n", allok);

    // free everything
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    free(calculated_grads_memory);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}