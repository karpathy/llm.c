/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void matmul_forward_ngc92(float* out,
    const float* inp, const float* weight, const float* bias,
    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    if (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            float result[LOOP_UNROLL];
            for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define NUM_KERNELS 2

void matmul_forward(int kernel_num,
    float* out,
    const float* inp, const float* weight, const float* bias,
    int B, int T, int C, int OC) {

    switch (kernel_num) {
        case 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            break;
        case 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


void validate_results_cpu(const float* device_result, const float* cpu_reference, const char* name, int num_elements, float tolerance);
float* make_random_float(size_t N);

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP
    int RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    float* out = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    float* grad_out = make_random_float(B * T * OC);
    float* grad_inp = make_random_float(B * T * C);
    float* grad_weight = make_random_float(OC * C);
    float* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    for (int kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Verifying kernel #%d\n", kernel_num);

        srand(137);

        float* kernel_out = make_random_float(B * T * OC);
        float* kernel_inp = make_random_float(B * T * C);
        float* kernel_weight = make_random_float(OC * C);
        float* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_num, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        free(kernel_out);
        free(kernel_inp);
        free(kernel_weight);
        free(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    for (int kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Running kernel #%d\n", kernel_num);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < RUNS; i++) {
            matmul_forward(kernel_num, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);

    free(grad_out);
    free(grad_inp);
    free(grad_weight);
    free(grad_bias);

    return 0;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

void validate_results_cpu(const float* kernel_result, const float* cpu_reference, const char* name, int num_elements, float tolerance) {
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        float t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            if (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    if (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}