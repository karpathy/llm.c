/*  Kernels for fused forward/backward classifier part
This fuses softmax, crossentropy, and logit gradients into a single pass, so we don't have to write unnecessary
(B, T, V) tensors. Such an operation is only possible if `dloss` can be known beforehand, which doesn't seem like
much of a restriction: In pretraining, it is just a constant 1/batch_size tensor, for fine-tuning we might zero
out the input prompt, but that is known in advance.

Compile example:
nvcc -O3 --use_fast_math classifier_fused.cu -o classifier_fused
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= sum;
        }
    }
}


void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs, const int* targets,
                                       int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] = (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};
namespace cg = cooperative_groups;
__device__ SoftmaxParams prepare_softmax(cg::thread_block_tile<32>& warp,
                                         int idx, const float* inp, int V, int P) {
    // one row of inp, i.e. inp[idx, :] of shape (V,)
    const float* x = inp + idx * P;

    float maxval = -INFINITY;
    float sumval = 0.0f;

    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float v = x[i];
        float old_maxval = maxval;
        maxval = fmaxf(maxval, v);
        sumval *= expf((old_maxval - maxval));
        sumval += expf(v - maxval);
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf((maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    return SoftmaxParams{norm, global_maxval};
}


__global__ void fused_classifier_kernel(float* dlogits, float* losses,
                             const float* logits, const float* dlosses, const int* targets,
                             int B, int T, int V, int P) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= B * T) {
        return;
    }

    // local indices
    int b = idx / T;
    int t = idx % T;

    auto sp = prepare_softmax(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update.
    // single-threaded
    if(warp.thread_rank() == 0) {
        int ix = targets[b * T + t];
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[b * T + t] = -logf(prob);
    }

    // calculate all the gradients
    for (int i = warp.thread_rank(); i < V; i += warp.size()) {
        float prob = expf(logits[idx * P + i] - sp.Offset) * sp.Scale;
        float* dlogits_bt = dlogits + b * T * P + t * P;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = prob;
        float indicator = i == ix ? 1.0f : 0.0f;
        dlogits_bt[i] = (p - indicator) * dloss;
    }

}

// ----------------------------------------------------------------------------
// kernel launcher

void fused_classifier1(float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel<<<grid_size, block_size>>>(dlogits, losses, logits, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

void fused_classifier(int kernel_num, float* dlogits, float* losses,
                      const float* logits, const float* dlosses, const int* targets,
                      int B, int T, int V, int P, int block_size) {
    switch (kernel_num) {
        case 1:
            fused_classifier1(dlogits, losses, logits, dlosses, targets, B, T, V, P, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int V = 50257;
    // padded size
    int P = (V + 63) & ~63;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* logits = make_random_float_01(B * T * V);
    float* probs = (float*)malloc(B * T * V * sizeof(float));
    float* dlogits = (float*)malloc(B * T * V * sizeof(float));
    float* losses = (float*)malloc(B * T * sizeof(float));
    const float* dlosses = make_random_float(B * T);
    const int* targets = make_random_int(B * T, V);

    // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
    // and the tests are not actually meaningful.
    const int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k) {
        for(int j = 0; j < B * T; ++j) {
            logits[j * V +  outliers[j*3 + k]] *= 20;
        }
    }

    // move to GPU
    float* d_logits;
    float* d_dlogits;
    float* d_dlogits_no_pad;
    float* d_losses;
    float* d_dlosses;
    int* d_targets;

    cudaCheck(cudaMalloc(&d_dlogits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_logits, B * T * P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits_no_pad, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_losses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));

    // move to GPU
    cudaCheck(cudaMemset(d_logits, 0xff, B * T * P * sizeof(float)));
    cudaCheck(cudaMemcpy2D(d_logits, P * sizeof(float), logits, V * sizeof(float), V * sizeof(float), B * T, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    softmax_forward_cpu(probs, logits, B * T, V);
    crossentropy_forward_cpu(losses, probs, targets, B, T, V);
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        fused_classifier(kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets, B, T, V, P, block_size);
        validate_result(d_losses, losses, "losses", B * T, 1e-4f);
        // undo the padding before we can check for correctness
        cudaCheck(cudaMemcpy2D(d_dlogits_no_pad, V * sizeof(float), d_dlogits, P * sizeof(float), V * sizeof(float), B * T, cudaMemcpyDeviceToDevice));
        validate_result(d_dlogits_no_pad, dlogits, "dlogits", B * T * V, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, fused_classifier,
                                              kernel_num, d_dlogits, d_losses, d_logits, d_dlosses, d_targets,
                                              B, T, V, P, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free((void*)logits);
    free(probs);
    free(dlogits);
    free(losses);
    free((void*)dlosses);
    free((void*)targets);

    cudaCheck(cudaFree(d_dlogits));
    cudaCheck(cudaFree(d_losses));
    cudaCheck(cudaFree(d_logits));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_targets));

    return 0;
}