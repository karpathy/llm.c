/*
This code is a convenience tool for profiling the CUDA kernels in the training
loop of train_gpt2.cu. Compile:

make profile_gpt2cu NO_MULTI_GPU=1

And then e.g. use ncu from NVIDIA. The CLI docs for example:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/

TLDR run like:

sudo ncu --set full --import-source yes -o profile -f ./profile_gpt2cu

This:
- `--set full` means we'll collect A LOT of metrics. take out for less
- `--import-source yes` means we'll get the source code in the profile
- `-o profile` writes the results into file profile.ncu-rep
- `-f` forces overwrite of the profile.ncu-rep file
- `./profile_gpt2cu` is the executable we want to profile

This writes results into profile.ncu-rep output file.
You can open this up in NVIDIA Nsight Compute UI.
For example, I have NVIDIA Nsight Compute installed on my Mac, and I rsync
the profile.ncu-rep from a cloud box to local to pretty view.
*/

#define TESTING
#include "train_gpt2.cu"

int main(int argc, char *argv[]) {
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    int num_processes = -1;  // doesn't matter when using MPI
    int process_rank = -1;  // doesn't matter when using MPI
    int gpus_per_node = -1;  // doesn't matter when using MPI
    char server_ip[256] = "";  // doesn't matter when using MPI
    char fs_path[256] = "";  // doesn't matter when using MPI
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(true, true);

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");

    int B = 24; // if program OOMs decrease this number, e.g. all the way down to 4 or etc
    int T = 1024; // if even that OOMs move on to this one. keep them nice and powers of 2
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    for(int  i = 0; i < B  * T; ++i) {
        x[i] = i % model.config.vocab_size;
        y[i] = i % model.config.vocab_size;
    }

    // override number of layers to 1 because all layers repeat the same kernels, only profile once
    model.config.num_layers = 1;
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    // do a training step
    gpt2_forward(&model, x, B, T);
    gpt2_backward_and_reduce(&model, x, y, 1, 0);
    float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
    float grad_scale = (grad_norm > 1.0f) ? 1.0f / grad_norm : 1.0f;
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, grad_scale, 1, &multi_gpu_config);
    cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings

    // free
    gpt2_free(&model);
    common_free(model);
    return 0;
}
