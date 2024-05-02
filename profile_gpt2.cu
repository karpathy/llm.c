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

int main() {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    printf("[System]\n");
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    #ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
    #endif

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");

    int B = 4;
    int T = 1024;
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);

    int* x = (int*)mallocCheck(B * T * sizeof(int));
    int* y = (int*)mallocCheck(B * T * sizeof(int));
    for(int  i = 0; i < B  * T; ++i) {
        x[i] = i % model.config.vocab_size;
        y[i] = i % model.config.vocab_size;
    }

    model.config.num_layers = 1;

    // do a training step
    gpt2_forward(&model, x, y, B, T);
    gpt2_zero_grad(&model);
    gpt2_backward(&model);
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
    cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
    // free
    gpt2_free(&model);

    #ifdef ENABLE_CUDNN
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    checkCudnnErr(cudnnDestroy(cudnn_handle));
    #endif
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));

    return 0;
}
