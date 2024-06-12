# dev/cuda

This directory is scratch space for developing various versions of the needed CUDA kernels. Each file develops a kernel, and usually multiple versions of that kernel that could have different running times and of different code or time complexity.

See the top of each file for how to compile and run the kernel. Alternatively, the commands are also all grouped in the `Makefile` in this directory for convenience.

For example, we can look at the top of `layernorm_forward.cu` to build the forward pass kernels for the LayerNorm:

```bash
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward
```

or simply

```bash
make layernorm_forward
```

The comments at the top then document the different versions of this kernel available, usually these are in increasing complexity and decreasing running times. For example, inspecting the comments in the file on top, the most naive kernel we can then run as:

```bash
./layernorm_forward 1
```

You'll see that this first forwards the reference code on the CPU, then it runs kernel 1 on the GPU, compares the results to check for correctness, and then runs a number of configurations of this kernel (most often and most notably the block size), to time the kernel in these launch configurations. We can then run one of the faster kernels (kernel 4) instead:

```bash
./layernorm_forward 4
```

You'll see that this matches all the CPU results but runs much much faster. The typical process from here on is we copy paste the kernel that ran fastest, adjust it manually (e.g. to hardcode the best block size) and drop it into the training code file, e.g. `train_gpt2.cu`.

To add a new version of a kernel, add the kernel to the corresponding file and adjust the docs. To add a new kernel, add the new file and adjust the Makefile. Run `make clean` to clean up binaries from your directory.

If you do not have a GPU or is having trouble with CUDA dependencies, you can run the benchmarks on the [Modal platform](http://modal.com). For example, to run the benchmark for the attention forward pass on an A100 GPU with 80GB of memory, you can run the following command:

```bash
GPU_MEM=80 modal run benchmark_on_modal.py --compile-command "nvcc -O3 --use_fast_math attention_forward.cu -o attention_forward -lcublas" --run-command "./attention_forward 1"
```
