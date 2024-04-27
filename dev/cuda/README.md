# dev/cuda

This directory is scratch space for developing various versions of the needed CUDA kernels. Each file develops a kernel, see the top of each file for instructions on how to compile and run each one using the `nvcc` compiler.

An alternative to invoking `nvcc` manually is to use `make` with the accompanying `Makefile` in this directory. Each kernel has its own `make` build target, invoking `make` for the target builds the associated binary. 

For example, `make gelu_forward` builds the forward GELU kernel, creating a binary that can be executed by running `./gelu_forward`. `make` or `make all` builds all the kernels in this directory. To delete all binary build targets, run `make clean`.
