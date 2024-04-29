# Makefile for building dev/cuda kernels
# Collects all the make commands in one file but each file also
# has the compile and run commands in the header comments section.

# Find nvcc (NVIDIA CUDA compiler)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

# Compiler flags
CFLAGS = -O3 --use_fast_math
MPI_PATHS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/

# Default rule for our CUDA files
%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

# Build all targets
TARGETS = adamw attention_backward attention_forward classifier_fused crossentropy_forward crossentropy_softmax_backward encoder_backward encoder_forward gelu_forward layernorm_backward layernorm_forward matmul_backward matmul_backward_bias matmul_forward nccl_all_reduce residual_forward softmax_forward trimat_forward
all: $(TARGETS)

# matmul fwd/bwd also uses OpenMP (optionally) and cuBLASLt libs
matmul_forward: matmul_forward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

matmul_backward: matmul_backward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward -lcublas

# NCCL communication kernels
nccl_all_reduce: nccl_all_reduce.cu
	$(NVCC) -lmpi -lnccl $(MPI_PATHS) nccl_all_reduce.cu -o nccl_all_reduce

# Run all targets
run_all: all
	@for target in $(TARGETS); do \
		echo "\n========================================"; \
		echo "Running $$target ..."; \
		echo "========================================\n"; \
		./$$target; \
	done

# Clean up
clean:
	rm -f $(TARGETS)
