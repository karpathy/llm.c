# Makefile for building dev/cuda kernels

# Find nvcc
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

# Compiler flags
CFLAGS = -O3 --use_fast_math
MPI_PATHS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ 


%: %.cu
	$(NVCC) $(CFLAGS) $< -o $@ -lcublas

TARGETS = adamw attention_backward attention_forward classifier_fused crossentropy_forward crossentropy_softmax_backward encoder_backward encoder_forward gelu_forward layernorm_backward layernorm_forward matmul_backward matmul_backward_bias matmul_forward nccl_all_reduce residual_forward softmax_forward trimat_forward

all: $(TARGETS)

# Forward kernels

attention_forward: attention_forward.cu
classifier_fused: classifier_fused.cu
crossentropy_forward: crossentropy_forward.cu
encoder_forward: encoder_forward.cu
gelu_forward: gelu_forward.cu
layernorm_forward: layernorm_forward.cu
matmul_forward: matmul_forward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt
residual_forward: residual_forward.cu
softmax_forward: softmax_forward.cu
trimat_forward: trimat_forward.cu

# Backward kernels

attention_backward: attention_backward.cu
crossentropy_softmax_backward: crossentropy_softmax_backward.cu
encoder_backward: encoder_backward.cu
layernorm_backward: layernorm_backward.cu
matmul_backward_bias: matmul_backward_bias.cu
matmul_backward: matmul_backward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward -lcublas

# Update kernels

adamw: adamw.cu

# NCCL

nccl_all_reduce: nccl_all_reduce.cu
	$(NVCC) -lmpi -lnccl $(MPI_PATHS) nccl_all_reduce.cu -o nccl_all_reduce

run_all: all
	@for target in $(TARGETS); do \
		echo "\n========================================"; \
		echo "Running $$target ..."; \
		echo "========================================\n"; \
		./$$target; \
	done

clean:
	rm -f $(TARGETS)
