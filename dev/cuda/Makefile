# Makefile for building dev/cuda kernels
# Collects all the make commands in one file but each file also
# has the compile and run commands in the header comments section.

# Find nvcc (NVIDIA CUDA compiler)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

ifneq ($(CI),true) # if not in CI, then use the GPU query
  ifndef GPU_COMPUTE_CAPABILITY # set to defaults if: make GPU_COMPUTE_CAPABILITY= 
    GPU_COMPUTE_CAPABILITY = $(shell __nvcc_device_query) # assume if NVCC is present, then this likely is too
    GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
  endif
endif

# Compiler flags
ifeq ($(GPU_COMPUTE_CAPABILITY),) # set to defaults if: make GPU_COMPUTE_CAPABILITY= 
  CFLAGS = -O3 --use_fast_math 
else
  CFLAGS = -O3 --use_fast_math --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

NVCCFLAGS = -lcublas -lcublasLt -std=c++17
MPI_PATHS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/

# Default rule for our CUDA files
%: %.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $< -o $@

# Build all targets
TARGETS = adamw attention_backward attention_forward classifier_fused crossentropy_forward crossentropy_softmax_backward encoder_backward encoder_forward gelu_backward gelu_forward layernorm_backward layernorm_forward matmul_backward matmul_backward_bias matmul_forward nccl_all_reduce residual_forward softmax_forward trimat_forward fused_residual_forward  global_norm
all: $(TARGETS)
all_ptx:  $(TARGETS:%=%.ptx)
all_sass: $(TARGETS:%=%.sass)

# Individual targets: forward pass
attention_forward: attention_forward.cu
classifier_fused: classifier_fused.cu
crossentropy_forward: crossentropy_forward.cu
encoder_forward: encoder_forward.cu
gelu_forward: gelu_forward.cu
layernorm_forward: layernorm_forward.cu
fused_residual_forward: fused_residual_forward.cu
residual_forward: residual_forward.cu
softmax_forward: softmax_forward.cu
trimat_forward: trimat_forward.cu
# matmul fwd/bwd also uses OpenMP (optionally) and cuBLASLt libs
matmul_forward: matmul_forward.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward

# Individual targets: backward pass
attention_backward: attention_backward.cu
crossentropy_softmax_backward: crossentropy_softmax_backward.cu
encoder_backward: encoder_backward.cu
gelu_backward: gelu_backward.cu
layernorm_backward: layernorm_backward.cu
matmul_backward_bias: matmul_backward_bias.cu
matmul_backward: matmul_backward.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

# Update kernels
adamw: adamw.cu
global_norm: global_norm.cu

# NCCL communication kernels
nccl_all_reduce: nccl_all_reduce.cu
	$(NVCC) -lmpi -lnccl $(NVCCFLAGS) $(MPI_PATHS) nccl_all_reduce.cu -o nccl_all_reduce

# Generate PTX using cuobjdump
%.ptx: %
	cuobjdump --dump-ptx $< > $@

# Generate SASS using cuobjdump
%.sass: %
	cuobjdump --dump-sass $< > $@

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
	rm -f $(TARGETS) *.ptx *.sass
