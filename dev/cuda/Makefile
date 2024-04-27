NVCC := $(shell which nvcc 2>/dev/null)

ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

TARGETS = adamw attention_backward attention_forward classifier_fused crossentropy_forward crossentropy_softmax_backward encoder_backward encoder_forward gelu_forward layernorm_backward layernorm_forward matmul_backward matmul_backward_bias matmul_forward nccl_all_reduce residual_forward softmax_forward trimat_forward

CFLAGS = -O3 --use_fast_math

all: $(TARGETS)

adamw: adamw.cu
	$(NVCC) $(CFLAGS) adamw.cu -o adamw

attention_backward: attention_backward.cu
	$(NVCC) $(CFLAGS) attention_backward.cu -o attention_backward -lcublas

attention_forward: attention_forward.cu
	$(NVCC) $(CFLAGS) attention_forward.cu -o attention_forward -lcublas

classifier_fused: classifier_fused.cu
	$(NVCC) $(CFLAGS) classifier_fused.cu -o classifier_fused

crossentropy_forward: crossentropy_forward.cu
	$(NVCC) $(CFLAGS) crossentropy_forward.cu -o crossentropy_forward

crossentropy_softmax_backward: crossentropy_softmax_backward.cu
	$(NVCC) $(CFLAGS) crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

encoder_backward: encoder_backward.cu
	$(NVCC) $(CFLAGS) encoder_backward.cu -o encoder_backward

encoder_forward: encoder_forward.cu
	$(NVCC) $(CFLAGS) encoder_forward.cu -o encoder_forward

gelu_forward: gelu_forward.cu
	$(NVCC) $(CFLAGS) gelu_forward.cu -o gelu_forward

layernorm_backward: layernorm_backward.cu
	$(NVCC) $(CFLAGS) layernorm_backward.cu -o layernorm_backward

layernorm_forward: layernorm_forward.cu
	$(NVCC) $(CFLAGS) layernorm_forward.cu -o layernorm_forward

matmul_backward: matmul_backward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward -lcublas

matmul_backward_bias: matmul_backward_bias.cu
	$(NVCC) $(CFLAGS) matmul_backward_bias.cu -o matmul_backward_bias

matmul_forward: matmul_forward.cu
	$(NVCC) $(CFLAGS) -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

nccl_all_reduce: nccl_all_reduce.cu
	$(NVCC) -lmpi -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ nccl_all_reduce.cu -o nccl_all_reduce

residual_forward: residual_forward.cu
	$(NVCC) $(CFLAGS) residual_forward.cu -o residual_forward

softmax_forward: softmax_forward.cu
	$(NVCC) $(CFLAGS) softmax_forward.cu -o softmax_forward

trimat_forward: trimat_forward.cu
	$(NVCC) $(CFLAGS) trimat_forward.cu -o trimat_forward -lcublas

clean:
	rm -f $(TARGETS)

run_all: all
	./adamw
	./attention_backward
	./attention_forward
	./classifier_fused
	./crossentropy_forward
	./crossentropy_softmax_backward
	./encoder_backward
	./encoder_forward
	./gelu_forward
	./layernorm_backward
	./layernorm_forward
	./matmul_backward
	./matmul_backward_bias
	./matmul_forward
	./nccl_all_reduce
	./residual_forward
	./softmax_forward
	./trimat_forward
