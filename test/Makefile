# Makefile for building the unit tests

# Find nvcc (NVIDIA CUDA compiler)
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

# Compiler flags
CFLAGS = -O3 --use_fast_math
NVCCFLAGS = -lcublas -lcublasLt -std=c++17 -I..
MPI_PATHS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/

# Default rule for our CUDA files
%: %.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $< -o $@

# Build all targets
TARGETS = bits
all: $(TARGETS)

# Individual targets
bits: bits.cu

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
