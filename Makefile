CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Find nvcc
NVCC := $(shell which nvcc 2>/dev/null)

# NVCC flags
NVCC_FLAGS = -O3 --use_fast_math
NVCC_LDFLAGS = -lcublas -lcublasLt

# Function to test if the compiler accepts a given flag.
define check_and_add_flag
    $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
    ifeq ($(FLAG_SUPPORTED),yes)
        CFLAGS += $(1)
    endif
endef

# Check each flag and add it if supported
$(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
# First, check if NO_OMP is set to 1, if not, proceed with the OpenMP checks
ifeq ($(NO_OMP), 1)
  $(info OpenMP is manually disabled)
else
  # Detect if running on macOS or Linux
  ifeq ($(shell uname), Darwin)
    # Check for Homebrew's libomp installation in different common directories
    ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
      # macOS with Homebrew on ARM (Apple Silicon)
      CFLAGS += -Xclang -fopenmp -DOMP
      LDFLAGS += -L/opt/homebrew/opt/libomp/lib
      LDLIBS += -lomp
      INCLUDES += -I/opt/homebrew/opt/libomp/include
      $(info OpenMP found, compiling with OpenMP support)
    else ifeq ($(shell [ -d /usr/local/opt/libomp/lib ] && echo "exists"), exists)
      # macOS with Homebrew on Intel
      CFLAGS += -Xclang -fopenmp -DOMP
      LDFLAGS += -L/usr/local/opt/libomp/lib
      LDLIBS += -lomp
      INCLUDES += -I/usr/local/opt/libomp/include
      $(info OpenMP found, compiling with OpenMP support)
    else
      $(warning OpenMP not found, skipping OpenMP support)
    endif
  else
    # Check for OpenMP support in GCC or Clang on Linux
    ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
      CFLAGS += -fopenmp -DOMP
      LDLIBS += -lgomp
      $(info OpenMP found, compiling with OpenMP support)
    else
      $(warning OpenMP not found, skipping OpenMP support)
    endif
  endif
endif

# Check if OpenMPI is available
# OpenMPI enables DDP training strategy
# e.g. on MacOS: brew install open-mpi
# e.g. on Ubuntu: sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
# later, run the program with number of processes required, e.g.: mpirun -np 4 ./gpt2
ifeq ($(NO_MPI), 1)
  $(info OpenMPI is manually disabled)
else
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S), Darwin)
    ifeq ($(shell [ -d /opt/homebrew/opt/open-mpi/lib ] && echo "exists"), exists)
      # macOS with Homebrew on ARM (Apple Silicon)
      CFLAGS += -DOMPI
      LDFLAGS += -L/opt/homebrew/opt/open-mpi/lib
      LDLIBS += -lmpi
      INCLUDES += -I/opt/homebrew/opt/open-mpi/include
      $(info OpenMPI found, compiling with OpenMPI support)
    else
      $(warning OpenMPI not found, skipping OpenMPI support)
    endif
  else ifeq ($(UNAME_S), Linux)
    ifeq ($(shell [ -d /usr/lib/x86_64-linux-gnu/openmpi/lib ] && echo "exists"), exists)
      # Linux with apt-get installed OpenMPI
      CFLAGS += -DOMPI
      LDFLAGS += -L/usr/lib/x86_64-linux-gnu/openmpi/lib
      LDLIBS += -lmpi
      INCLUDES += -I/usr/lib/x86_64-linux-gnu/openmpi/include
      $(info OpenMPI found, compiling with OpenMPI support)
    else
      $(warning OpenMPI not found, skipping OpenMPI support)
    endif
  endif
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu

# Add targets
TARGETS = train_gpt2 test_gpt2

# Conditional inclusion of CUDA targets
ifeq ($(NVCC),)
    $(info nvcc not found, skipping CUDA builds)
else
    $(info nvcc found, including CUDA builds)
    TARGETS += train_gpt2cu test_gpt2cu
endif

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

# possibly may want to disable warnings? e.g. append -Xcompiler -Wno-unused-result
train_gpt2cu: train_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@

test_gpt2cu: test_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@

profile_gpt2cu: profile_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo $< $(NVCC_LDFLAGS) -o $@

clean:
	rm -f train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu

