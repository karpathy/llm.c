CC ?= gcc
# example: make test_dataloader TEST_CFLAGS=-fsanitize=address -fno-omit-frame-pointer 
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -g
CFLAGS += $(TEST_CFLAGS)
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native

# Find nvcc
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# NVCC flags
# -t=0 is short for --threads, 0 = number of CPUs on the machine
NVCC_FLAGS = -O3 -t=0 --use_fast_math -std=c++17
NVCC_LDFLAGS = -lcublas -lcublasLt
NVCC_INCLUDES =
NVCC_LDLIBS =
NVCC_CUDNN =
# By default we don't build with cudnn because it blows up compile time from a few seconds to ~minute
USE_CUDNN ?= 0

# We will place .o files in the `build` directory (create it if it doesn't exist)
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o

# Function to check if a file exists in the PATH
define file_exists_in_path
  $(which $(1) 2>/dev/null)
endef

ifneq ($(CI),true) # if not in CI, then use the GPU query
  ifndef GPU_COMPUTE_CAPABILITY # set to defaults if: make GPU_COMPUTE_CAPABILITY=
    ifneq ($(call file_exists_in_path, __nvcc_device_query),)
      GPU_COMPUTE_CAPABILITY = $(shell __nvcc_device_query)
      GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
    endif
  endif
endif

# set to defaults if - make GPU_COMPUTE_CAPABILITY= otherwise use the compute capability detected above
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# autodect a lot of various supports on current platform
$(info ---------------------------------------------)

NVCC := $(shell which nvcc 2>/dev/null)

# Check and include cudnn if available
# You can override the path to cudnn frontend by setting CUDNN_FRONTEND_PATH on the make command line
# By default, we look for it in HOME/cudnn-frontend/include and ./cudnn-frontend/include
# Refer to the README for cuDNN install instructions
ifeq ($(USE_CUDNN), 1)
  ifeq ($(shell [ -d $(HOME)/cudnn-frontend/include ] && echo "exists"), exists)
    $(info ✓ cuDNN found, will run with flash-attention)
    CUDNN_FRONTEND_PATH ?= $(HOME)/cudnn-frontend/include
  else ifeq ($(shell [ -d cudnn-frontend/include ] && echo "exists"), exists)
    $(info ✓ cuDNN found, will run with flash-attention)
    CUDNN_FRONTEND_PATH ?= cudnn-frontend/include
  else
    $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
  endif
  NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
  NVCC_LDFLAGS += -lcudnn
  NVCC_FLAGS += -DENABLE_CUDNN
  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
else
  $(info → cuDNN is manually disabled by default, run make with `USE_CUDNN=1` to try to enable)
endif

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
  ifneq ($(OS), Windows_NT)
    # Check for OpenMP support in GCC or Clang on Linux
    ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
      CFLAGS += -fopenmp -DOMP
      LDLIBS += -lgomp
      $(info ✓ OpenMP found)
    else
      $(info ✗ OpenMP not found)
    endif
  endif
endif

# Check if OpenMPI and NCCL are available, include them if so, for multi-GPU training
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU (OpenMPI + NCCL) is manually disabled)
else
  ifeq ($(shell [ -d /usr/lib/x86_64-linux-gnu/openmpi/lib/ ] && [ -d /usr/lib/x86_64-linux-gnu/openmpi/include/ ] && echo "exists"), exists)
    $(info ✓ OpenMPI found, OK to train with multiple GPUs)
    NVCC_INCLUDES += -I/usr/lib/x86_64-linux-gnu/openmpi/include
    NVCC_LDFLAGS += -L/usr/lib/x86_64-linux-gnu/openmpi/lib/
    NVCC_LDLIBS += -lmpi -lnccl
    NVCC_FLAGS += -DMULTI_GPU
  else
    $(info ✗ OpenMPI is not found, disabling multi-GPU support)
    $(info ---> On Linux you can try install OpenMPI with `sudo apt install openmpi-bin openmpi-doc libopenmpi-dev`)
  endif
endif

# Precision settings, default to bf16 but ability to override
ifeq ($(MAKECMDGOALS), clean)
  PRECISION=BF16 
endif

VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# PHONY means these targets will always be executed
.PHONY: all clean

# Add targets
TARGETS = test_dataloader

# Dependency files
test_dataloader_dependencies = test_dataloader.d
HEADER_DEPENDENCIES = $(test_dataloader_dependencies)

# Conditional inclusion of CUDA targets
ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU/CUDA builds)
else
    $(info ✓ nvcc found, including GPU/CUDA support)
    TARGETS += 
endif

$(info ---------Build Configuration Complete - Build Targets -------------------------)

all: $(TARGETS)

# Generate dependency files
%.d: %.c
	$(CC) $(CFLAGS) -MMD -MP -MF $@ -c $<

# Include the dependency files
-include test_dataloader.d

test_dataloader: test_dataloader.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -MMD -MP $^ $(LDLIBS) $(OUTPUT_FILE)

clean:
	$(REMOVE_FILES) $(TARGETS) *.d *.o
	$(REMOVE_BUILD_OBJECT_FILES)
