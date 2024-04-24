CC ?= clang
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes
LDFLAGS =
LDLIBS = -lm
INCLUDES =
CFLAGS_COND = -march=native
SHELL_UNAME = $(shell uname)
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@

# NVCC flags
NVCC_FLAGS = -O3 --use_fast_math
NVCC_LDFLAGS = -lcublas -lcublasLt

ifneq ($(OS), Windows_NT)
  NVCC := $(shell which nvcc 2>/dev/null)

  # Function to test if the compiler accepts a given flag.
  define check_and_add_flag
    $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
    ifeq ($(FLAG_SUPPORTED),yes)
        CFLAGS += $(1)
    endif
  endef

  # Check each flag and add it if supported
  $(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))
else
  CFLAGS :=
  REMOVE_FILES = del
  SHELL_UNAME := Windows
  NVCC := $(shell where nvcc 2> nul)
  CC := cl
  CFLAGS = /Idev /Zi /nologo /Wall /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
   /external:W3 /Gd /TP /wd4996 /FC /openmp:llvm
  LDFLAGS :=
  LDLIBS :=
  INCLUDES :=
  NVCC_FLAGS += -I"dev"
  WIN_CUDA_RENAME = rename $@.exe $@
	OUTPUT_FILE = /link /OUT:$@
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
  # Detect if running on macOS or Linux
  ifeq ($(SHELL_UNAME), Darwin)
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
    ifneq ($(OS), Windows_NT)
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
endif

# DEFAULT means these targets will always be executed
.DEFAULT: all 

# Add targets
TARGETS = train_gpt2 test_gpt2

# Conditional inclusion of CUDA targets
ifeq ($(NVCC),)
    $(info nvcc not found, skipping CUDA builds)
else
    $(info nvcc found, including CUDA builds)
    TARGETS += train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu profile_gpt2cu
endif

all: $(TARGETS)

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) $(OUTPUT_FILE)

train_gpt2cu: train_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@
	$(WIN_CUDA_RENAME)

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@
	$(WIN_CUDA_RENAME)

test_gpt2cu: test_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@
	$(WIN_CUDA_RENAME)

test_gpt2fp32cu: test_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $< $(NVCC_LDFLAGS) -o $@
	$(WIN_CUDA_RENAME)

profile_gpt2cu: profile_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) -lineinfo $< $(NVCC_LDFLAGS) -o $@
	$(WIN_CUDA_RENAME)

clean:
	$(REMOVE_FILES) train_gpt2 test_gpt2 train_gpt2cu train_gpt2fp32cu test_gpt2cu test_gpt2fp32cu profile_gpt2cu
