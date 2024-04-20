CC ?= clang
CFLAGS = -Ofast -fno-finite-math-only -Wno-unused-result -march=native
LDFLAGS =
LDLIBS = -lm
INCLUDES =

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
ifeq ($(shell uname), Darwin)
  # Check if the libomp directory exists
  ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
    # macOS with Homebrew and directory exists
    CFLAGS += -Xclang -fopenmp -DOMP
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib
    LDLIBS += -lomp
    INCLUDES += -I/opt/homebrew/opt/libomp/include
    $(info NICE Compiling with OpenMP support)
  else ifeq ($(shell [ -d /usr/local/opt/libomp/lib ] && echo "exists"), exists)
    CFLAGS += -Xclang -fopenmp -DOMP
    LDFLAGS += -L/usr/local/opt/libomp/lib
    LDLIBS += -lomp
    INCLUDES += -I/usr/local/opt/libomp/include
    $(info NICE Compiling with OpenMP support)
  else
    $(warning OOPS Compiling without OpenMP support)
  endif
else
  ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
    # Ubuntu or other Linux distributions
    CFLAGS += -fopenmp -DOMP
    LDLIBS += -lgomp
    $(info NICE Compiling with OpenMP support)
  else
    ifneq ($(OS), Windows_NT)
      $(warning OOPS Compiling without OpenMP support)
    endif
  endif
endif

ifeq ($(OS), Windows_NT)
  CC := cl 
  CFLAGS := /Iplatform\windows /Zi /nologo /W3 /WX- /diagnostics:column /sdl /Ox /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
   /external:W3 /Gd /TP /wd4996 /FC /openmp:llvm 
  LDFLAGS :=
  LDLIBS :=
  INCLUDES :=
  CUDA_INCLUDES := -I"platform\windows"
  EXEFILE := .exe
  WINDOWS_UNISTD_H := platform\windows\unistd.h

  # check if windows unistd.h does not exist, and create it if it's missing.
$(WINDOWS_UNISTD_H): 
  ifneq ($(wildcard $(WINDOWS_UNISTD_H))",")
	@echo $(WINDOWS_UNISTD_H) present.
  else
	@echo $(WINDOWS_UNISTD_H) not present. Creating it...
	@echo #ifndef UNISTD_H >> $(WINDOWS_UNISTD_H) 
	@echo #define UNISTD_H >> $(WINDOWS_UNISTD_H)
	@echo #define _CRT_SECURE_NO_WARNINGS >> $(WINDOWS_UNISTD_H)
	@echo #define _USE_MATH_DEFINES >> $(WINDOWS_UNISTD_H)
	@echo #include ^<math.h^> >>  $(WINDOWS_UNISTD_H)
	@echo ^/^/#define gen_max_length 64 ^/^/ compile as C++ to skip this VLA issue >>  $(WINDOWS_UNISTD_H)
	@echo #include ^<time.h^> >>  $(WINDOWS_UNISTD_H)
	@echo #define CLOCK_MONOTONIC 0 >>  $(WINDOWS_UNISTD_H)
	@echo int clock_gettime(int ignore_variable, struct timespec* tv) >>  $(WINDOWS_UNISTD_H)
	@echo { >>  $(WINDOWS_UNISTD_H)
	@echo     return timespec_get(tv, TIME_UTC); ^/^/ TODO: not sure this is the best solution. Need to review. >>  $(WINDOWS_UNISTD_H)
	@echo } >>  $(WINDOWS_UNISTD_H)
	@echo #define OMP ^/* turn it on *^/ >>  $(WINDOWS_UNISTD_H)
	@echo #include  ^<io.h^> ^/* needed for access below *^/ >>  $(WINDOWS_UNISTD_H)
	@echo #define F_OK 0 >>  $(WINDOWS_UNISTD_H)
	@echo #define access _access >>  $(WINDOWS_UNISTD_H)
	@echo #define TURN_OFF_FP_FAST __pragma(float_control( precise, on, push )) ^/^/ Save current setting and turn on ^/fp:precise  >>  $(WINDOWS_UNISTD_H)
	@echo #define TURN_ON_FP_FAST  __pragma(float_control(pop)) ^/^/ Restore file's default settings  >>  $(WINDOWS_UNISTD_H)
	@echo #endif >>  $(WINDOWS_UNISTD_H)
	@echo $(WINDOWS_UNISTD_H) created.
   endif
else
	MAC_LINUX_OUTPUT_FILE = -o $@
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu $(WINDOWS_UNISTD_H)

# default target is all
all: train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu $(WINDOWS_UNISTD_H)

train_gpt2: train_gpt2.c $(WINDOWS_UNISTD_H)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) $(MAC_LINUX_OUTPUT_FILE)

test_gpt2: test_gpt2.c $(WINDOWS_UNISTD_H)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) $(MAC_LINUX_OUTPUT_FILE)

# possibly may want to disable warnings? e.g. append -Xcompiler -Wno-unused-result
train_gpt2cu: train_gpt2.cu $(WINDOWS_UNISTD_H)
	nvcc $(CUDA_INCLUDES) -O3 --use_fast_math $< -lcublas -lcublasLt -o $@$(EXEFILE)

test_gpt2cu: test_gpt2.cu $(WINDOWS_UNISTD_H)
	nvcc  $(CUDA_INCLUDES) -O3 --use_fast_math $< -lcublas -lcublasLt -o $@$(EXEFILE)

profile_gpt2cu: profile_gpt2.cu
	nvcc -O3 --use_fast_math -lineinfo $< -lcublas -lcublasLt -o $@

clean:
	rm -f train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu $(WINDOWS_UNISTD_H)
