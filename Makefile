CC = clang
CFLAGS = -O3 -Ofast
LDFLAGS =
LDLIBS = -lm
INCLUDES =

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
ifeq ($(shell echo | $(CC) -Xpreprocessor -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
  ifeq ($(shell uname), Darwin)
    # macOS with Homebrew
    CFLAGS += -Xclang -fopenmp
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib
    LDLIBS += -lomp
    INCLUDES += -I/opt/homebrew/opt/libomp/include
  else
    # Ubuntu or other Linux distributions
    CFLAGS += -fopenmp
    LDLIBS += -lgomp
  endif
  $(info NICE Compiling with OpenMP support)
else
  $(warning OOPS Compiling without OpenMP support)
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2

# default target is all
all: train_gpt2 test_gpt2

train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f train_gpt2 test_gpt2
