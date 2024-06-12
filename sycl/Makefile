# Compiler
CC          := icpx
# Flags
CXXFLAGS      := -std=c++17 -fsycl -O3 -fno-finite-math-only -qmkl=parallel 
OPTIMIZE    := yes
PVC         := yes
CUDA        := no
CUDA_ARCH   := sm_70
HIP         := no
HIP_ARCH    := gfx908
DNN         := yes
NOWARN      := yes

# Debug flag
ifeq ($(DEBUG),yes)
    CXXFLAGS += -g -DDEBUG
endif

#PVC flag
ifeq ($(SPIRV),yes)
    CXXFLAGS += -fsycl-targets=spir64_gen -device=pvc -fPIC -DMKL_ILP64 
endif


# CUDA flag
ifeq ($(CUDA),yes)
    CXXFLAGS += -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

# HIP flag
ifeq ($(HIP),yes)
    CXXFLAGS += -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

#DNN
ifeq ($(DNN),yes)
  CXXFLAGS += -ldnnl
endif

ifeq ($(NOWARN),yes)
  CXXFLAGS += -w
endif

# Build rule for individual files
%: %.cpp
	$(CC) $(CXXFLAGS) $< -o $@.o


