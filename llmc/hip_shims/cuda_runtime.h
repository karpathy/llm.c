// HIP-only forwarding shim: <cuda_runtime.h> -> the llm.c compat header.
// On the HIP build this dir is on the include path; on NVIDIA it is absent so
// the real CUDA toolkit header wins. See llmc/cuda_to_hip.h.
#include "../cuda_to_hip.h"
