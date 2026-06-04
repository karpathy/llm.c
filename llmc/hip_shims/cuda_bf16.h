// HIP-only forwarding shim: <cuda_bf16.h> -> the llm.c compat header
// (which includes <hip/hip_bf16.h> and aliases __nv_bfloat16 -> __hip_bfloat16).
#include "../cuda_to_hip.h"
