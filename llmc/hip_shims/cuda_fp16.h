// HIP-only forwarding shim: <cuda_fp16.h> -> the llm.c compat header
// (which includes <hip/hip_fp16.h>; HIP defines `half`/`__half`).
#include "../cuda_to_hip.h"
