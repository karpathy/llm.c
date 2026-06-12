// HIP-only forwarding shim: <cublas_v2.h> -> the llm.c compat header
// (which includes <hipblas/hipblas.h> and aliases the cublas* symbols used).
#include "../cuda_to_hip.h"
