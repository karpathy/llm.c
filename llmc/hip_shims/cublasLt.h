// HIP-only forwarding shim: <cublasLt.h> -> the llm.c compat header
// (which includes <hipblaslt/hipblaslt.h> and aliases the cublasLt* symbols).
#include "../cuda_to_hip.h"
