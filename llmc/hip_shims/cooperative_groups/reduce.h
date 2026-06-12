// HIP-only forwarding shim: <cooperative_groups/reduce.h> -> the llm.c compat
// header, which provides the cg::reduce shim (HIP's CG has no reduce.h).
#include "../../cuda_to_hip.h"
