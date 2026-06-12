// HIP-only forwarding shim: <cooperative_groups.h> -> HIP's CG header plus the
// llm.c compat header (which adds the cg::reduce / cg::plus / cg::greater that
// HIP's cooperative groups lacks).
#include <hip/hip_cooperative_groups.h>
#include "../cuda_to_hip.h"
