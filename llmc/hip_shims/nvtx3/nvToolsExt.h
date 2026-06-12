// HIP-only forwarding shim: <nvtx3/nvToolsExt.h> -> the llm.c compat header
// (which stubs nvtxRangePush/Pop to no-ops).
#include "../../cuda_to_hip.h"
