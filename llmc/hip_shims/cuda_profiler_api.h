// HIP-only forwarding shim: <cuda_profiler_api.h> -> the llm.c compat header
// (which maps cudaProfilerStart/Stop to hipSuccess no-ops -- hipProfilerStart
// returns hipErrorNotSupported and would trip cudaCheck).
#include "../cuda_to_hip.h"
