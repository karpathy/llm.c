// HIP-only forwarding shim: <nccl.h> -> RCCL (the ROCm NCCL drop-in). Only used
// by the optional MULTI_GPU build path; the RCCL API mirrors NCCL 1:1.
#include <rccl/rccl.h>
