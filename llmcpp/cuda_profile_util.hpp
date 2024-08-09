#ifndef LLM_CPP_LLMCPP_CUDA_PROFILE_UTIL_HPP_
#define LLM_CPP_LLMCPP_CUDA_PROFILE_UTIL_HPP_

#include <nvtx3/nvToolsExt.h>
#include <string>

// Profiler utils
class NvtxRange {
 public:
  NvtxRange(const char* s) { nvtxRangePush(s); }
  NvtxRange(const char* prefix, const char* s) {
    std::string message = std::string(prefix) + "::" + std::string(s);
    nvtxRangePush(message.c_str());
  }
  NvtxRange(const std::string& base_str, int number) {
    std::string range_string = base_str + " " + std::to_string(number);
    nvtxRangePush(range_string.c_str());
  }
  ~NvtxRange() { nvtxRangePop(); }
};
#define NVTX_RANGE_FN(prefix) NvtxRange nvtx_range(prefix, __FUNCTION__)

#endif  // LLM_CPP_LLMCPP_CUDA_PROFILE_UTIL_HPP_
