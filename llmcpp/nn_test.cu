#include "gtest/gtest.h"
#include "nn.hpp"

TEST(Random, UniformFill) {
  nn::ManualSeed(42);
  std::vector<float> expected_num = {0.882269, 0.915004, 0.382864, 0.959306,
                                     0.390448, 0.600895, 0.256572, 0.793641,
                                     0.940771, 0.133186};
  size_t length = expected_num.size();
  float* d_num = (float*)nn::g_device.allocate(sizeof(float) * length);
  std::vector<float> num(10);
  nn::UniformFill(absl::MakeSpan(d_num, length));
  nn::g_device.memcpyDeviceToHost(num.data(), d_num, sizeof(float) * length);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }
}
