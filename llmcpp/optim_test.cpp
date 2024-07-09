#include "optim.hpp"
#include "gtest/gtest.h"

TEST(Optimizer, SGD) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Linear(3, 2)
optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
x = torch.randn(4, 3)
for _ in range(10):
  y = m(x)
  loss = torch.sum(y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  */

  nn::ManualSeed(42);
  int B = 4, in_features = 3, out_features = 2;
  nn::Linear m(in_features, out_features, true);

  // forward
  std::vector<float> x(B * in_features), y(B * out_features);
  nn::NormalFill(absl::MakeSpan(x));
  auto xm = Eigen::TensorMap<nn::Tensor2D>(x.data(), B, in_features);
  auto ym = Eigen::TensorMap<nn::Tensor2D>(y.data(), B, out_features);

  // optimizer
  std::vector<nn::Parameter*> parameters;
  m.Parameters(&parameters);
  optim::SGD sgd(parameters, 0.01);

  // backward
  std::vector<float> y_grad(y.size(), 1.0f);
  std::vector<float> x_grad(x.size(), 0.f);
  auto y_gradm = Eigen::TensorMap<nn::Tensor2D>(y_grad.data(), B, out_features);
  auto x_gradm = Eigen::TensorMap<nn::Tensor2D>(x_grad.data(), B, in_features);

  int step = 10;
  for (int i = 0; i < step; ++i) {
    m.Forward(xm, ym);
    sgd.ZeroGrad();
    m.Backward(xm, y_gradm, x_gradm);
    sgd.Step();
  }

  auto weight = m.weight_->View();
  auto bias = m.bias_->View();
  std::vector<float> expected_weight = {0.732981, 0.469633,  -0.589639,
                                        0.821935, -0.136072, -0.337878};
  std::vector<float> expected_bias = {-0.681086, -0.060932};
  for (size_t i = 0; i < expected_weight.size(); ++i) {
    EXPECT_NEAR(expected_weight[i], weight[i], 1e-5);
  }
  for (size_t i = 0; i < expected_bias.size(); ++i) {
    EXPECT_NEAR(expected_bias[i], bias[i], 1e-5);
  }
}

TEST(Optimizer, AdamW) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Linear(3, 2)
optimizer = torch.optim.AdamW(m.parameters(), lr=0.01, betas=(0.9, 0.999),
eps=1e-8, weight_decay=0.001)
x = torch.randn(4, 3)
for _ in range(10):
  y = m(x)
  loss = torch.sum(y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  */

  nn::ManualSeed(42);
  int B = 4, in_features = 3, out_features = 2;
  nn::Linear m(in_features, out_features, true);

  // forward
  std::vector<float> x(B * in_features), y(B * out_features);
  nn::NormalFill(absl::MakeSpan(x));
  auto xm = Eigen::TensorMap<nn::Tensor2D>(x.data(), B, in_features);
  auto ym = Eigen::TensorMap<nn::Tensor2D>(y.data(), B, out_features);

  // optimizer
  std::vector<nn::Parameter*> parameters;
  m.Parameters(&parameters);
  optim::AdamW adam_w(parameters, 0.01f, 0.9f, 0.999f, 1e-8f, 0.001f);

  // backward
  std::vector<float> y_grad(y.size(), 1.0f);
  std::vector<float> x_grad(x.size(), 0.f);
  auto y_gradm = Eigen::TensorMap<nn::Tensor2D>(y_grad.data(), B, out_features);
  auto x_gradm = Eigen::TensorMap<nn::Tensor2D>(x_grad.data(), B, in_features);

  int step = 10;
  for (int i = 0; i < step; ++i) {
    m.Forward(xm, ym);
    adam_w.ZeroGrad();
    m.Backward(xm, y_gradm, x_gradm);
    adam_w.Step(i + 1);
  }

  auto weight = m.weight_->View();
  auto bias = m.bias_->View();
  std::vector<float> expected_weight = {0.541358, 0.379162,  -0.235239,
                                        0.630303, -0.226482, 0.016497};
  std::vector<float> expected_bias = {-0.381053, 0.239038};
  for (size_t i = 0; i < expected_weight.size(); ++i) {
    EXPECT_NEAR(expected_weight[i], weight[i], 1e-5);
  }
  for (size_t i = 0; i < expected_bias.size(); ++i) {
    EXPECT_NEAR(expected_bias[i], bias[i], 1e-5);
  }
}
