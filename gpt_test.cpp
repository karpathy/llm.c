#include "gpt.hpp"
#include "gtest/gtest.h"

TEST(MLP, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
class MLP(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
    self.gelu    = NewGELU()
    self.c_proj  = nn.Linear(4 * n_embd, n_embd)
    self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

B, n_embed = 4, 3
m = MLP(n_embed)
x = torch.randn(4, 3)
x = nn.Parameter(x)
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  int B = 4, n_embed = 3;
  gpt::MLP mlp(n_embed);

  // initialize
  std::vector<float> x = {0.548416, -0.441472, 1.581529, -0.198127,
                          0.955371, -1.090163, 2.595151, 2.750435,
                          0.648762, 0.449632,  0.322023, -1.050281};
  // forward
  std::vector<float> y(x.size(), 0);
  auto xm = Eigen::Map<nn::Matrix>(x.data(), B, n_embed);
  auto ym = Eigen::Map<nn::Matrix>(y.data(), B, n_embed);
  mlp.Forward(xm, ym);

  std::vector<float> expected_y = {0.068570,  0.127020, 0.150101,  0.318999,
                                   -0.118447, 0.216090, 0.046674,  -0.052894,
                                   0.213587,  0.255707, -0.119030, 0.244947};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> x_grad(x.size(), 0), y_grad(y.size(), 1);
  auto x_grad_m = Eigen::Map<nn::Matrix>(x_grad.data(), B, n_embed);
  auto y_grad_m = Eigen::Map<nn::Matrix>(y_grad.data(), B, n_embed);
  mlp.Backward(xm, y_grad_m, x_grad_m);

  std::vector<float> expected_x_grad = {
      0.081099, -0.311384, -0.132192, 0.183097, 0.171755, -0.127825,
      0.128794, -0.148038, -0.287548, 0.066056, 0.149289, 0.009199};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(CausalSelfAttention, Forward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(n_embd=6, n_head=2)x
attn = CausalSelfAttention(config=config)
B, T, C = 2, 4, 6
x = torch.randn((B, T, C))
y = attn(x)
  */

  nn::ManualSeed(42);
  int B = 2, T = 4, C = 6, nh = 2, hs = C / nh;
  gpt::CausalSelfAttention attn(T, nh, C);

  std::vector<float> x(B * T * C), y(B * T * C);
  nn::NormalFill(absl::MakeSpan(x));
  Eigen::TensorMap<nn::Tensor3D> xt(x.data(), B, T, C);
  Eigen::TensorMap<nn::Tensor3D> yt(y.data(), B, T, C);
  //  std::cout << "input\n" << xt << std::endl;
  attn.Forward(xt, yt);

  std::vector<float> expected_y = {
      2.908887e-01,  -7.630450e-01, 9.319627e-02,  1.153549e-01,  -2.430970e-01,
      1.394629e-01,  3.247743e-01,  -6.480877e-01, 1.452082e-01,  -4.993150e-02,
      -2.286499e-01, 2.117769e-01,  1.664312e-01,  -3.892012e-01, 1.849052e-01,
      -2.083585e-01, -1.347157e-01, 9.595731e-02,  1.765069e-01,  -5.302818e-01,
      1.738015e-01,  -2.988278e-01, -1.474487e-01, 2.397015e-01,  -6.180903e-02,
      9.246814e-02,  -1.355038e-01, -5.402240e-01, -3.939950e-01, -3.815557e-01,
      -9.731026e-02, 5.925810e-02,  2.927732e-02,  -5.060250e-01, -2.718721e-01,
      -1.118071e-01, -1.467107e-01, 3.228784e-04,  1.027589e-01,  -3.759150e-01,
      -2.224361e-01, 4.695907e-02,  -8.113065e-02, 1.003927e-02,  2.985954e-03,
      -4.915146e-01, -2.742706e-01, -1.377932e-01};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }
}

TEST(Block, Forward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(n_embd=6, n_head=2)x
b = Block(config=config)
B, T, C = 2, 4, 6
x = torch.randn((B, T, C))
y = b(x)
  */

  nn::ManualSeed(42);
  int B = 2, T = 4, C = 6, nh = 2, hs = C / nh;
  gpt::Block block(T, nh, C);

  std::vector<float> x(B * T * C), y(B * T * C);
  nn::NormalFill(absl::MakeSpan(x));
  Eigen::TensorMap<nn::Tensor3D> xt(x.data(), B, T, C);
  Eigen::TensorMap<nn::Tensor3D> yt(y.data(), B, T, C);
  block.Forward(xt, yt);

  std::vector<float> expected_y = {
      -1.582970, -0.700668, -1.026945, 2.316710,  -1.464859, -1.201760,
      -1.180800, -0.050612, -0.057298, -0.403900, -2.935433, -1.328336,
      0.837885,  0.920688,  0.547759,  0.835524,  -1.913520, 1.494968,
      -1.152616, 1.295917,  0.097820,  0.347780,  0.411883,  0.530892,
      -0.818162, 1.144261,  -1.390720, 0.676694,  0.590124,  0.132416,
      -0.988301, 0.740322,  -0.836062, -1.058745, -0.941910, -1.701103,
      -1.085605, -0.035773, -1.859936, 0.101301,  -2.202660, -0.783755,
      -0.486846, -0.363502, -0.317611, 1.436486,  0.691510,  -0.423726};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }
}

TEST(GPT, Forward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(n_embd=6, n_head=2)x
b = Block(config=config)
B, T, C = 2, 4, 6
x = torch.randn((B, T, C))
y = b(x)
  */

  nn::ManualSeed(42);
  int block_size = 4, n_embd = 6, n_head = 2, n_layer = 3, vocab_size = 10;
  int B = 2, T = block_size, C = n_embd, nh = n_head, hs = n_embd / nh;
  gpt::GPT gpt(block_size, vocab_size, n_layer, n_head, n_embd);

  std::vector<int> idx = {1, 2, 4, 5, 4, 3, 2, 9};
  auto idx_m = Eigen::Map<nn::MatrixInt>(idx.data(), B, T);
  auto targets = Eigen::Map<nn::MatrixInt>(nullptr, 0, 0);
  std::vector<float> logits(B*T*vocab_size);
  auto logits_2d = Eigen::Map<nn::Matrix>(logits.data(), B * T, vocab_size);
  float loss = 0;
  gpt.Forward(idx_m, logits_2d);

  //  std::vector<float> expected_y = {
  //      -1.582970, -0.700668, -1.026945, 2.316710,  -1.464859, -1.201760,
  //      -1.180800, -0.050612, -0.057298, -0.403900, -2.935433, -1.328336,
  //      0.837885,  0.920688,  0.547759,  0.835524,  -1.913520, 1.494968,
  //      -1.152616, 1.295917,  0.097820,  0.347780,  0.411883,  0.530892,
  //      -0.818162, 1.144261,  -1.390720, 0.676694,  0.590124,  0.132416,
  //      -0.988301, 0.740322,  -0.836062, -1.058745, -0.941910, -1.701103,
  //      -1.085605, -0.035773, -1.859936, 0.101301,  -2.202660, -0.783755,
  //      -0.486846, -0.363502, -0.317611, 1.436486,  0.691510,  -0.423726};
  //  for (size_t i = 0; i < expected_y.size(); ++i) {
  //    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  //  }
}
