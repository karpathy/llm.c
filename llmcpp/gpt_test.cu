#include "gpt.hpp"
#include "gtest/gtest.h"

using nn::DT_FLOAT;
using nn::Parameter;

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
  gpt::MLP<float> mlp(n_embed);

  // initialize
  std::vector<float> x = {0.548416, -0.441472, 1.581529, -0.198127,
                          0.955371, -1.090163, 2.595151, 2.750435,
                          0.648762, 0.449632,  0.322023, -1.050281};
  nn::Parameter d_x(DT_FLOAT, x.size()), d_y(DT_FLOAT, x.size());
  nn::g_device.memcpyHostToDevice(d_x.data<float>(), x.data(),
                                  sizeof(float) * x.size());

  // forward
  std::vector<float> y(x.size(), 0);
  auto xm = MakeConstMatrix(d_x.data<float>(), B, n_embed);
  auto ym = MakeMatrix(d_y.data<float>(), B, n_embed);
  mlp.Forward(xm, ym);
  nn::g_device.memcpyDeviceToHost(y.data(), d_y.data<float>(),
                                  sizeof(float) * x.size());
  nn::g_device.synchronize();

  std::vector<float> expected_y = {0.068570,  0.127020, 0.150101,  0.318999,
                                   -0.118447, 0.216090, 0.046674,  -0.052894,
                                   0.213587,  0.255707, -0.119030, 0.244947};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> x_grad(x.size(), 0), y_grad(y.size(), 1);
  Parameter d_x_grad(DT_FLOAT, x.size()), d_y_grad(DT_FLOAT, x.size());
  nn::g_device.memcpyHostToDevice(d_y_grad.data<float>(), y_grad.data(),
                                  sizeof(float) * y_grad.size());
  auto x_grad_m = MakeMatrix(d_x_grad.data<float>(), B, n_embed);
  auto y_grad_m = MakeConstMatrix(d_y_grad.data<float>(), B, n_embed);
  mlp.Backward(xm, y_grad_m, x_grad_m);
  nn::g_device.memcpyDeviceToHost(x_grad.data(), d_x_grad.data<float>(),
                                  sizeof(float) * x_grad.size());
  nn::g_device.synchronize();
  std::vector<float> expected_x_grad = {
      0.081099, -0.311384, -0.132192, 0.183097, 0.171755, -0.127825,
      0.128794, -0.148038, -0.287548, 0.066056, 0.149289, 0.009199};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(CausalSelfAttention, ForwardAndBackward) {
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
  gpt::CausalSelfAttention<float> attn(T, nh, C);

  Parameter d_x(DT_FLOAT, B * T * C), d_y(DT_FLOAT, B * T * C),
      d_x_grad(DT_FLOAT, B * T * C), d_y_grad(DT_FLOAT, B * T * C);
  std::vector<float> y(B * T * C), x_grad(B * T * C), y_grad(B * T * C, 1.0f);
  nn::g_device.memcpyHostToDevice(d_y_grad.data<float>(), y_grad.data(),
                                  sizeof(float) * y_grad.size());
  nn::NormalFill(d_x.span<float>());
  auto xt = MakeConst3DTensor(d_x.data<float>(), B, T, C);
  auto yt = Make3DTensor(d_y.data<float>(), B, T, C);
  attn.Forward(xt, yt);
  nn::g_device.memcpyDeviceToHost(y.data(), d_y.data<float>(),
                                  sizeof(float) * y.size());
  nn::g_device.synchronize();

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

  // backward
  auto x_grad_t = Make3DTensor(d_x_grad.data<float>(), B, T, C);
  auto y_grad_t = MakeConst3DTensor(d_y_grad.data<float>(), B, T, C);
  attn.Backward(xt, y_grad_t, x_grad_t);
  nn::g_device.memcpyDeviceToHost(x_grad.data(), d_x_grad.data<float>(),
                                  sizeof(float) * x_grad.size());
  nn::g_device.synchronize();
  std::vector<float> expected_x_grad = {
      1.341534, 0.306510, -0.398248, -1.307186, 0.308231,  1.364989,
      0.722659, 0.155880, -0.282098, -0.693888, 0.194182,  0.756926,
      0.348222, 0.125258, 0.032523,  -0.342366, 0.064738,  0.288353,
      0.173461, 0.051528, -0.036712, -0.158987, 0.042912,  0.170093,
      1.512321, 0.363056, -0.505860, -1.425764, 0.427224,  1.476094,
      0.599044, 0.077030, -0.059634, -0.600186, 0.065210,  0.628942,
      0.319907, 0.015346, -0.002683, -0.284920, -0.009112, 0.374621,
      0.279890, 0.017215, -0.036771, -0.177604, -0.009638, 0.332324};

  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(Block, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(n_embd=6, n_head=2)
b = Block(config=config)
B, T, C = 2, 4, 6
x = torch.randn((B, T, C))
x = nn.Parameter(x)
y = b(x)
loss = torch.sum(y)
loss.backward()
*/

  nn::ManualSeed(42);
  int B = 2, T = 4, C = 6, nh = 2, hs = C / nh;
  gpt::Block<float> block(T, nh, C);

  Parameter d_x(DT_FLOAT, B * T * C), d_y(DT_FLOAT, B * T * C);
  std::vector<float> y(B * T * C);
  nn::NormalFill(d_x.span<float>());
  auto xt = MakeConst3DTensor(d_x.data<float>(), B, T, C);
  auto yt = Make3DTensor(d_y.data<float>(), B, T, C);
  block.Forward(xt, yt);
  nn::g_device.memcpyDeviceToHost(y.data(), d_y.data<float>(),
                                  sizeof(float) * y.size());
  nn::g_device.synchronize();

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

  // backward
  Parameter d_x_grad(DT_FLOAT, B * T * C), d_y_grad(DT_FLOAT, B * T * C);
  std::vector<float> x_grad(B * T * C, 0.0), y_grad(B * T * C, 1.0f);
  nn::g_device.memcpyHostToDevice(d_y_grad.data<float>(), y_grad.data(),
                                  sizeof(float) * y_grad.size());
  auto x_grad_3d = Make3DTensor(d_x_grad.data<float>(), B, T, C);
  auto y_grad_3d = MakeConst3DTensor(d_y_grad.data<float>(), B, T, C);
  block.Backward(xt, y_grad_3d, x_grad_3d);
  nn::g_device.memcpyDeviceToHost(x_grad.data(), d_x_grad.data<float>(),
                                  sizeof(float) * x_grad.size());
  nn::g_device.synchronize();

  std::vector<float> expected_x_grad = {
      1.226099, 0.810447, 0.295263, 0.956450, 1.194399,  1.517342, 1.408440,
      1.329727, 0.766482, 0.535780, 0.735261, 1.224310,  1.032075, 1.120100,
      1.104306, 1.070927, 0.915230, 0.757362, 0.889456,  0.844044, 0.743556,
      1.444354, 1.488363, 0.590226, 1.685244, 0.941806,  0.423899, 0.210837,
      1.328500, 1.409714, 2.011978, 1.126987, -0.052393, 0.161476, 1.464273,
      1.287678, 0.967238, 1.173121, 0.719205, 1.138348,  1.343792, 0.658296,
      1.138839, 0.948495, 0.854385, 0.838419, 1.305177,  0.914685};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}
