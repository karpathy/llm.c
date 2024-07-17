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
  gpt::MLP<float> mlp(n_embed);

  // initialize
  std::vector<float> x = {0.548416, -0.441472, 1.581529, -0.198127,
                          0.955371, -1.090163, 2.595151, 2.750435,
                          0.648762, 0.449632,  0.322023, -1.050281};
  // forward
  std::vector<float> y(x.size(), 0);
  auto xm = MakeConstMatrix(x.data(), B, n_embed);
  auto ym = MakeMatrix(y.data(), B, n_embed);
  mlp.Forward(xm, ym);

  std::vector<float> expected_y = {0.068570,  0.127020, 0.150101,  0.318999,
                                   -0.118447, 0.216090, 0.046674,  -0.052894,
                                   0.213587,  0.255707, -0.119030, 0.244947};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> x_grad(x.size(), 0), y_grad(y.size(), 1);
  auto x_grad_m = MakeMatrix(x_grad.data(), B, n_embed);
  auto y_grad_m = MakeConstMatrix(y_grad.data(), B, n_embed);
  mlp.Backward(xm, y_grad_m, x_grad_m);

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

  std::vector<float> x(B * T * C), y(B * T * C), x_grad(B * T * C),
      y_grad(B * T * C, 1.0f);
  nn::NormalFill(absl::MakeSpan(x));
  auto xt = MakeConst3DTensor(x.data(), B, T, C);
  auto yt = Make3DTensor(y.data(), B, T, C);
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

  // backward
  auto x_grad_t = Make3DTensor(x_grad.data(), B, T, C);
  auto y_grad_t = MakeConst3DTensor(y_grad.data(), B, T, C);
  attn.Backward(xt, y_grad_t, x_grad_t);
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

  std::vector<float> x(B * T * C), y(B * T * C);
  nn::NormalFill(absl::MakeSpan(x));
  auto xt = MakeConst3DTensor(x.data(), B, T, C);
  auto yt = Make3DTensor(y.data(), B, T, C);
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

  // backward
  std::vector<float> x_grad(B * T * C, 0.0), y_grad(B * T * C, 1.0f);
  auto x_grad_3d = Make3DTensor(x_grad.data(), B, T, C);
  auto y_grad_3d = MakeConst3DTensor(y_grad.data(), B, T, C);
  block.Backward(xt, y_grad_3d, x_grad_3d);

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

TEST(GPT, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(block_size=4, n_embd=6, n_head=2, n_layer=12, vocab_size=10)
gpt2 = GPT(config=config)
B, T, C = 2, 4, 6
idx = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
targets = torch.LongTensor([[2, 4, 5, 6], [3, 2, 9, 0]])
logits, loss = gpt2(idx)
  */

  nn::ManualSeed(42);
  int block_size = 4, n_embd = 6, n_head = 2, n_layer = 12, vocab_size = 10;
  int B = 2, T = block_size, C = n_embd, nh = n_head, hs = n_embd / nh;
  gpt::GPT<float> gpt(block_size, vocab_size, vocab_size, n_layer, n_head, n_embd);

  std::vector<int> idx = {1, 2, 4, 5, 4, 3, 2, 9};
  auto idx_m = TTypes<int>::ConstMatrix(idx.data(), B, T);
  std::vector<float> logits(B * T * vocab_size);
  //  auto logits_2d = Eigen::Map<nn::Matrix>(logits.data(), B * T, vocab_size);
  auto logits_3d =
      Make3DTensor(logits.data(), B, T, vocab_size);

  // Without targets
  gpt.Forward(idx_m, logits_3d);
  nn::Tensor2D logits_last_t = logits_3d.chip(T - 1, 1);
  auto logits_last_t_span =
      absl::MakeSpan(logits_last_t.data(), B * vocab_size);
  std::vector<float> expected_logits_last_t = {
      0.678302,  0.421532, 0.378493, -0.835950, 0.857062, 0.964537, 0.516393,
      -0.197802, 0.534203, 1.200121, 0.472574,  0.484028, 0.511667, -0.864123,
      0.911163,  0.713747, 0.539312, -0.185592, 0.479005, 1.159166};
  for (size_t i = 0; i < expected_logits_last_t.size(); ++i) {
    EXPECT_NEAR(expected_logits_last_t[i], logits_last_t_span[i], 1e-5);
  }

  // With targets
  std::vector<int> target = {2, 4, 5, 6, 3, 2, 9, 0};
  auto target_m = TTypes<int>::ConstMatrix (target.data(), B, T);
  float loss = 0.0;
  gpt.Forward(idx_m, target_m, logits_3d, &loss);

  std::vector<float> expected_logits = {
      0.412538,  1.435400,  -0.845702, -0.836272, -0.912762, 0.366170,
      -1.017333, 0.049309,  0.443828,  0.382751,  0.405025,  -0.624728,
      -0.233263, 0.605841,  -0.331391, 0.600770,  -0.219692, -0.349443,
      0.020259,  -0.885378, 0.655647,  -0.468965, 0.474971,  -0.146528,
      0.848733,  1.132179,  0.723393,  -0.373132, 0.341525,  0.446182,
      0.678302,  0.421532,  0.378493,  -0.835950, 0.857062,  0.964537,
      0.516393,  -0.197802, 0.534203,  1.200121,  -0.333841, 0.735604,
      -0.513971, -0.085284, -1.131126, -0.249146, -0.997829, -0.158174,
      0.046483,  -0.956481, 0.138260,  -0.737712, -0.238145, 0.771908,
      -0.482225, 0.335722,  -0.248654, -0.296130, -0.121716, -1.160217,
      0.330651,  -0.616385, 0.539336,  0.109500,  0.612342,  0.884791,
      0.590758,  -0.489924, 0.210640,  -0.198728, 0.472574,  0.484028,
      0.511667,  -0.864123, 0.911163,  0.713747,  0.539312,  -0.185592,
      0.479005,  1.159166};
  for (size_t i = 0; i < expected_logits.size(); ++i) {
    EXPECT_NEAR(expected_logits[i], logits[i], 1e-5);
  }
  EXPECT_NEAR(loss, 2.485592, 1e-5);

  // Backward
  gpt.Backward(idx_m, target_m);
  auto wte_grad = gpt.wte_->weight_->View(nn::Parameter::kGrad);
  auto wpe_grad = gpt.wpe_->weight_->View(nn::Parameter::kGrad);

  std::vector<float> expected_wte_grad = {
      -1.274265e-01, -7.693546e-02, 1.870265e-01,  1.102872e-01,
      -8.144964e-02, -1.150192e-02, 7.081421e-02,  -5.410360e-02,
      -2.618172e-02, -3.396006e-02, -8.665409e-02, 1.300852e-01,
      -3.946760e-02, 2.253419e-01,  -6.542038e-02, 2.730466e-03,
      9.105621e-02,  -2.142406e-01, -8.473338e-02, -2.970357e-02,
      1.036018e-01,  -1.241457e-01, 1.957041e-01,  -6.072314e-02,
      5.077662e-01,  -9.879547e-02, -4.231800e-01, -1.456355e-01,
      9.783387e-02,  6.201106e-02,  1.025773e-02,  -5.858323e-02,
      -4.027011e-02, 1.477207e-01,  -1.223148e-01, 6.318975e-02,
      -1.104821e-01, -1.816282e-02, 1.313495e-01,  1.263026e-01,
      -7.210705e-02, -5.690017e-02, -9.386335e-05, -1.019029e-02,
      2.964642e-02,  -3.682173e-02, -4.120179e-02, 5.866125e-02,
      6.840312e-03,  -8.432365e-03, 3.812137e-02,  -7.966967e-02,
      -4.943971e-02, 9.258006e-02,  8.474199e-02,  -5.575795e-02,
      -1.460089e-01, 1.147824e-01,  3.205336e-02,  -2.981084e-02};
  for (size_t i = 0; i < expected_wte_grad.size(); ++i) {
    EXPECT_NEAR(expected_wte_grad[i], wte_grad[i], 1e-5);
  }

  std::vector<float> expected_wpe_grad = {
      0.338320,  -0.198764, -0.204031, -0.057509, 0.055975,  0.066009,
      0.028694,  -0.040975, 0.019426,  0.036110,  -0.040719, -0.002536,
      -0.051870, 0.060403,  -0.010102, 0.067178,  -0.027745, -0.037864,
      0.003073,  -0.005986, -0.016609, 0.035812,  -0.024031, 0.007741};
  for (size_t i = 0; i < expected_wpe_grad.size(); ++i) {
    EXPECT_NEAR(expected_wpe_grad[i], wpe_grad[i], 1e-5);
  }
}
