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
  gpt::CausalSelfAttention attn(T, nh, C);

  std::vector<float> x(B * T * C), y(B * T * C), x_grad(B * T * C),
      y_grad(B * T * C, 1.0f);
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

  // backward
  Eigen::TensorMap<nn::Tensor3D> x_grad_t(x_grad.data(), B, T, C);
  Eigen::TensorMap<nn::Tensor3D> y_grad_t(y_grad.data(), B, T, C);
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

  // backward
  std::vector<float> x_grad(B * T * C, 0.0), y_grad(B * T * C, 1.0f);
  auto x_grad_3d = Eigen::TensorMap<nn::Tensor3D>(x_grad.data(), B, T, C);
  auto y_grad_3d = Eigen::TensorMap<nn::Tensor3D>(y_grad.data(), B, T, C);
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

TEST(GPT, Forward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(block_size=4, n_embd=6, n_head=2, n_layer=3, vocab_size=10)
gpt2 = GPT(config=config)
B, T, C = 2, 4, 6
idx = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
targets = torch.LongTensor([[2, 4, 5, 6], [3, 2, 9, 0]])
logits, loss = gpt2(idx)
  */

  nn::ManualSeed(42);
  int block_size = 4, n_embd = 6, n_head = 2, n_layer = 3, vocab_size = 10;
  int B = 2, T = block_size, C = n_embd, nh = n_head, hs = n_embd / nh;
  gpt::GPT gpt(block_size, vocab_size, vocab_size, n_layer, n_head, n_embd);

  std::vector<int> idx = {1, 2, 4, 5, 4, 3, 2, 9};
  auto idx_m = Eigen::Map<nn::MatrixInt>(idx.data(), B, T);
  std::vector<float> logits(B * T * vocab_size);
  //  auto logits_2d = Eigen::Map<nn::Matrix>(logits.data(), B * T, vocab_size);
  auto logits_3d =
      Eigen::TensorMap<nn::Tensor3D>(logits.data(), B, T, vocab_size);

  // Without targets
  gpt.Forward(idx_m, logits_3d);
  nn::Tensor2D logits_last_t = logits_3d.chip(T - 1, 1);
  auto logits_last_t_span =
      absl::MakeSpan(logits_last_t.data(), B * vocab_size);
  std::vector<float> expected_logits_last_t = {
      0.090488,  0.989613,  0.485091,  -0.415317, 0.200022,
      0.379014,  -0.672617, -0.472669, -1.369164, -0.364832,
      -0.003667, 0.883650,  0.330532,  -0.294204, 0.039559,
      0.409321,  -0.487800, -0.464272, -1.270504, -0.087506};
  for (size_t i = 0; i < expected_logits_last_t.size(); ++i) {
    EXPECT_NEAR(expected_logits_last_t[i], logits_last_t_span[i], 1e-5);
  }

  // With targets
  std::vector<int> target = {2, 4, 5, 6, 3, 2, 9, 0};
  auto target_m = Eigen::Map<nn::MatrixInt>(target.data(), B, T);
  float loss = 0.0;
  gpt.Forward(idx_m, target_m, logits_3d, &loss);

  std::vector<float> expected_logits = {
      0.216903,  0.192299,  0.709738,  0.015778,  0.065583,  -0.281998,
      -1.127933, 0.322185,  -0.480652, -0.533178, -0.062928, 0.123535,
      0.435241,  -0.103620, 0.005828,  -0.286582, -0.763048, 0.407561,
      -0.216112, -0.469412, -0.259017, 0.510337,  0.519246,  -0.298064,
      0.184779,  0.055917,  -0.883520, 0.297700,  -0.795307, -0.192879,
      0.090488,  0.989613,  0.485091,  -0.415317, 0.200022,  0.379014,
      -0.672618, -0.472669, -1.369164, -0.364831, 0.196027,  0.133897,
      0.728029,  -0.036481, 0.245341,  -0.258133, -1.119066, 0.406560,
      -0.388505, -0.505914, -0.096193, 0.023417,  0.352678,  -0.013193,
      -0.164390, -0.367955, -0.687712, 0.449032,  -0.084536, -0.430618,
      -0.296299, 0.513148,  0.478178,  -0.196040, -0.039647, 0.026517,
      -0.887767, 0.282241,  -0.849578, -0.103470, -0.003667, 0.883650,
      0.330532,  -0.294204, 0.039559,  0.409321,  -0.487800, -0.464272,
      -1.270504, -0.087506};
  for (size_t i = 0; i < expected_logits.size(); ++i) {
    EXPECT_NEAR(expected_logits[i], logits[i], 1e-5);
  }
  EXPECT_NEAR(loss, 2.280785, 1e-5);

  // Backward
  gpt.Backward(idx_m, target_m);
  auto wte_grad = absl::MakeSpan(gpt.wte_->weight_grad_.get(), vocab_size * C);
  auto wpe_grad = absl::MakeSpan(gpt.wpe_->weight_grad_.get(), block_size * C);

  std::vector<float> expected_wte_grad = {
      -0.071291, -0.008528, 0.145227,  0.131039,  -0.004713, -0.191734,
      -0.177156, -0.114964, 0.029494,  0.120911,  -0.022896, 0.164611,
      0.049951,  0.142361,  -0.146608, -0.181549, 0.029493,  0.106352,
      -0.013085, 0.062030,  -0.027398, -0.161614, 0.087651,  0.052417,
      0.043494,  0.049613,  -0.075816, -0.031530, -0.037264, 0.051504,
      0.081909,  -0.025799, -0.034390, 0.105682,  -0.076949, -0.050454,
      0.027316,  0.071168,  0.120285,  -0.014125, 0.004194,  -0.208838,
      -0.142323, -0.113452, 0.073817,  0.127261,  -0.021897, 0.076595,
      -0.063724, -0.052385, 0.035635,  0.061177,  -0.008474, 0.027771,
      0.114922,  0.037734,  -0.043024, 0.021416,  -0.019129, -0.111919};
  for (size_t i = 0; i < expected_wte_grad.size(); ++i) {
    EXPECT_NEAR(expected_wte_grad[i], wte_grad[i], 1e-5);
  }

  std::vector<float> expected_wpe_grad = {
      -0.041526, 0.041383,  -0.004762, 0.021982,  -0.001184, -0.015893,
      -0.000587, -0.013092, 0.025597,  -0.027305, 0.024447,  -0.009061,
      -0.050238, 0.023291,  0.003437,  0.081925,  0.008172,  -0.066587,
      -0.057636, -0.003805, 0.052951,  0.102064,  -0.101419, 0.007845};
  for (size_t i = 0; i < expected_wpe_grad.size(); ++i) {
    EXPECT_NEAR(expected_wpe_grad[i], wpe_grad[i], 1e-5);
  }
}
