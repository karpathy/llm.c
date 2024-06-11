#include "nn.hpp"
#include "gtest/gtest.h"

TEST(Random, UniformFill) {
  nn::ManualSeed(42);
  std::vector<float> expected_num = {0.882269, 0.915004, 0.382864, 0.959306,
                                     0.390448, 0.600895, 0.256572, 0.793641,
                                     0.940771, 0.133186};
  std::vector<float> num(10);
  nn::UniformFill(absl::MakeSpan(num));
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }
}

TEST(Random, NormalFill) {
  nn::ManualSeed(42);
  std::vector<float> expected_num = {0.336690,  0.128809,  0.234462, 0.230333,
                                     -1.122856, -0.186328, 2.208201, -0.637997,
                                     0.461657,  0.267351};
  std::vector<float> num(10);
  nn::NormalFill(absl::MakeSpan(num));
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }

  expected_num = {-0.758131, 1.078318,  0.800801,  1.680621,  0.355860,
                  -0.686623, -0.493356, 0.241488,  -0.231624, 0.041759,
                  -0.251575, 0.859859,  -0.309727, -0.395710, 0.803409,
                  -0.621595, 0.318880,  -0.424519, 0.305721,  -0.774593,
                  0.034912,  0.321103,  1.573600,  -0.845467, -1.274151,
                  2.122785,  -1.234653, -0.487914, -1.418060, 0.896268,
                  0.049905,  2.266718};
  num.resize(32);
  nn::NormalFill(absl::MakeSpan(num));
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }
}

TEST(Random, KaimingUniformFill) {
  nn::ManualSeed(42);
  int in_features = 4, out_features = 3, num_samples = 12;
  std::vector<float> expected_num = {0.382269,  0.415004,  -0.117136, 0.459306,
                                     -0.109552, 0.100895,  -0.243428, 0.293641,
                                     0.440771,  -0.366814, 0.434598,  0.093580};
  std::vector<float> num(num_samples);
  nn::KaimingUniformFill(absl::MakeSpan(num), in_features);
  for (size_t i = 0; i < expected_num.size(); ++i) {
    EXPECT_NEAR(expected_num[i], num[i], 1e-5);
  }
}

TEST(MatMul, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Linear(3, 2, bias=False)
x = torch.randn(4, 3)
x = nn.Parameter(x)
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  int B = 4, in_features = 3, out_features = 2;
  std::vector<float> x(B * in_features), w(out_features * in_features);
  nn::KaimingUniformFill(absl::MakeSpan(w), in_features);
  nn::NormalFill(absl::MakeSpan(x));

  // forward
  std::vector<float> y(B * out_features);
  auto xm = Eigen::Map<nn::Matrix>(x.data(), B, in_features);
  auto wm = Eigen::Map<nn::Matrix>(w.data(), out_features, in_features);
  auto ym = Eigen::Map<nn::Matrix>(y.data(), B, out_features);
  nn::MatMul::Forward(xm, wm, ym);

  std::vector<float> expected_y = {1.033869, 0.275703,  0.364816, 0.123444,
                                   0.565156, -0.005050, 1.014951, 0.552350};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad(y.size(), 1.0);
  std::vector<float> x_grad(x.size(), 0.0), w_grad(w.size(), 0.0);
  auto y_gradm = Eigen::Map<nn::Matrix>(y_grad.data(), B, out_features);
  auto x_gradm = Eigen::Map<nn::Matrix>(x_grad.data(), B, in_features);
  auto w_gradm =
      Eigen::Map<nn::Matrix>(w_grad.data(), out_features, in_features);
  nn::MatMul::Backward(xm, wm, y_gradm, x_gradm, w_gradm);

  std::vector<float> expected_w_grad = {3.267881, 1.874520, -4.717285,
                                        3.267881, 1.874520, -4.717285};
  std::vector<float> expected_x_grad = {
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753,
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753};

  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], w_grad[i], 1e-5);
  }
}

TEST(Linear, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Linear(3, 2)
x = torch.randn(4, 3)
x = nn.Parameter(x)
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  nn::ManualSeed(42);
  int B = 4, in_features = 3, out_features = 2;
  nn::Linear m(in_features, out_features, true);
  std::vector<float> x = {-1.122856, -0.186328, 2.208201,  -0.637997,
                          0.461657,  0.267351,  0.534905,  0.809357,
                          1.110290,  -1.689799, -0.988960, 0.957972};

  // forward
  std::vector<float> y(8);
  auto xm = Eigen::Map<nn::Matrix>(x.data(), B, in_features);
  auto ym = Eigen::Map<nn::Matrix>(y.data(), B, out_features);
  m.Forward(xm, ym);

  std::vector<float> expected_y = {-1.164687, 0.024384, -0.377635, -0.026553,
                                   0.192698,  0.649730, -1.630462, -0.320424};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-4);
  }

  // backward
  std::vector<float> y_grad(y.size(), 1.0f);
  std::vector<float> x_grad(x.size(), 0.f);
  auto y_gradm = Eigen::Map<nn::Matrix>(y_grad.data(), B, out_features);
  auto x_gradm = Eigen::Map<nn::Matrix>(x_grad.data(), B, in_features);
  m.Backward(xm, y_gradm, x_gradm);

  std::vector<float> expected_w_grad = {-2.915748, 0.095726, 4.543815,
                                        -2.915748, 0.095726, 4.543815};
  std::vector<float> expected_b_grad = {4, 4};
  std::vector<float> expected_x_grad = {
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753,
      0.971767, 0.352706, -0.018753, 0.971767, 0.352706, -0.018753};

  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], m.weight_grad_.data()[i], 1e-5);
  }
  for (size_t i = 0; i < expected_b_grad.size(); ++i) {
    EXPECT_NEAR(expected_b_grad[i], m.bias_grad_.data()[i], 1e-5);
  }
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(Embedding, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
m = nn.Embedding(10, 3)
idx = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
emb = m(idx)
loss = torch.sum(emb)
loss.backward()
*/

  nn::ManualSeed(42);
  int vocab_size = 10, dim = 3;
  nn::Embedding m(vocab_size, dim);

  std::vector<int> idx = {1, 2, 4, 5, 4, 3, 2, 9};
  std::vector<float> embedding(idx.size() * dim);
  m.Forward(idx, absl::MakeSpan(embedding));

  std::vector<float> expected_embedding = {
      -2.105521, 0.678418,  -1.234545, -0.043067, -1.604667, -0.752135,
      -0.727881, -0.559430, -2.316923, -0.216805, -1.384674, -0.871236,
      -0.727881, -0.559430, -2.316923, 1.648723,  -0.392479, -1.403607,
      -0.043067, -1.604667, -0.752135, -0.601142, -1.274151, 2.122785};
  for (size_t i = 0; i < expected_embedding.size(); ++i) {
    EXPECT_NEAR(expected_embedding[i], embedding[i], 1e-5);
  }

  std::vector<float> expected_w_grad = {0., 0., 0., 1., 1., 1., 2., 2., 2., 1.,
                                        1., 1., 2., 2., 2., 1., 1., 1., 0., 0.,
                                        0., 0., 0., 0., 0., 0., 0., 1., 1., 1.};
  std::vector<float> grad_embedding(idx.size() * dim, 1.0f);
  m.Backward(idx, absl::MakeSpan(grad_embedding));

  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], m.weight_grad_[i], 1e-5);
  }
}

TEST(LayerNorm, ForwardAndBackward) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
batch, sentence_length, embedding_dim = 4, 2, 3
x = torch.randn(batch, sentence_length, embedding_dim)
x = nn.Parameter(x)
m = nn.LayerNorm(embedding_dim)
y = m(x)
coeff = torch.Tensor([1,2,3])
loss = torch.sum(y * coeff)
loss.backward()
  */

  int batch = 4, sentence_length = 2, embedding_dim = 3;
  auto m = nn::LayerNorm(embedding_dim);
  std::vector<float> x = {1.926915,  1.487284,  0.900717,  -2.105521, 0.678418,
                          -1.234545, -0.043067, -1.604667, 0.355860,  -0.686623,
                          -0.493356, 0.241488,  -1.110904, 0.091546,  -2.316923,
                          -0.216805, -0.309727, -0.395710, 0.803409,  -0.621595,
                          -0.592001, -0.063074, -0.828554, 0.330898};
  Eigen::Map<nn::Matrix> x_m(x.data(), batch * sentence_length, embedding_dim);
  int row_size = batch * sentence_length;
  std::vector<float> y(x.size(), 0), mean(row_size, 0), rstd(row_size, 0);
  Eigen::Map<nn::Matrix> y_m(y.data(), row_size, embedding_dim);
  Eigen::Map<Eigen::RowVectorXf> mean_m(mean.data(), row_size),
      rstd_m(rstd.data(), row_size);
  m.Forward(x_m, y_m, mean_m, rstd_m);
  std::vector<float> expected_y = {
      1.162292e+00,  1.165090e-01,  -1.278801e+00, -1.047755e+00, 1.346462e+00,
      -2.987067e-01, 4.581040e-01,  -1.387752e+00, 9.296476e-01,  -9.348618e-01,
      -4.514984e-01, 1.386360e+00,  1.210087e-03,  1.224133e+00,  -1.225343e+00,
      1.239106e+00,  -3.162789e-02, -1.207479e+00, 1.413964e+00,  -7.292372e-01,
      -6.847267e-01, 2.572481e-01,  -1.332909e+00, 1.075661e+00};
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad(x.size(), 0), x_grad(x.size(), 0);
  for (int i = 0; i < row_size; ++i) {
    y_grad[i * embedding_dim + 0] = 1.0;
    y_grad[i * embedding_dim + 1] = 2.0;
    y_grad[i * embedding_dim + 2] = 3.0;
  }
  Eigen::Map<nn::Matrix> y_grad_m(y_grad.data(), row_size, embedding_dim);
  Eigen::Map<nn::Matrix> x_grad_m(x_grad.data(), row_size, embedding_dim);
  m.Backward(x_m, y_grad_m, mean_m, rstd_m, x_grad_m);

  std::vector<float> expected_w_grad = {2.549308, -2.491841, -3.910163},
                     expected_b_grad = {8., 16., 24.},
                     expected_x_grad = {
                         -0.129036, 0.225515,  -0.096478, -0.635026, -0.289126,
                         0.924152,  -1.267141, 0.257834,  1.009307,  -0.691930,
                         0.873713,  -0.181782, -1.016524, 0.509009,  0.507514,
                         0.143953,  -0.352736, 0.208773,  -0.016307, -0.767260,
                         0.783567,  -2.223118, 0.755368,  1.467751};
  for (size_t i = 0; i < expected_w_grad.size(); ++i) {
    EXPECT_NEAR(expected_w_grad[i], m.weight_grad_[i], 1e-5);
  }
  for (size_t i = 0; i < expected_b_grad.size(); ++i) {
    EXPECT_NEAR(expected_b_grad[i], m.bias_grad_[i], 1e-5);
  }
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 4 * 1e-5);
  }
}

TEST(NewGELU, ForwardAndBackward) {
  /*
class NewGELU(nn.Module):
  def forward(self, input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input +
0.044715 * torch.pow(input, 3.0))))

torch.manual_seed(42)
batch, dim = 4, 3
x = torch.randn(batch, dim)
x = nn.Parameter(x)
m = NewGELU()
y = m(x)
loss = torch.sum(y)
loss.backward()
  */

  // forward
  std::vector<float> x = {0.336690,  0.128809,  0.234462, 0.230333,
                          -1.122856, -0.186328, 2.208201, -0.637997,
                          0.461657,  0.267351,  0.534905, 0.809357};
  std::vector<float> expected_y = {0.212725,  0.071006,  0.138962, 0.136145,
                                   -0.147006, -0.079394, 2.178409, -0.167029,
                                   0.312915,  0.161853,  0.376359, 0.639989};
  std::vector<float> y(x.size(), 0);
  nn::NewGELU m;
  m.Forward(x, absl::MakeSpan(y));
  for (size_t i = 0; i < expected_y.size(); ++i) {
    EXPECT_NEAR(expected_y[i], y[i], 1e-5);
  }

  // backward
  std::vector<float> y_grad(x.size(), 1.0f), x_grad(x.size(), 0.f);
  m.Backward(x, y_grad, absl::MakeSpan(x_grad));
  std::vector<float> expected_x_grad = {
      0.758699, 0.602206, 0.683672, 0.680552, -0.107435, 0.353047,
      1.064087, 0.054299, 0.843291, 0.708290, 0.888446,  1.023232};
  for (size_t i = 0; i < expected_x_grad.size(); ++i) {
    EXPECT_NEAR(expected_x_grad[i], x_grad[i], 1e-5);
  }
}

TEST(CrossEntropy, Forward) {
  /*

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42)
torch.set_printoptions(precision=6)
batch, dim = 4, 3
x = torch.randn(batch, dim)
x = nn.Parameter(x)
target = torch.LongTensor([1, 2, 1, 0])
l = nn.CrossEntropyLoss()
loss = l(x, target)
loss.backward()

  */

  nn::ManualSeed(42);
  int batch = 4, dim = 3;
  // forward
  std::vector<float> logits(batch * dim), probs(batch * dim),
      logits_grad(batch * dim, 0);
  nn::NormalFill(absl::MakeSpan(logits));
  std::vector<int> target = {1, 2, 1, 0};

  auto logits_m = Eigen::Map<nn::Matrix>(logits.data(), batch, dim);
  auto probs_m = Eigen::Map<nn::Matrix>(probs.data(), batch, dim);
  float loss1 = 0.0, loss2 = 0.0;

  // Reduction: MEAN
  nn::CrossEntropy criterion1(nn::CrossEntropy::MEAN);
  criterion1.Forward(logits_m, absl::MakeSpan(target), probs_m, &loss1);

  auto logits_grad_m = Eigen::Map<nn::Matrix>(logits_grad.data(), batch, dim);
  criterion1.Backward(probs_m, absl::MakeSpan(target), logits_grad_m);

  std::vector<float> expected_logits_grad1 = {
      0.092077, -0.175206, 0.083129, 0.130367,  0.033689, -0.164056,
      0.202850, -0.238222, 0.035372, -0.187907, 0.081141, 0.106766};

  for (size_t i = 0; i < expected_logits_grad1.size(); ++i) {
    EXPECT_NEAR(expected_logits_grad1[i], logits_grad[i], 1e-5);
  }

  // Reduction: SUM
  nn::CrossEntropy criterion2(nn::CrossEntropy::SUM);
  criterion2.Forward(logits_m, absl::MakeSpan(target), probs_m, &loss2);
  EXPECT_NEAR(loss1 * batch, loss2, 1e-5);

  logits_grad_m.setZero();
  criterion2.Backward(probs_m, absl::MakeSpan(target), logits_grad_m);
  std::vector<float> expected_logits_grad2 = {
      0.368307, -0.700823, 0.332516, 0.521469,  0.134755, -0.656224,
      0.811398, -0.952886, 0.141488, -0.751628, 0.324564, 0.427064};

  for (size_t i = 0; i < expected_logits_grad2.size(); ++i) {
    EXPECT_NEAR(expected_logits_grad2[i], logits_grad[i], 1e-5);
  }
}
