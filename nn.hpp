#ifndef LLM_CPP__NN_HPP_
#define LLM_CPP__NN_HPP_

#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llmc/rand.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace nn {

mt19937_state g_mt19937_state;

void ManualSeed(unsigned int seed) { manual_seed(&g_mt19937_state, seed); }

void UniformFill(absl::Span<float> weight, float from = 0.0, float to = 1.0) {
  uniform_(weight.data(), weight.size(), from, to, &g_mt19937_state);
}

void NormalFill(absl::Span<float> weight, float mean = 0.0, float std = 1.0) {
  normal_(weight.data(), weight.size(), mean, std, &g_mt19937_state);
}

void KaimingUniformFill(absl::Span<float> weight, int in_features) {
  const float bound = std::sqrt(1.0f / in_features);
  uniform_(weight.data(), weight.size(), -bound, bound, &g_mt19937_state);
}

std::string DebugString(absl::Span<const float> span, int num_elements) {
  std::stringstream ss;
  ss << std::setprecision(6) << std::fixed << "[";
  auto x1 = span.subspan(0, num_elements / 2);
  auto x2 = span.subspan(span.size() - (num_elements - num_elements / 2));
  bool first = true;

  auto print_fn = [&](absl::Span<const float> x) {
    for (int i = 0; i < x.size(); ++i) {
      ss << x[i];
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
    }
  };

  print_fn(x1);
  print_fn(x2);
  ss << "]";

  return ss.str();
}

using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixInt =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

inline Eigen::Map<Matrix> GetMatrixMap(Matrix& m) {
  return Eigen::Map<Matrix>(m.data(), m.rows(), m.cols());
}

// Parameter weight and its corresponding gradient
struct Parameter {
  enum DataType { kValue, kGrad };

  Parameter(int64_t length) : length_(length) {
    value_ = std::make_unique<float[]>(length);
    grad_ = nullptr;
  }

  int64_t size() const { return length_; }
  float* data() const { return value_.get(); }
  float* grad() const { return grad_.get(); }
  const std::string& name() const { return name_; }

  void SetOffset(int offset) { offset_ = offset; }

  void SetName(const std::string& name) { name_ = name; }
  void AllocateGradient() {
    if (grad_ == nullptr) {
      grad_ = std::make_unique<float[]>(length_);
    }
  }

  void ZeroGrad() {
    if (grad_ != nullptr) {
      std::memset(grad_.get(), 0, sizeof(float) * length_);
    }
  }

  absl::Span<float> View(DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    return {type == kValue ? value_.get() : grad_.get(),
            static_cast<size_t>(length_)};
  }

  Eigen::Map<Eigen::RowVectorXf> View(int length,
                                      DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    CHECK_EQ(length, length_);
    return {type == kValue ? value_.get() : grad_.get(),
            static_cast<Eigen::Index>(length_)};
  }

  Eigen::Map<nn::Matrix> View(int rows, int cols,
                              DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    CHECK_EQ(rows * cols, length_);
    return {type == kValue ? value_.get() : grad_.get(), rows, cols};
  }

 private:
  std::unique_ptr<float[]> value_;
  std::unique_ptr<float[]> grad_;
  int64_t length_;
  std::string name_;
  int offset_ = 0;
};

struct MatMul {
  static void Forward(const Eigen::Map<Matrix>& x1,
                      const Eigen::Map<Matrix>& x2, Eigen::Map<Matrix>& y) {
    // x: [M, N], x2: [N, K], y: [M, K]
    CHECK_EQ(x1.rows(), y.rows());
    CHECK_EQ(x1.cols(), x2.rows());
    CHECK_EQ(x2.cols(), y.cols());

    // y = x1 * x2
    y.noalias() = x1 * x2;
  }

  static void Backward(const Eigen::Map<Matrix>& x1,
                       const Eigen::Map<Matrix>& x2,
                       const Eigen::Map<Matrix>& y_grad,
                       Eigen::Map<Matrix>& x1_grad,
                       Eigen::Map<Matrix>& x2_grad) {
    // input:
    // x1: [M, N], x2:[N, K]
    // y_grad: [M, K]
    //
    // output:
    // x1_grad: [M, N], x2_grad: [N, K]
    int M = x1.rows(), N = x1.cols(), K = x2.cols();
    CHECK(M == y_grad.rows() && M == x1_grad.rows());
    CHECK(N == x2.rows() && N == x1_grad.cols() && N == x2_grad.rows());
    CHECK(K == y_grad.cols() && K == x2_grad.cols());

    // x1_grad = dL/dy * dy/dx1
    //        = y_grad(M, K) * x2^T (K, N)
    //        = [M, N]
    x1_grad.noalias() += y_grad * x2.transpose();

    // x2_grad = dL/dy * dy/dx2
    //        = x1^T(N, M) * y_grad(M, K)
    //        = [N, K]
    x2_grad.noalias() += x1.transpose() * y_grad;
  }
};

struct Residual {
  static void Forward(absl::Span<const float> x, absl::Span<const float> Fx,
                      absl::Span<float> Hx) {
    int N = x.size();
    CHECK(N == Fx.size() && N == Hx.size());

    // H(x) = x + F(x) -> F(x) = H(x) - x
    for (int i = 0; i < N; ++i) {
      Hx[i] = x[i] + Fx[i];
    }
  }

  static void Backward(absl::Span<const float> Hx_grad,
                       absl::Span<float> x_grad, absl::Span<float> Fx_grad) {
    int N = Hx_grad.size();
    CHECK(N == x_grad.size() && N == Fx_grad.size());

    for (int i = 0; i < N; ++i) {
      x_grad[i] += Hx_grad[i];
      Fx_grad[i] += Hx_grad[i];
    }
  }
};

struct Linear {
  Linear(int in_features, int out_features, bool bias = true)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias) {
    weight_ = std::make_unique<Parameter>(out_features * in_features);
    KaimingUniformFill(weight_->View(), in_features);
    if (bias) {
      bias_ = std::make_unique<Parameter>(out_features);
      const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
      UniformFill(bias_->View(), -bound, bound);
    }
  }

  void Forward(const Eigen::Map<Matrix>& x, Eigen::Map<Matrix>& y) const {
    // x: [B, in_features], y: [B, out_features]
    CHECK_EQ(x.cols(), in_features_);
    CHECK_EQ(y.cols(), out_features_);
    CHECK_EQ(x.rows(), y.rows());

    auto weight = weight_->View(out_features_, in_features_);
    // y = x * w^T + b
    if (has_bias_) {
      auto bias = bias_->View(out_features_);
      y.noalias() = (x * weight.transpose()).rowwise() + bias;
    } else {
      y.noalias() = x * weight.transpose();
    }
  }

  void Backward(const Eigen::Map<Matrix>& x, const Eigen::Map<Matrix>& y_grad,
                Eigen::Map<Matrix>& x_grad) {
    // x: [B, in_features], y_grad: [B, out_features], x_grad: [B, in_features]
    CHECK_EQ(x.cols(), in_features_);
    CHECK_EQ(y_grad.cols(), out_features_);
    CHECK_EQ(x.rows(), y_grad.rows());
    CHECK_EQ(x.rows(), x_grad.rows());

    // Lazily allocate the memory for gradients
    weight_->AllocateGradient();
    bias_->AllocateGradient();
    auto weight = weight_->View(out_features_, in_features_);
    auto bias = bias_->View(out_features_);
    auto weight_grad =
        weight_->View(out_features_, in_features_, Parameter::kGrad);
    auto bias_grad = bias_->View(out_features_, Parameter::kGrad);

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    x_grad.noalias() += y_grad * weight;

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    weight_grad.noalias() += y_grad.transpose() * x;

    if (has_bias_) {
      // b_grad = dL/dy * dy/db
      //        = \sum_i^(B)(y_grad(B, out_features))
      //        = [out_features,]
      bias_grad.noalias() += y_grad.colwise().sum();
    }
  }

  size_t NumParameters() const {
    size_t num_parameters = out_features_ * in_features_;
    if (has_bias_) {
      num_parameters += out_features_;
    }

    return num_parameters;
  }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
    if (has_bias_) {
      parameters->push_back(bias_.get());
    }
  }

  bool has_bias_;
  int in_features_;
  int out_features_;
  std::unique_ptr<Parameter> weight_;  // out_features x in_features
  std::unique_ptr<Parameter> bias_;    // out_features
};

struct Embedding {
  Embedding(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    weight_ = std::make_unique<Parameter>(num_embeddings * embedding_dim);
    NormalFill(weight_->View());
  }

  void Forward(absl::Span<const int> idx, absl::Span<float> embedding) const {
    CHECK_EQ(embedding.size(), idx.size() * embedding_dim_);
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      void* dst = embedding.data() + i * embedding_dim_;
      void* src = weight_->data() + idx[i] * embedding_dim_;
      std::memcpy(dst, src, sizeof(float) * embedding_dim_);
    }
  }

  void Backward(absl::Span<const int> idx,
                absl::Span<const float> grad_embedding) {
    CHECK_EQ(grad_embedding.size(), idx.size() * embedding_dim_);

    // Lazily allocate the memory for gradients
    weight_->AllocateGradient();

    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      const float* g = grad_embedding.data() + i * embedding_dim_;
      float* grad = weight_->grad() + idx[i] * embedding_dim_;
      for (int j = 0; j < embedding_dim_; ++j) {
        grad[j] += g[j];
      }
    }
  }

  size_t NumParameters() const { return num_embeddings_ * embedding_dim_; }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
  }

  int num_embeddings_;
  int embedding_dim_;
  std::unique_ptr<Parameter> weight_;
};

struct LayerNorm {
  LayerNorm(int normalized_shape)
      : normalized_shape_(normalized_shape), eps_(1e-5) {
    weight_ = std::make_unique<Parameter>(normalized_shape);
    auto w = weight_->View();
    absl::c_fill(w, 1.0f);
    bias_ = std::make_unique<Parameter>(normalized_shape);
  }

  void Forward(const Eigen::Map<Matrix>& x, Eigen::Map<Matrix>& y,
               Eigen::Map<Eigen::RowVectorXf>& mean,
               Eigen::Map<Eigen::RowVectorXf>& rstd) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.cols(), normalized_shape_);
    CHECK_EQ(y.cols(), normalized_shape_);
    CHECK_EQ(x.rows(), y.rows());
    int B = x.rows();

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);
    mean.noalias() = x.rowwise().mean();

    // x_zero_centered(B, D) = x.colwise() - m.transpose()
    // x_zero_centered_square(B, D) = x_zero_centered.array().square()
    // var(B,) = x_zero_centered_square.rowwise().mean()
    // std(B,) = (var + eps).sqrt()
    // rstd(B,) = 1.f / std;
    rstd = 1.f /
           ((x.colwise() - mean.transpose()).array().square().rowwise().mean() +
            eps_)
               .sqrt();

    auto weight = weight_->View(normalized_shape_);
    auto bias = bias_->View(normalized_shape_);
    // normalize: (x - mean) / std
    // && scale:  (x - mean) / std * weight
    // && shift:  (x - mean) / std * weight + bias
    y = (((x.colwise() - mean.transpose()).array().colwise() *
          rstd.transpose().array())
             .array()
             .rowwise() *
         weight.array())
            .array()
            .rowwise() +
        bias.array();
  }

  void Backward(const Eigen::Map<Matrix>& x, const Eigen::Map<Matrix>& y_grad,
                const Eigen::Map<Eigen::RowVectorXf>& mean,
                const Eigen::Map<Eigen::RowVectorXf>& rstd,
                Eigen::Map<Matrix>& x_grad) {
    // x: [B, D], y_grad: [B, D], x_grad: [B, D]
    CHECK_EQ(x.cols(), normalized_shape_);
    CHECK_EQ(y_grad.cols(), normalized_shape_);
    CHECK_EQ(x_grad.cols(), normalized_shape_);
    CHECK_EQ(x.rows(), y_grad.rows());
    CHECK_EQ(x.rows(), x_grad.rows());
    int B = x.rows();

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    // Lazily allocate the memory for gradients
    weight_->AllocateGradient();
    bias_->AllocateGradient();
    auto weight = weight_->View(normalized_shape_);
    auto weight_grad = weight_->View(normalized_shape_, Parameter::kGrad);
    auto bias = bias_->View(normalized_shape_);
    auto bias_grad = bias_->View(normalized_shape_, Parameter::kGrad);

    // x_grad = dL/dy * dy/dnorm
    //                * [dnorm/dxmean * dxmean/dx
    //                  + dnorm/dmean * dmean/dx
    //                  + dnorm/dstd * dstd/dx
    //                  ]
    nn::Matrix norm = (x.colwise() - mean.transpose()).array().colwise() *
                      rstd.transpose().array();                    // [B, D]
    nn::Matrix dnorm = y_grad.array().rowwise() * weight.array();  // [B, D]
    Eigen::RowVectorXf dnorm_mean = dnorm.rowwise().mean();        // [B,]
    Eigen::RowVectorXf dnorm_norm_mean =
        (dnorm.array() * norm.array()).rowwise().mean();  // [B,]
    x_grad.array() +=
        ((dnorm.array().colwise() - dnorm_mean.transpose().array()).array() -
         (norm.array().colwise() * dnorm_norm_mean.transpose().array()))
            .array()
            .colwise() *
        rstd.transpose().array();
    //    std::cout << "x_grad: " << x_grad << std::endl;

    // w_grad = dL/dy * dy/dw
    //        = dL/dy * x_norm(B,D)
    //        = \sum_i^B [y_grad(B, D) \elewise_dot x_norm(B, D)]
    weight_grad.array() +=
        (y_grad.array() * ((x.colwise() - mean.transpose()).array().colwise() *
                           rstd.transpose().array())
                              .array())
            .colwise()
            .sum()
            .array();

    // b_grad = dL/dy * dy/db
    //        = \sum_i^(B)(y_grad(B, D))
    //        = [D,]
    bias_grad.noalias() += y_grad.colwise().sum();
  }

  size_t NumParameters() const { return normalized_shape_ * 2; }

  void Parameters(std::vector<Parameter*>* parameters) const {
    parameters->push_back(weight_.get());
    parameters->push_back(bias_.get());
  }

  int normalized_shape_;
  float eps_;
  std::unique_ptr<Parameter> weight_;
  std::unique_ptr<Parameter> bias_;
};

// Careful there are a few versions of GeLU, this one is the exact one used by
// OpenAI
struct NewGELU {
  void Forward(absl::Span<const float> x, absl::Span<float> y) {
    CHECK_EQ(x.size(), y.size());
    const float sqrt_2_over_pi = std::sqrt(M_2_PI);

    // y = 0.5 * x * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    for (size_t i = 0; i < x.size(); ++i) {
      float _x = x[i];
      float cube = 0.044715f * _x * _x * _x;
      y[i] = 0.5f * _x * (1.0f + std::tanh(sqrt_2_over_pi * (_x + cube)));
    }
  }

  void Backward(absl::Span<const float> x, absl::Span<const float> y_grad,
                absl::Span<float> x_grad) {
    CHECK_EQ(x.size(), y_grad.size());
    CHECK_EQ(x.size(), x_grad.size());

    // dL/dx = dL/dy * dy/dx
    //       = dL/dy * [ 0.5 * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    //                 + 0.5 * x * (1 - (tanh[sqrt(2/pi) * (x + 0.044715 *
    //                 x^3)])^2
    //                           *  (sqrt(2/pi) * (1 + 0.044715 * 3 * x^2))
    //                             )
    //                 ]
    const float sqrt_2_over_pi = std::sqrt(M_2_PI);
    for (size_t i = 0; i < x.size(); ++i) {
      float _x = x[i];
      float cube = 0.044715f * _x * _x * _x;
      float tanh_arg = sqrt_2_over_pi * (_x + cube);
      float tanh_out = std::tanh(tanh_arg);
      float dydx = 0.5f * (1.0f + tanh_out) +
                   0.5f * _x * (1.0f - tanh_out * tanh_out) *
                       (sqrt_2_over_pi * (1.0f + 3.f * 0.044715f * _x * _x));
      x_grad[i] += y_grad[i] * dydx;
    }
  }
};

struct Softmax {
  Softmax(bool stable_softmax = true) : stable_softmax_(stable_softmax) {}

  void Forward(const Eigen::Map<Matrix>& x, Eigen::Map<Matrix>& y) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.rows(), y.rows());
    CHECK_EQ(x.cols(), y.cols());

    if (stable_softmax_) {
      auto x_exp = (x.colwise() - x.rowwise().maxCoeff()).array().exp();
      y = x_exp.array().colwise() / x_exp.rowwise().sum().array();
    } else {
      auto x_exp = x.array().exp();
      y = x_exp.array().colwise() / x_exp.rowwise().sum().array();
    }
  }

  void Backward(const Eigen::Map<Matrix>& y, const Eigen::Map<Matrix>& y_grad,
                Eigen::Map<Matrix>& x_grad) {
    // y:[B, D], y_grad: [B, D], x_grad: [B, D]
    int B = y.rows(), D = y.cols();
    CHECK(B == y_grad.rows() && B == x_grad.rows());
    CHECK(D == y_grad.cols() && D == x_grad.cols());

    // dy_j / dx_i = S_i(1 - S_j) for i==j
    //             = -S_j*S_i     for i!=j
    // dL/dx_i = \sum_j dL/dy_j * dy_j / dx_i

    for (int b = 0; b < B; ++b) {
      for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
          float indicator = i == j ? 1.0f : 0.0f;
          x_grad(b, i) += y_grad(b, j) * y(b, i) * (indicator - y(b, j));
        }
      }
    }
  }

  bool stable_softmax_;
};

struct SoftmaxCrossEntropy {
  enum Reduction { MEAN, SUM };

  SoftmaxCrossEntropy(Reduction reduction = Reduction::MEAN,
                      bool stable_softmax = false)
      : reduction_(reduction) {
    softmax_ = std::make_unique<Softmax>(stable_softmax);
  }

  void Forward(const Eigen::Map<Matrix>& logits, absl::Span<const int> targets,
               Eigen::Map<Matrix>& probs, float* loss) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.rows(), C = logits.cols();
    CHECK(B == targets.size() && B == probs.rows());
    CHECK_EQ(C, probs.cols());

    // apply softmax to convert logits to (normalized) probabilities
    softmax_->Forward(logits, probs);

    // targets: [B,]
    for (int i = 0; i < targets.size(); ++i) {
      int ix = targets[i];
      *loss += -std::log(probs(i, ix));
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(const Eigen::Map<Matrix>& probs, absl::Span<const int> targets,
                Eigen::Map<Matrix>& logits_grad) {
    // probs: [B, C], targets: [B,]
    // logits_grad: [B, C]
    int B = probs.rows(), C = probs.cols();
    CHECK(B == targets.size() && B == logits_grad.rows());
    CHECK_EQ(C, logits_grad.cols());

    float factor =
        reduction_ == Reduction::MEAN ? 1.0f / static_cast<float>(B) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      for (int c = 0; c < C; ++c) {
        float indicator = c == ix ? 1.0f : 0.0f;
        logits_grad(b, c) += (probs(b, c) - indicator) * factor;
      }
    }
  }

  Reduction reduction_;
  std::unique_ptr<Softmax> softmax_;
};

struct VanillaCrossEntropy {
  enum Reduction { MEAN, SUM };

  VanillaCrossEntropy(Reduction reduction = Reduction::MEAN)
      : reduction_(reduction) {}

  void Forward(const Eigen::Map<Matrix>& probs, absl::Span<const int> targets,
               float* loss) {
    // probs:[B, C], targets: [B,] loss: scalar
    int B = probs.rows(), C = probs.cols();
    CHECK_EQ(B, targets.size());

    // targets: [B,]
    for (int i = 0; i < targets.size(); ++i) {
      int ix = targets[i];
      *loss += -std::log(probs(i, ix));
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(const Eigen::Map<Matrix>& probs, absl::Span<const int> targets,
                Eigen::Map<Matrix>& probs_grad) {
    // probs: [B, C], targets: [B,]
    // probs_grad: [B, C]
    int B = probs.rows(), C = probs.cols();
    CHECK(B == targets.size() && B == probs_grad.rows());
    CHECK_EQ(C, probs_grad.cols());

    float factor =
        reduction_ == Reduction::MEAN ? 1.0f / static_cast<float>(B) : 1.0f;

    for (int b = 0; b < B; ++b) {
      int ix = targets[b];
      probs_grad(b, ix) += -1.0f / probs(b, ix) * factor;
    }
  }

  Reduction reduction_;
};

}  // namespace nn

#endif  // LLM_CPP__NN_HPP_
