#include <unistd.h>
#include <iostream>
#include <memory>
#include <random>

#include "Eigen/Core"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "glog/logging.h"
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

struct MatMul {
  static void Forward(const Eigen::Map<Matrix>& x, const Eigen::Map<Matrix>& w,
                      Eigen::Map<Matrix>& y) {
    // x: [B, in_features], w: [out_features, in_features], y: [B, out_features]
    CHECK_EQ(x.rows(), y.rows());  // B
    CHECK_EQ(x.cols(), w.cols());  // in_features
    CHECK_EQ(w.rows(), y.cols());  // out_features

    // y = x * w^T
    y.noalias() = x * w.transpose();
  }

  static void Backward(const Eigen::Map<Matrix>& x, const Eigen::Map<Matrix>& w,
                       const Eigen::Map<Matrix>& y_grad,
                       Eigen::Map<Matrix>& x_grad, Eigen::Map<Matrix>& w_grad) {
    // input:
    // x: [B, in_features], w:[out_features, in_features]
    // y_grad: [B, out_features]
    //
    // output:
    // x_grad: [B, in_features], w_grad: [out_features, in_features]
    CHECK(x.rows() == y_grad.rows() && x.rows() == x_grad.rows());  // B
    CHECK(x.cols() == w.cols() && x.cols() == x_grad.cols() &&
          x.cols() == w_grad.cols());  // in_features
    CHECK(w.rows() == y_grad.cols() &&
          w.rows() == w_grad.rows());  // out_features

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    x_grad.noalias() += y_grad * w;

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    w_grad.noalias() += y_grad.transpose() * x;
  }
};

struct Linear {
  Linear(int in_features, int out_features, bool bias = true)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias) {
    weight_ = Matrix::Zero(out_features, in_features);
    bias_ = Eigen::RowVectorXf::Zero(out_features);
    KaimingUniformFill(absl::MakeSpan(weight_.data(), weight_.size()),
                       in_features);
    if (bias) {
      const float bound = 1.f / std::sqrt(in_features);
      UniformFill(absl::MakeSpan(bias_.data(), bias_.size()), -bound, bound);
    }
  }

  void Forward(const Eigen::Map<Matrix>& x, Eigen::Map<Matrix>& y) const {
    // x: [B, in_features], y: [B, out_features]
    CHECK_EQ(x.cols(), in_features_);
    CHECK_EQ(y.cols(), out_features_);
    CHECK_EQ(x.rows(), y.rows());

    // y = x * w^T + b
    y.noalias() = (x * weight_.transpose()).rowwise() + bias_;
  }

  void Backward(const Eigen::Map<Matrix>& x, const Eigen::Map<Matrix>& y_grad,
                Eigen::Map<Matrix>& x_grad) {
    // x: [B, in_features], y_grad: [B, out_features], x_grad: [B, in_features]
    CHECK_EQ(x.cols(), in_features_);
    CHECK_EQ(y_grad.cols(), out_features_);
    CHECK_EQ(x.rows(), y_grad.rows());
    CHECK_EQ(x.rows(), x_grad.rows());

    // Lazily allocate the memory for gradients
    LazilyAllocateGradMemory();

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    x_grad.noalias() += y_grad * weight_;

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    weight_grad_.noalias() += y_grad.transpose() * x;

    if (has_bias_) {
      // b_grad = dL/dy * dy/db
      //        = \sum_i^(B)(y_grad(B, out_features))
      //        = [out_features,]
      bias_grad_.noalias() += y_grad.colwise().sum();
    }
  }

  void LazilyAllocateGradMemory() {
    if (weight_grad_.size() == 0) {
      weight_grad_ = Matrix::Zero(out_features_, in_features_);
    }
    if (bias_grad_.size() == 0) {
      bias_grad_ = Eigen::RowVectorXf::Zero(out_features_);
    }
  }

  size_t NumParameters() const {
    size_t num_parameters = out_features_ * in_features_;
    if (has_bias_) {
      num_parameters += out_features_;
    }

    return num_parameters;
  }

  bool has_bias_;
  int in_features_;
  int out_features_;
  Matrix weight_, weight_grad_;          // out_features x in_features
  Eigen::RowVectorXf bias_, bias_grad_;  // out_features
};

struct Embedding {
  Embedding(int num_embeddings, int embedding_dim)
      : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    weight_ = std::make_unique<float[]>(num_embeddings * embedding_dim);
    NormalFill(absl::MakeSpan(weight_.get(), num_embeddings * embedding_dim));
  }

  void Forward(absl::Span<const int> idx, absl::Span<float> embedding) {
    CHECK_EQ(embedding.size(), idx.size() * embedding_dim_);
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      void* dst = embedding.data() + i * embedding_dim_;
      void* src = weight_.get() + idx[i] * embedding_dim_;
      std::memcpy(dst, src, sizeof(float) * embedding_dim_);
    }
  }

  void Backward(absl::Span<const int> idx,
                absl::Span<const float> grad_embedding) {
    CHECK_EQ(grad_embedding.size(), idx.size() * embedding_dim_);

    // Lazily allocate the memory for gradients
    LazilyAllocateGradMemory();

    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      const float* g = grad_embedding.data() + i * embedding_dim_;
      float* grad = weight_grad_.get() + idx[i] * embedding_dim_;
      for (int j = 0; j < embedding_dim_; ++j) {
        grad[j] += g[j];
      }
    }
  }

  void LazilyAllocateGradMemory() {
    if (weight_grad_ == nullptr) {
      weight_grad_ =
          std::make_unique<float[]>(num_embeddings_ * embedding_dim_);
    }
  }

  size_t NumParameters() const { return num_embeddings_ * embedding_dim_; }

  int num_embeddings_;
  int embedding_dim_;
  std::unique_ptr<float[]> weight_;
  std::unique_ptr<float[]> weight_grad_;
};

struct LayerNorm {
  LayerNorm(int normalized_shape)
      : normalized_shape_(normalized_shape), eps_(1e-5) {
    weight_ = Eigen::RowVectorXf::Ones(normalized_shape);
    bias_ = Eigen::RowVectorXf::Zero(normalized_shape);
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
    // normalize: (x - mean) / std
    // && scale:  (x - mean) / std * weight
    // && shift:  (x - mean) / std * weight + bias
    y = (((x.colwise() - mean.transpose()).array().colwise() *
          rstd.transpose().array())
             .array()
             .rowwise() *
         weight_.array())
            .array()
            .rowwise() +
        bias_.array();
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
    LazilyAllocateGradMemory();

    // x_grad = dL/dy * dy/dnorm
    //                * [dnorm/dxmean * dxmean/dx
    //                  + dnorm/dmean * dmean/dx
    //                  + dnorm/dstd * dstd/dx
    //                  ]
    nn::Matrix norm = (x.colwise() - mean.transpose()).array().colwise() *
                      rstd.transpose().array();                     // [B, D]
    nn::Matrix dnorm = y_grad.array().rowwise() * weight_.array();  // [B, D]
    Eigen::RowVectorXf dnorm_mean = dnorm.rowwise().mean();         // [B,]
    Eigen::RowVectorXf dnorm_norm_mean =
        (dnorm.array() * norm.array()).rowwise().mean();  // [B,]
    x_grad =
        ((dnorm.array().colwise() - dnorm_mean.transpose().array()).array() -
         (norm.array().colwise() * dnorm_norm_mean.transpose().array()))
            .array()
            .colwise() *
        rstd.transpose().array();
    //    std::cout << "x_grad: " << x_grad << std::endl;

    // w_grad = dL/dy * dy/dw
    //        = dL/dy * x_norm(B,D)
    //        = \sum_i^B [y_grad(B, D) \elewise_dot x_norm(B, D)]
    weight_grad_.array() +=
        (y_grad.array() * ((x.colwise() - mean.transpose()).array().colwise() *
                           rstd.transpose().array())
                              .array())
            .colwise()
            .sum()
            .array();

    // b_grad = dL/dy * dy/db
    //        = \sum_i^(B)(y_grad(B, D))
    //        = [D,]
    bias_grad_.noalias() += y_grad.colwise().sum();
  }

  void LazilyAllocateGradMemory() {
    if (weight_grad_.size() == 0) {
      weight_grad_ = Eigen::RowVectorXf::Zero(normalized_shape_);
    }
    if (bias_grad_.size() == 0) {
      bias_grad_ = Eigen::RowVectorXf::Zero(normalized_shape_);
    }
  }

  size_t NumParameters() const { return normalized_shape_ * 2; }

  int normalized_shape_;
  float eps_;
  Eigen::RowVectorXf weight_, weight_grad_;
  Eigen::RowVectorXf bias_, bias_grad_;
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
  Softmax(bool stable_softmax = false) : stable_softmax_(stable_softmax) {}

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
                      bool softmax_subtract_max_value = false)
      : reduction_(reduction) {
    softmax_ = std::make_unique<Softmax>(softmax_subtract_max_value);
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
