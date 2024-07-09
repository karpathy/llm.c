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
#include "unsupported/Eigen/CXX11/ThreadPool"

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

std::pair<int, int> SplitRange(int total, int idx, int n) {
  int q = total / n;
  int r = total % n;
  if (idx < r) {
    return {(q + 1) * idx, (q + 1) * (idx + 1)};
  } else {
    return {q * idx + r, q * (idx + 1) + r};
  }
}

using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixInt =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using Tensor1D = Eigen::Tensor<float, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

template <typename T>
struct Span2D {
  size_t size() const { return flat_.size(); }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }

  Eigen::TensorMap<Tensor1D> View1D() const {
    return Eigen::TensorMap<Tensor1D>(flat_.data(), flat_.size());
  }

  Eigen::TensorMap<Tensor2D> View2D() const {
    return Eigen::TensorMap<Tensor2D>(flat_.data(), rows_, cols_);
  }

  Eigen::TensorMap<Tensor3D> View3D(size_t dim0, size_t dim1,
                                    size_t dim2) const {
    CHECK_EQ(dim0 * dim1 * dim2, flat_.size());
    return Eigen::TensorMap<Tensor3D>(flat_.data(), dim0, dim1, dim2);
  }

  Eigen::TensorMap<Tensor4D> View4D(size_t dim0, size_t dim1, size_t dim2,
                                    size_t dim3) const {
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, flat_.size());
    return Eigen::TensorMap<Tensor4D>(flat_.data(), dim0, dim1, dim2, dim3);
  }

 private:
  Span2D(T* array, size_t rows, size_t cols)
      : flat_(array, rows * cols), rows_(rows), cols_(cols) {}

  absl::Span<T> flat_;
  size_t rows_;
  size_t cols_;
};

Eigen::ThreadPool g_thread_pool(16 /* number of threads in pool */);
Eigen::ThreadPoolDevice g_cpu_device(&g_thread_pool,
                                     12 /* number of threads to use */);

// Parameter weight and its corresponding gradient
struct Parameter {
  enum DataType { kValue, kGrad };

  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  Parameter(int64_t length) : length_(length) {
    value_ = static_cast<float*>(g_cpu_device.allocate(sizeof(float) * length));
    g_cpu_device.memset(value_, 0, sizeof(float) * length);
    grad_ = nullptr;
  }

  ~Parameter() {
    g_cpu_device.deallocate(value_);
    g_cpu_device.deallocate(grad_);
  }

  int64_t size() const { return length_; }
  float* data() const { return value_; }
  float* grad() const { return grad_; }

  void AllocateGradient() {
    if (grad_ == nullptr) {
      grad_ =
          static_cast<float*>(g_cpu_device.allocate(sizeof(float) * length_));
      g_cpu_device.memset(grad_, 0, sizeof(float) * length_);
    }
  }

  void ZeroGrad() {
    if (grad_ != nullptr) {
      g_cpu_device.memset(grad_, 0, sizeof(float) * length_);
    }
  }

  absl::Span<float> View(DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    return {type == kValue ? value_ : grad_, static_cast<size_t>(length_)};
  }

  Eigen::Map<Eigen::RowVectorXf> View(int length,
                                      DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    CHECK_EQ(length, length_);
    return {type == kValue ? value_ : grad_,
            static_cast<Eigen::Index>(length_)};
  }

  Eigen::Map<nn::Matrix> View(int rows, int cols,
                              DataType type = DataType::kValue) const {
    LOG_IF(FATAL, type == kGrad && grad_ == nullptr)
        << "Gradient memory has not been allocated!";
    CHECK_EQ(rows * cols, length_);
    return {type == kValue ? value_ : grad_, rows, cols};
  }

 private:
  float* value_;
  float* grad_;
  int64_t length_;
};

struct MatMul {
  static void Forward(const Eigen::TensorMap<Tensor2D>& x1,
                      const Eigen::TensorMap<Tensor2D>& x2,
                      Eigen::TensorMap<Tensor2D>& y) {
    // x: [M, N], x2: [N, K], y: [M, K]
    CHECK_EQ(x1.dimension(0), y.dimension(0));
    CHECK_EQ(x1.dimension(1), x2.dimension(0));
    CHECK_EQ(x2.dimension(1), y.dimension(1));

    // y = x1 * x2
    //    y.noalias() = x1 * x2;
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    y.device(g_cpu_device) = x1.contract(x2, product_dims);
  }

  static void Backward(const Eigen::TensorMap<Tensor2D>& x1,
                       const Eigen::TensorMap<Tensor2D>& x2,
                       const Eigen::TensorMap<Tensor2D>& y_grad,
                       Eigen::TensorMap<Tensor2D>& x1_grad,
                       Eigen::TensorMap<Tensor2D>& x2_grad) {
    // input:
    // x1: [M, N], x2:[N, K]
    // y_grad: [M, K]
    //
    // output:
    // x1_grad: [M, N], x2_grad: [N, K]
    int M = x1.dimension(0), N = x1.dimension(1), K = x2.dimension(1);
    CHECK(M == y_grad.dimension(0) && M == x1_grad.dimension(0));
    CHECK(N == x2.dimension(0) && N == x1_grad.dimension(1) &&
          N == x2_grad.dimension(0));
    CHECK(K == y_grad.dimension(1) && K == x2_grad.dimension(1));

    // x1_grad = dL/dy * dy/dx1
    //        = y_grad(M, K) * x2^T (K, N)
    //        = [M, N]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    x1_grad.device(g_cpu_device) += y_grad.contract(x2, product_dims);

    // x2_grad = dL/dy * dy/dx2
    //        = x1^T(N, M) * y_grad(M, K)
    //        = [N, K]

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    x2_grad.device(g_cpu_device) += x1.contract(y_grad, product_dims2);
  }
};

struct Residual {
  static void Forward(const Eigen::TensorMap<Tensor1D>& x,
                      const Eigen::TensorMap<Tensor1D>& Fx,
                      Eigen::TensorMap<Tensor1D>& Hx) {
    int N = x.size();
    CHECK(N == Fx.size() && N == Hx.size());

    // H(x) = x + F(x) -> F(x) = H(x) - x
    //    for (int i = 0; i < N; ++i) {
    //      Hx[i] = x[i] + Fx[i];
    //    }
    Hx.device(g_cpu_device) = x + Fx;
  }

  static void Backward(const Eigen::TensorMap<Tensor1D>& Hx_grad,
                       Eigen::TensorMap<Tensor1D> x_grad,
                       Eigen::TensorMap<Tensor1D> Fx_grad) {
    int N = Hx_grad.size();
    CHECK(N == x_grad.size() && N == Fx_grad.size());

    //    for (int i = 0; i < N; ++i) {
    //      x_grad[i] += Hx_grad[i];
    //      Fx_grad[i] += Hx_grad[i];
    //    }
    x_grad.device(g_cpu_device) += Hx_grad;
    Fx_grad.device(g_cpu_device) += Hx_grad;
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

  void Forward(const Eigen::TensorMap<Tensor2D>& x,
               Eigen::TensorMap<Tensor2D>& y) const {
    // x: [B, in_features], y: [B, out_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y.dimension(0));

    auto weight = Eigen::TensorMap<Tensor2D>(weight_->data(), out_features_,
                                             in_features_);
    // y = x * w^T + b
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    if (has_bias_) {
      auto bias = Eigen::TensorMap<Tensor1D>(bias_->data(), out_features_);
      Eigen::array<int, 2> broadcast_dims = {static_cast<int>(y.dimension(0)),
                                             1};
      y.device(g_cpu_device) =
          x.contract(weight, product_dims) + bias.broadcast(broadcast_dims);
    } else {
      y.device(g_cpu_device) = x.contract(weight, product_dims);
    }
  }

  void Backward(const Eigen::TensorMap<Tensor2D>& x,
                const Eigen::TensorMap<Tensor2D>& y_grad,
                Eigen::TensorMap<Tensor2D>& x_grad) {
    // x: [B, in_features], y_grad: [B, out_features], x_grad: [B, in_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y_grad.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));

    // Lazily allocate the memory for gradients
    weight_->AllocateGradient();
    //    auto weight = weight_->View(out_features_, in_features_);
    auto weight = Eigen::TensorMap<nn::Tensor2D>(weight_->data(), out_features_,
                                                 in_features_);
    //    auto weight_grad =
    //        weight_->View(out_features_, in_features_, Parameter::kGrad);
    auto weight_grad = Eigen::TensorMap<nn::Tensor2D>(
        weight_->grad(), out_features_, in_features_);

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    x_grad.device(g_cpu_device) += y_grad.contract(weight, product_dims);

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    weight_grad.device(g_cpu_device) += y_grad.contract(x, product_dims2);

    if (has_bias_) {
      // b_grad = dL/dy * dy/db
      //        = \sum_i^(B)(y_grad(B, out_features))
      //        = [out_features,]
      bias_->AllocateGradient();

      //      auto bias_grad = bias_->View(out_features_, Parameter::kGrad);
      //      auto y_grad_matrix = Eigen::Map<nn::Matrix>(
      //          y_grad.data(), y_grad.dimension(0), y_grad.dimension(1));
      //      bias_grad.noalias() += y_grad_matrix.colwise().sum();
      auto bias_grad = Eigen::TensorMap<Tensor1D>(bias_->grad(), out_features_);
      Eigen::array<Eigen::Index, 1> along_batch = {0};
      bias_grad.device(g_cpu_device) = y_grad.sum(along_batch);
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
    auto b = bias_->View();
    absl::c_fill(b, 0.0f);
  }

  void Forward(const Eigen::Map<Matrix>& x, Eigen::Map<Matrix>& y,
               Eigen::Map<Eigen::RowVectorXf>& mean,
               Eigen::Map<Eigen::RowVectorXf>& rstd) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.cols(), normalized_shape_);
    CHECK_EQ(y.cols(), normalized_shape_);
    CHECK_EQ(x.rows(), y.rows());
    int B = x.rows();

    auto x2d = Eigen::TensorMap<Tensor2D>(const_cast<float*>(x.data()),
                                          x.rows(), x.cols());
    auto y2d = Eigen::TensorMap<Tensor2D>(y.data(), y.rows(), y.cols());
    auto mean_1d = Eigen::TensorMap<Tensor1D>(mean.data(), mean.size());
    auto rstd_1d = Eigen::TensorMap<Tensor1D>(rstd.data(), rstd.size());

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    /*
    mean.noalias() = x.rowwise().mean();
    */

    Eigen::array<Eigen::Index, 1> along_class = {1};
    mean_1d.device(g_cpu_device) = x2d.mean(along_class);

    // x_zero_centered(B, D) = x.colwise() - m.transpose()
    // x_zero_centered_square(B, D) = x_zero_centered.array().square()
    // var(B,) = x_zero_centered_square.rowwise().mean()
    // std(B,) = (var + eps).sqrt()
    // rstd(B,) = 1.f / std;

    /*
    rstd = 1.f /
           ((x.colwise() - mean.transpose()).array().square().rowwise().mean() +
            eps_)
               .sqrt();
    */

    int batch_size = x2d.dimension(0), num_class = x2d.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    rstd_1d.device(g_cpu_device) =
        ((x2d - mean_1d.reshape(batch_by_one).broadcast(one_by_class))
             .square()
             .mean(along_class) +
         eps_)
            .sqrt()
            .inverse();

    auto weight = weight_->View(normalized_shape_);
    auto bias = bias_->View(normalized_shape_);
    // normalize: (x - mean) / std
    // && scale:  (x - mean) / std * weight
    // && shift:  (x - mean) / std * weight + bias

    /*
    y = (((x.colwise() - mean.transpose()).array().colwise() *
          rstd.transpose().array())
             .array()
             .rowwise() *
         weight.array())
            .array()
            .rowwise() +
        bias.array();
    */

    auto weight_1d =
        Eigen::TensorMap<Tensor1D>(weight_->data(), normalized_shape_);
    auto bias_1d = Eigen::TensorMap<Tensor1D>(bias_->data(), normalized_shape_);
    y2d.device(g_cpu_device) =
        (x2d - mean_1d.reshape(batch_by_one).broadcast(one_by_class)) *
            rstd_1d.reshape(batch_by_one).broadcast(one_by_class) *
            weight_1d.reshape(one_by_class).broadcast(batch_by_one) +
        bias_1d.reshape(one_by_class).broadcast(batch_by_one);
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

    auto x2d = Eigen::TensorMap<Tensor2D>(const_cast<float*>(x.data()),
                                          x.rows(), x.cols());
    auto y_grad_2d = Eigen::TensorMap<Tensor2D>(
        const_cast<float*>(y_grad.data()), y_grad.rows(), y_grad.cols());
    auto mean_1d = Eigen::TensorMap<Tensor1D>(const_cast<float*>(mean.data()),
                                              mean.size());
    auto rstd_1d = Eigen::TensorMap<Tensor1D>(const_cast<float*>(rstd.data()),
                                              rstd.size());
    int batch_size = x2d.dimension(0), num_class = x2d.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // Lazily allocate the memory for gradients
    weight_->AllocateGradient();
    bias_->AllocateGradient();
    auto weight = weight_->View(normalized_shape_);
    auto weight_grad = weight_->View(normalized_shape_, Parameter::kGrad);
    auto bias = bias_->View(normalized_shape_);
    auto bias_grad = bias_->View(normalized_shape_, Parameter::kGrad);
    auto weight_1d =
        Eigen::TensorMap<Tensor1D>(weight_->data(), normalized_shape_);
    auto weight_grad_1d =
        Eigen::TensorMap<Tensor1D>(weight_->grad(), normalized_shape_);
    auto bias_1d = Eigen::TensorMap<Tensor1D>(bias_->data(), normalized_shape_);
    auto bias_grad_1d =
        Eigen::TensorMap<Tensor1D>(bias_->grad(), normalized_shape_);

    // x_grad = dL/dy * dy/dnorm
    //                * [dnorm/dxmean * dxmean/dx
    //                  + dnorm/dmean * dmean/dx
    //                  + dnorm/dstd * dstd/dx
    //                  ]

    /*
    nn::Matrix norm = (x.colwise() - mean.transpose()).array().colwise() *
                      rstd.transpose().array();                    // [B,D]
    nn::Matrix dnorm = y_grad.array().rowwise() * weight.array();  // [B,D]
    Eigen::RowVectorXf dnorm_mean = dnorm.rowwise().mean();        //[B,]
    Eigen::RowVectorXf dnorm_norm_mean =
        (dnorm.array() * norm.array()).rowwise().mean();  // [B,]
    x_grad.array() +=
        ((dnorm.array().colwise() - dnorm_mean.transpose().array()).array() -
         (norm.array().colwise() * dnorm_norm_mean.transpose().array()))
            .array()
            .colwise() *
        rstd.transpose().array();
    */

    Tensor2D norm_2d =
        (x2d - mean_1d.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd_1d.reshape(batch_by_one).broadcast(one_by_class);  // [B, D]
    Tensor2D dnorm_2d =
        y_grad_2d *
        weight_1d.reshape(one_by_class).broadcast(batch_by_one);  // [B, D]
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Tensor1D dnorm_mean_1d = dnorm_2d.mean(along_class);  // [B,]
    Tensor1D dnorm_norm_mean_1d =
        (dnorm_2d * norm_2d).mean(along_class);  // [B,]
    auto x_grad_2d =
        Eigen::TensorMap<Tensor2D>(x_grad.data(), x_grad.rows(), x_grad.cols());
    x_grad_2d.device(g_cpu_device) +=
        ((dnorm_2d -
          dnorm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) -
         norm_2d *
             dnorm_norm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd_1d.reshape(batch_by_one).broadcast(one_by_class);

    // w_grad = dL/dy * dy/dw
    //        = dL/dy * x_norm(B,D)
    //        = \sum_i^B [y_grad(B, D) \elewise_dot x_norm(B, D)]
    /*
    weight_grad.array() +=
        (y_grad.array() * ((x.colwise() - mean.transpose()).array().colwise() *
                           rstd.transpose().array())
                              .array())
            .colwise()
            .sum()
            .array();
    */

    Eigen::array<Eigen::Index, 1> along_batch = {0};
    weight_grad_1d.device(g_cpu_device) +=
        (y_grad_2d * norm_2d).sum(along_batch);

    // b_grad = dL/dy * dy/db
    //        = \sum_i^(B)(y_grad(B, D))
    //        = [D,]

    //    bias_grad.noalias() += y_grad.colwise().sum();

    bias_grad_1d.device(g_cpu_device) += y_grad_2d.sum(along_batch);
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
    //    for (size_t i = 0; i < x.size(); ++i) {
    //      float _x = x[i];
    //      float cube = 0.044715f * _x * _x * _x;
    //      y[i] = 0.5f * _x * (1.0f + std::tanh(sqrt_2_over_pi * (_x + cube)));
    //    }

    float coeff = 0.044715f;
    auto input = Eigen::TensorMap<Tensor1D>(
        const_cast<Tensor1D::Scalar*>(x.data()), x.size());
    auto output = Eigen::TensorMap<Tensor1D>(y.data(), y.size());
    output.device(g_cpu_device) =
        0.5 * input *
        (1.0 +
         ((sqrt_2_over_pi * (input + coeff * input * input * input)).tanh()));
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

    //    for (size_t i = 0; i < x.size(); ++i) {
    //      float _x = x[i];
    //      float cube = 0.044715f * _x * _x * _x;
    //      float tanh_arg = sqrt_2_over_pi * (_x + cube);
    //      float tanh_out = std::tanh(tanh_arg);
    //      float dydx = 0.5f * (1.0f + tanh_out) +
    //                   0.5f * _x * (1.0f - tanh_out * tanh_out) *
    //                       (sqrt_2_over_pi * (1.0f + 3.f * 0.044715f * _x *
    //                       _x));
    //      x_grad[i] += y_grad[i] * dydx;
    //    }

    const float sqrt_2_over_pi = std::sqrt(M_2_PI);
    float coeff = 0.044715f;
    auto input = Eigen::TensorMap<Tensor1D>(
        const_cast<Tensor1D::Scalar*>(x.data()), x.size());
    auto output_grad = Eigen::TensorMap<Tensor1D>(
        const_cast<Tensor1D::Scalar*>(y_grad.data()), y_grad.size());
    auto input_grad = Eigen::TensorMap<Tensor1D>(x_grad.data(), x_grad.size());
    auto cube = coeff * input * input * input;
    auto tanh_arg = sqrt_2_over_pi * (input + cube);
    auto tanh_out = tanh_arg.tanh();
    auto dydx = 0.5f * (1.0f + tanh_out) +
                0.5f * input * (1.0f - tanh_out * tanh_out) *
                    (sqrt_2_over_pi * (1.0f + 3.0f * coeff * input * input));
    input_grad.device(g_cpu_device) += output_grad * dydx;
  }
};

struct Softmax {
  Softmax() {}

  void Forward(const Eigen::TensorMap<Tensor2D>& x,
               Eigen::TensorMap<Tensor2D>& y) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    //      auto x_exp = (x.colwise() - x.rowwise().maxCoeff()).array().exp();
    //      y = x_exp.array().colwise() / x_exp.rowwise().sum().array();

    y.device(g_cpu_device) = (x - x.maximum(along_class)
                                      .eval()
                                      .reshape(batch_by_one)
                                      .broadcast(one_by_class))
                                 .exp();
    y.device(g_cpu_device) = y * y.sum(along_class)
                                     .inverse()
                                     .eval()
                                     .reshape(batch_by_one)
                                     .broadcast(one_by_class);

    /*
    int B = x.dimension(0), V = x.dimension(1);
    int thread_num = g_cpu_device.numThreads();
    auto fn = [&x, &y, V](int begin, int end) {
      for (int b = begin; b < end; b++) {
        // probs <- softmax(logits)
        const float* logits_bt = x.data() + b * V;
        float* probs_bt = y.data() + b * V;

        // maxval is only calculated and subtracted for numerical
        // stability
        float maxval = -10000.0f;  // TODO something better
        for (int i = 0; i < V; i++) {
          if (logits_bt[i] > maxval) {
            maxval = logits_bt[i];
          }
        }
        float sum = 0.0f;
        for (int i = 0; i < V; i++) {
          probs_bt[i] = expf(logits_bt[i] - maxval);
          sum += probs_bt[i];
        }
        // note we only loop to V, leaving the padded dimensions
        for (int i = 0; i < V; i++) {
          probs_bt[i] /= sum;
        }
      }
    };

    Eigen::Barrier barrier(thread_num);
    for (int t = 0; t < thread_num; ++t) {
      auto range = SplitRange(B, t, thread_num);
      g_cpu_device.enqueue_with_barrier(&barrier, fn, range.first,
                                        range.second);
    }
    barrier.Wait();
    */
  }

  void Backward(const Eigen::TensorMap<Tensor2D>& y,
                const Eigen::TensorMap<Tensor2D>& y_grad,
                Eigen::TensorMap<Tensor2D>& x_grad) {
    // y:[B, D], y_grad: [B, D], x_grad: [B, D]
    int B = y.dimension(0), D = y.dimension(1);
    CHECK(B == y_grad.dimension(0) && B == x_grad.dimension(0));
    CHECK(D == y_grad.dimension(1) && D == x_grad.dimension(1));

    // Using alternative formula:
    // dL/dx = dL/dy * y - sum(dL/dy * y) * y
    //    = (dL/dy - sum(dL/dy * y)) * y
    int batch_size = y.dimension(0), num_class = y.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    Eigen::array<Eigen::Index, 1> along_class = {1};
    auto dyy = y_grad * y;
    auto sum = dyy.sum(along_class).reshape(batch_by_one);
    auto sub = y_grad - sum.broadcast(one_by_class);
    x_grad.device(g_cpu_device) += sub * y;

    /*
    // dy_j / dx_i = S_i(1 - S_j) for i==j
    //             = -S_j*S_i     for i!=j
    // dL/dx_i = \sum_j dL/dy_j * dy_j / dx_i
    auto fn = [D, &x_grad, &y_grad, &y](int begin, int end) {
      for (int b = begin; b < end; ++b) {
        float* x_grad_b = x_grad.data() + b * D;
        float* y_grad_b = y_grad.data() + b * D;
        float* y_b = y.data() + b * D;
        for (int i = 0; i < D; ++i) {
          for (int j = 0; j < D; ++j) {
            float indicator = i == j ? 1.0f : 0.0f;
            //            x_grad(b, i) += y_grad(b, j) * y(b, i) * (indicator -
            //            y(b, j));
            x_grad_b[i] += y_grad_b[j] * y_b[i] * (indicator - y_b[j]);
          }
        }
      }
    };

    int thread_num = g_cpu_device.numThreads();
    Eigen::Barrier barrier(thread_num);
    for (int t = 0; t < thread_num; ++t) {
      auto range = SplitRange(B, t, thread_num);
      g_cpu_device.enqueue_with_barrier(&barrier, fn, range.first,
                                        range.second);
    }
    barrier.Wait();
    */
  }
};

struct SoftmaxCrossEntropy {
  enum Reduction { MEAN, SUM };

  SoftmaxCrossEntropy(Reduction reduction = Reduction::MEAN)
      : reduction_(reduction) {
    softmax_ = std::make_unique<Softmax>();
  }

  void Forward(const Eigen::TensorMap<Tensor2D>& logits,
               absl::Span<const int> targets, Eigen::TensorMap<Tensor2D>& probs,
               float* loss) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == targets.size() && B == probs.dimension(0));
    CHECK_EQ(C, probs.dimension(1));

    // apply softmax to convert logits to (normalized) probabilities
    softmax_->Forward(logits, probs);

    // targets: [B,]
    *loss = 0.0f;
    for (int i = 0; i < targets.size(); ++i) {
      int ix = targets[i];
      *loss += -std::log(probs(i, ix));
    }

    if (reduction_ == Reduction::MEAN) {
      *loss /= static_cast<float>(B);
    }
  }

  void Backward(const Eigen::TensorMap<Tensor2D>& probs,
                absl::Span<const int> targets,
                Eigen::TensorMap<Tensor2D>& logits_grad) {
    // probs: [B, C], targets: [B,]
    // logits_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == targets.size() && B == logits_grad.dimension(0));
    CHECK_EQ(C, logits_grad.dimension(1));

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

  void Forward(const Eigen::TensorMap<Tensor2D>& probs,
               absl::Span<const int> targets, float* loss) {
    // probs:[B, C], targets: [B,] loss: scalar
    int B = probs.dimension(0), C = probs.dimension(1);
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

  void Backward(const Eigen::TensorMap<Tensor2D>& probs,
                absl::Span<const int> targets,
                Eigen::TensorMap<Tensor2D>& probs_grad) {
    // probs: [B, C], targets: [B,]
    // probs_grad: [B, C]
    int B = probs.dimension(0), C = probs.dimension(1);
    CHECK(B == targets.size() && B == probs_grad.dimension(0));
    CHECK_EQ(C, probs_grad.dimension(1));

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
