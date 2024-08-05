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
#include "tensor_util.hpp"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace nn {

#ifdef EIGEN_USE_GPU
Eigen::GpuStreamDevice g_stream;
Eigen::GpuDevice g_device(&g_stream);
#else
Eigen::ThreadPool g_thread_pool(16 /* number of threads in pool */);
Eigen::ThreadPoolDevice g_device(&g_thread_pool,
                                 12 /* number of threads to use */);
#endif

mt19937_state g_mt19937_state;

void ManualSeed(unsigned int seed) { manual_seed(&g_mt19937_state, seed); }

void ConstantFill(absl::Span<float> weight, float C) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size(), C);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  absl::c_fill(weight, C);
#endif
}

void UniformFill(absl::Span<float> weight, float from = 0.0, float to = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  uniform_(w.data(), w.size(), from, to, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  uniform_(weight.data(), weight.size(), from, to, &g_mt19937_state);
#endif
}

void NormalFill(absl::Span<float> weight, float mean = 0.0, float std = 1.0) {
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  normal_(w.data(), w.size(), mean, std, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  normal_(weight.data(), weight.size(), mean, std, &g_mt19937_state);
#endif
}

void KaimingUniformFill(absl::Span<float> weight, int in_features) {
  const float bound = std::sqrt(1.0f / in_features);
#ifdef EIGEN_USE_GPU
  std::vector<float> w(weight.size());
  uniform_(w.data(), w.size(), -bound, bound, &g_mt19937_state);
  g_device.memcpyHostToDevice(weight.data(), w.data(),
                              sizeof(float) * w.size());
#else
  uniform_(weight.data(), weight.size(), -bound, bound, &g_mt19937_state);
#endif
}

void UpperTriangularWithNegativeInf(typename TTypes<float>::Matrix matrix) {
  using MatrixXf =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  MatrixXf m = MatrixXf::Zero(matrix.dimension(0), matrix.dimension(1));
  m.triangularView<Eigen::StrictlyUpper>().setConstant(
      -std::numeric_limits<float>::infinity());
#ifdef EIGEN_USE_GPU
  g_device.memcpyHostToDevice(matrix.data(), m.data(),
                              sizeof(float) * matrix.size());
#else
  g_device.memcpy(matrix.data(), m.data(), sizeof(float) * matrix.size());
#endif
}

void OntHot(typename TTypes<int>::ConstFlat target,
            typename TTypes<float>::Matrix label) {
  int batch_size = target.size(), num_class = label.dimension(1);
  CHECK_EQ(batch_size, label.dimension(0));
  for (int i = 0; i < batch_size; ++i) {
    int ix = target(i);
    CHECK_LT(ix, num_class);
    label(i, ix) = 1.0f;
  }
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

enum DataType : int { DT_FLOAT = 1, DT_HALF = 2, DT_INT32 = 3 };

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidDataType;

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)     \
  template <>                               \
  struct DataTypeToEnum<TYPE> {             \
    static DataType v() { return ENUM; }    \
    static constexpr DataType value = ENUM; \
  };                                        \
  template <>                               \
  struct IsValidDataType<TYPE> {            \
    static constexpr bool value = true;     \
  };                                        \
  template <>                               \
  struct EnumToDataType<ENUM> {             \
    typedef TYPE Type;                      \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
MATCH_TYPE_AND_ENUM(int, DT_INT32);

// Parameter weight and its corresponding gradient
struct Parameter {
  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  explicit Parameter(DataType dtype, int64_t num_element = 0)
      : dtype_(dtype),
        num_element_(num_element),
        data_(nullptr),
        grad_(nullptr) {
    if (num_element) {
      LazyAllocate(num_element);
    }
  }

  ~Parameter() {
    g_device.deallocate(data_);
    g_device.deallocate(grad_);
  }

  int64_t size() const { return num_element_; }

  void LazyAllocate(int num_element) {
    if (data_ == nullptr) {
      data_ = Allocate(dtype_, num_element);
      Zero(data_, dtype_, num_element);
      num_element_ = num_element;
    }
    CHECK_EQ(num_element, num_element_);
  }

  void LazyAllocateGradient() {
    if (grad_ == nullptr) {
      CHECK_GT(num_element_, 0);
      grad_ = Allocate(dtype_, num_element_);
      Zero(grad_, dtype_, num_element_);
    }
  }

  void ZeroData() {
    if (data_ != nullptr) {
      Zero(data_, dtype_, num_element_);
    }
  }

  void ZeroGrad() {
    if (grad_ != nullptr) {
      Zero(grad_, dtype_, num_element_);
    }
  }

  template <typename T>
  T* data() const {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* grad() const {
    return static_cast<T*>(grad_);
  }

  template <typename T>
  absl::Span<T> span() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), static_cast<size_t>(num_element_)};
  }

  template <typename T>
  absl::Span<T> span_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), static_cast<size_t>(num_element_)};
  }

  template <typename T>
  typename TTypes<T>::Flat flat() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), num_element_};
  }
  template <typename T>
  typename TTypes<T>::ConstFlat const_flat() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {data<T>(), num_element_};
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {data<T>(), rows, cols};
  }
  template <typename T>
  typename TTypes<T>::ConstMatrix const_matrix(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {data<T>(), rows, cols};
  }

  template <typename T>
  typename TTypes<T, 3>::Tensor tensor_3d(int dim0, int dim1, int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {data<T>(), dim0, dim1, dim2};
  }
  template <typename T>
  typename TTypes<T, 3>::ConstTensor const_tensor_3d(int dim0, int dim1,
                                                     int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {data<T>(), dim0, dim1, dim2};
  }

  template <typename T>
  typename TTypes<T, 4>::Tensor tensor_4d(int dim0, int dim1, int dim2,
                                          int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {data<T>(), dim0, dim1, dim2, dim3};
  }
  template <typename T>
  typename TTypes<T, 4>::ConstTensor const_tensor_4d(int dim0, int dim1,
                                                     int dim2, int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {data<T>(), dim0, dim1, dim2, dim3};
  }

  template <typename T>
  typename TTypes<T>::Flat flat_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), num_element_};
  }
  template <typename T>
  typename TTypes<T>::ConstFlat const_flat_grad() const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    return {grad<T>(), num_element_};
  }

  template <typename T>
  typename TTypes<T>::Matrix matrix_grad(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {grad<T>(), rows, cols};
  }
  template <typename T>
  typename TTypes<T>::ConstMatrix const_matrix_grad(int rows, int cols) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(rows * cols, num_element_);
    return {grad<T>(), rows, cols};
  }

  template <typename T>
  typename TTypes<T, 3>::Tensor tensor_3d_grad(int dim0, int dim1,
                                               int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {grad<T>(), dim0, dim1, dim2};
  }
  template <typename T>
  typename TTypes<T, 3>::ConstTensor const_tensor_3d_grad(int dim0, int dim1,
                                                          int dim2) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2, num_element_);
    return {grad<T>(), dim0, dim1, dim2};
  }

  template <typename T>
  typename TTypes<T, 4>::Tensor tensor_4d_grad(int dim0, int dim1, int dim2,
                                               int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {grad<T>(), dim0, dim1, dim2, dim3};
  }

  template <typename T>
  typename TTypes<T, 4>::ConstTensor const_tensor_4d_grad(int dim0, int dim1,
                                                          int dim2,
                                                          int dim3) const {
    CHECK_EQ(DataTypeToEnum<T>::value, dtype_);
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, num_element_);
    return {grad<T>(), dim0, dim1, dim2, dim3};
  }

 private:
  static void* Allocate(DataType dtype, int64_t num_element) {
    if (dtype == DT_FLOAT) {
      return g_device.allocate(sizeof(float) * num_element);
    } else if (dtype == DT_HALF) {
      return g_device.allocate(sizeof(Eigen::half) * num_element);
    } else {
      throw std::invalid_argument("invalid data type: " +
                                  std::to_string(dtype));
    }
  }

  static void Zero(void* data, DataType dtype, int64_t num_element) {
    if (dtype == DT_FLOAT) {
      g_device.memset(data, 0, sizeof(float) * num_element);
    } else if (dtype == DT_HALF) {
      g_device.memset(data, 0, sizeof(Eigen::half) * num_element);
    } else {
      throw std::invalid_argument("invalid data type: " +
                                  std::to_string(dtype));
    }
  }

  DataType dtype_;
  int64_t num_element_;
  void* data_;
  void* grad_;
};

using Activation = Parameter;

template <typename T>
struct MatMul {
  static void Forward(typename TTypes<T>::ConstMatrix x1,
                      typename TTypes<T>::ConstMatrix x2,
                      typename TTypes<T>::Matrix y, T scale = 1.0f) {
    // x: [M, N], x2: [N, K], y: [M, K]
    CHECK_EQ(x1.dimension(0), y.dimension(0));
    CHECK_EQ(x1.dimension(1), x2.dimension(0));
    CHECK_EQ(x2.dimension(1), y.dimension(1));

    // y = x1 * x2
    //    y.noalias() = x1 * x2;
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    y.device(g_device) = x1.contract(x2, product_dims) * scale;
  }

  static void Backward(typename TTypes<T>::ConstMatrix x1,
                       typename TTypes<T>::ConstMatrix x2,
                       typename TTypes<T>::ConstMatrix y_grad,
                       typename TTypes<T>::Matrix x1_grad,
                       typename TTypes<T>::Matrix x2_grad, T scale = 1.0) {
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
    x1_grad.device(g_device) += y_grad.contract(x2, product_dims) * scale;

    // x2_grad = dL/dy * dy/dx2
    //        = x1^T(N, M) * y_grad(M, K)
    //        = [N, K]

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    x2_grad.device(g_device) += x1.contract(y_grad, product_dims2) * scale;
  }
};

template <typename T>
struct Residual {
  static void Forward(typename TTypes<T>::ConstFlat x,
                      typename TTypes<T>::ConstFlat Fx,
                      typename TTypes<T>::Flat Hx) {
    int N = x.size();
    CHECK(N == Fx.size() && N == Hx.size());

    // H(x) = x + F(x) -> F(x) = H(x) - x
    Hx.device(g_device) = x + Fx;
  }

  static void Backward(typename TTypes<T>::ConstFlat Hx_grad,
                       typename TTypes<T>::Flat x_grad,
                       typename TTypes<T>::Flat Fx_grad) {
    int N = Hx_grad.size();
    CHECK(N == x_grad.size() && N == Fx_grad.size());

    x_grad.device(g_device) += Hx_grad;
    Fx_grad.device(g_device) += Hx_grad;
  }
};

template <typename T>
struct Linear {
  Linear(int in_features, int out_features, bool bias = true)
      : in_features_(in_features),
        out_features_(out_features),
        has_bias_(bias) {
    auto dtype = DataTypeToEnum<T>::value;
    weight_ = std::make_unique<Parameter>(dtype, out_features * in_features);
    KaimingUniformFill(weight_->span<T>(), in_features);
    if (bias) {
      bias_ = std::make_unique<Parameter>(dtype, out_features);
      const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
      UniformFill(bias_->span<T>(), -bound, bound);
    }
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) const {
    // x: [B, in_features], y: [B, out_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y.dimension(0));

    auto weight = MakeMatrix(weight_->data<T>(), out_features_, in_features_);
    // y = x * w^T + b
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    if (has_bias_) {
      auto bias = MakeFlat(bias_->data<T>(), out_features_);
      Eigen::array<int, 2> broadcast_dims = {static_cast<int>(y.dimension(0)),
                                             1};
      y.device(g_device) =
          x.contract(weight, product_dims) + bias.broadcast(broadcast_dims);
    } else {
      y.device(g_device) = x.contract(weight, product_dims);
    }
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    // x: [B, in_features], y_grad: [B, out_features], x_grad: [B, in_features]
    CHECK_EQ(x.dimension(1), in_features_);
    CHECK_EQ(y_grad.dimension(1), out_features_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    auto weight = MakeMatrix(weight_->data<T>(), out_features_, in_features_);
    auto weight_grad =
        MakeMatrix(weight_->grad<T>(), out_features_, in_features_);

    // x_grad = dL/dy * dy/dx
    //        = y_grad(B, out_features) * W(out_features, in_features)
    //        = [B, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    x_grad.device(g_device) += y_grad.contract(weight, product_dims);

    // w_grad = dL/dy * dy/dw
    //        = y_grad^T(out_features, B) * x(B, in_features)
    //        = [out_features, in_features]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    weight_grad.device(g_device) += y_grad.contract(x, product_dims2);

    if (has_bias_) {
      // b_grad = dL/dy * dy/db
      //        = \sum_i^(B)(y_grad(B, out_features))
      //        = [out_features,]
      bias_->LazyAllocateGradient();
      auto bias_grad = MakeFlat(bias_->grad<T>(), out_features_);
      Eigen::array<Eigen::Index, 1> along_batch = {0};
      bias_grad.device(g_device) = y_grad.sum(along_batch);
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
    weight_ =
        std::make_unique<Parameter>(DT_FLOAT, num_embeddings * embedding_dim);
    NormalFill(weight_->span<float>());
  }

  void Forward(absl::Span<const int> idx, absl::Span<float> embedding) const {
    CHECK_EQ(embedding.size(), idx.size() * embedding_dim_);
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      void* dst = embedding.data() + i * embedding_dim_;
      void* src = weight_->data<float>() + idx[i] * embedding_dim_;
      g_device.memcpy(dst, src, sizeof(float) * embedding_dim_);
    }
  }

  void Backward(absl::Span<const int> idx,
                absl::Span<const float> grad_embedding) {
    CHECK_EQ(grad_embedding.size(), idx.size() * embedding_dim_);

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK_LT(idx[i], num_embeddings_);
      const float* g = grad_embedding.data() + i * embedding_dim_;
      float* grad = weight_->grad<float>() + idx[i] * embedding_dim_;
      auto g_1d = TTypes<float>::UnalignedConstFlat(g, embedding_dim_);
      auto grad_1d = TTypes<float>::UnalignedFlat(grad, embedding_dim_);
      grad_1d.device(g_device) += g_1d;
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

template <typename T>
struct LayerNorm {
  LayerNorm(int normalized_shape)
      : normalized_shape_(normalized_shape), eps_(1e-5) {
    auto dtype = DataTypeToEnum<T>::value;
    weight_ = std::make_unique<Parameter>(dtype, normalized_shape);
    auto w = weight_->span<T>();
    ConstantFill(w, 1.0f);
    bias_ = std::make_unique<Parameter>(dtype, normalized_shape);
    auto b = bias_->span<T>();
    ConstantFill(b, 0.0f);

    // activation gradient tensor
    norm_ = std::make_unique<Parameter>(dtype);             // [B, D]
    dnorm_ = std::make_unique<Parameter>(dtype);            // [B, D]
    dnorm_mean_ = std::make_unique<Parameter>(dtype);       // [B,]
    dnorm_norm_mean_ = std::make_unique<Parameter>(dtype);  // [B,]
  }

  void Forward(typename TTypes<T>::ConstMatrix x, typename TTypes<T>::Matrix y,
               typename TTypes<T>::Flat mean, typename TTypes<T>::Flat rstd) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(1), normalized_shape_);
    CHECK_EQ(y.dimension(1), normalized_shape_);
    CHECK_EQ(x.dimension(0), y.dimension(0));
    int B = x.dimension(0);

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    Eigen::array<Eigen::Index, 1> along_class = {1};
    mean.device(g_device) = x.mean(along_class);

    // x_zero_centered(B, D) = x.colwise() - m.transpose()
    // x_zero_centered_square(B, D) = x_zero_centered.array().square()
    // var(B,) = x_zero_centered_square.rowwise().mean()
    // std(B,) = (var + eps).sqrt()
    // rstd(B,) = 1.f / std;

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};
    rstd.device(g_device) =
        ((x - mean.reshape(batch_by_one).broadcast(one_by_class))
             .square()
             .mean(along_class) +
         eps_)
            .sqrt()
            .inverse();

    // normalize: (x - mean) / std
    // && scale:  (x - mean) / std * weight
    // && shift:  (x - mean) / std * weight + bias

    auto weight_1d = MakeFlat(weight_->data<T>(), normalized_shape_);
    auto bias_1d = MakeFlat(bias_->data<T>(), normalized_shape_);
    y.device(g_device) =
        (x - mean.reshape(batch_by_one).broadcast(one_by_class)) *
            rstd.reshape(batch_by_one).broadcast(one_by_class) *
            weight_1d.reshape(one_by_class).broadcast(batch_by_one) +
        bias_1d.reshape(one_by_class).broadcast(batch_by_one);
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::ConstFlat mean,
                typename TTypes<T>::ConstFlat rstd,
                typename TTypes<T>::Matrix x_grad) {
    // x: [B, D], y_grad: [B, D], x_grad: [B, D]
    CHECK_EQ(x.dimension(1), normalized_shape_);
    CHECK_EQ(y_grad.dimension(1), normalized_shape_);
    CHECK_EQ(x_grad.dimension(1), normalized_shape_);
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));
    int B = x.dimension(0), D = x.dimension(1);

    // mean: [B,], rstd: [B,]
    CHECK_EQ(mean.size(), B);
    CHECK_EQ(rstd.size(), B);

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // Lazily allocate the memory for gradients
    weight_->LazyAllocateGradient();
    bias_->LazyAllocateGradient();
    auto weight_1d = MakeFlat(weight_->data<T>(), normalized_shape_);
    auto weight_grad_1d = MakeFlat(weight_->grad<T>(), normalized_shape_);
    auto bias_1d = MakeFlat(bias_->data<T>(), normalized_shape_);
    auto bias_grad_1d = MakeFlat(bias_->grad<T>(), normalized_shape_);

    // x_grad = dL/dy * dy/dnorm
    //                * [dnorm/dxmean * dxmean/dx
    //                  + dnorm/dmean * dmean/dx
    //                  + dnorm/dstd * dstd/dx
    //                  ]

    // Eigen::Tensor<float, 2, Eigen::RowMajor>
    norm_->LazyAllocate(B * D);
    dnorm_->LazyAllocate(B * D);
    dnorm_mean_->LazyAllocate(B);
    dnorm_norm_mean_->LazyAllocate(B);
    auto norm_2d = norm_->matrix<T>(B, D);
    auto dnorm_2d = dnorm_->matrix<T>(B, D);
    auto dnorm_mean_1d = dnorm_mean_->flat<T>();
    auto dnorm_norm_mean_1d = dnorm_norm_mean_->flat<T>();
    norm_2d.device(g_device) =
        (x - mean.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd.reshape(batch_by_one).broadcast(one_by_class);  // [B, D]
    dnorm_2d.device(g_device) =
        y_grad *
        weight_1d.reshape(one_by_class).broadcast(batch_by_one);  // [B, D]
    Eigen::array<Eigen::Index, 1> along_class = {1};
    dnorm_mean_1d.device(g_device) = dnorm_2d.mean(along_class);  // [B,]
    dnorm_norm_mean_1d.device(g_device) =
        (dnorm_2d * norm_2d).mean(along_class);  // [B,]
    x_grad.device(g_device) +=
        ((dnorm_2d -
          dnorm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) -
         norm_2d *
             dnorm_norm_mean_1d.reshape(batch_by_one).broadcast(one_by_class)) *
        rstd.reshape(batch_by_one).broadcast(one_by_class);

    // w_grad = dL/dy * dy/dw
    //        = dL/dy * x_norm(B,D)
    //        = \sum_i^B [y_grad(B, D) \elewise_dot x_norm(B, D)]

    Eigen::array<Eigen::Index, 1> along_batch = {0};
    weight_grad_1d.device(g_device) += (y_grad * norm_2d).sum(along_batch);

    // b_grad = dL/dy * dy/db
    //        = \sum_i^(B)(y_grad(B, D))
    //        = [D,]

    bias_grad_1d.device(g_device) += y_grad.sum(along_batch);
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

  // activation gradient tensor
  std::unique_ptr<Parameter> norm_;             // [B, D]
  std::unique_ptr<Parameter> dnorm_;            // [B, D]
  std::unique_ptr<Parameter> dnorm_mean_;       // [B,]
  std::unique_ptr<Parameter> dnorm_norm_mean_;  // [B,]
};

// Careful there are a few versions of GeLU, this one is the exact one used by
// OpenAI
template <typename T>
struct NewGELU {
  void Forward(typename TTypes<T>::ConstFlat x, typename TTypes<T>::Flat y) {
    CHECK_EQ(x.size(), y.size());
    const float sqrt_2_over_pi = std::sqrt(M_2_PI);

    // y = 0.5 * x * (1.0 + tanh[sqrt(2/pi) * (x + 0.044715 * x^3)])
    float coeff = 0.044715f;
    y.device(g_device) =
        0.5 * x * (1.0 + ((sqrt_2_over_pi * (x + coeff * x * x * x)).tanh()));
  }

  void Backward(typename TTypes<T>::ConstFlat x,
                typename TTypes<T>::ConstFlat y_grad,
                typename TTypes<T>::Flat x_grad) {
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
    float coeff = 0.044715f;
    auto cube = coeff * x * x * x;
    auto tanh_arg = sqrt_2_over_pi * (x + cube);
    auto tanh_out = tanh_arg.tanh();
    auto dydx = 0.5f * (1.0f + tanh_out) +
                0.5f * x * (1.0f - tanh_out * tanh_out) *
                    (sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x));
    x_grad.device(g_device) += y_grad * dydx;
  }
};

template <typename T>
struct Softmax {
  Softmax() {}

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    // x: [B, D], y: [B, D]
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int batch_size = x.dimension(0), num_class = x.dimension(1);
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    y.device(g_device) = (x - x.maximum(along_class)
                                  .eval()
                                  .reshape(batch_by_one)
                                  .broadcast(one_by_class))
                             .exp();
    y.device(g_device) = y * y.sum(along_class)
                                 .inverse()
                                 .eval()
                                 .reshape(batch_by_one)
                                 .broadcast(one_by_class);
  }

  void Backward(typename TTypes<T>::ConstMatrix y,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
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
    x_grad.device(g_device) += sub * y;

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

template <typename T>
struct SoftmaxCrossEntropy {
  enum Reduction { MEAN, SUM };

  SoftmaxCrossEntropy(Reduction reduction = Reduction::MEAN)
      : reduction_(reduction) {
    softmax_ = std::make_unique<Softmax<T>>();
  }

  void Forward(typename TTypes<T>::ConstMatrix logits,
               absl::Span<const int> targets, typename TTypes<T>::Matrix probs,
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

  void Backward(typename TTypes<T>::ConstMatrix probs,
                absl::Span<const int> targets,
                typename TTypes<T>::Matrix logits_grad) {
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

  static void ForwardAndBackward(typename TTypes<T>::ConstMatrix logits,
                                 typename TTypes<T>::ConstMatrix labels,
                                 typename TTypes<T>::Flat scratch,
                                 typename TTypes<T>::Flat loss,
                                 typename TTypes<T>::Matrix logit_grad) {
    // logits: [B, C], targets: [B,], probs:[B, C], loss: scalar
    int B = logits.dimension(0), C = logits.dimension(1);
    CHECK(B == labels.dimension(0) && C == labels.dimension(1));
    CHECK(B == logit_grad.dimension(0) && C == logit_grad.dimension(1));
    CHECK_EQ(B, scratch.size());
    CHECK_EQ(B, loss.size());

    const int batch_size = B, num_class = C;
    Eigen::array<Eigen::Index, 1> along_class = {1};
    Eigen::array<Eigen::Index, 2> batch_by_one = {batch_size, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, num_class};

    // max_logits along classes.
    scratch.device(g_device) = logits.maximum(along_class);

    // logits - max_logits.
    logit_grad.device(g_device) =
        logits - scratch.reshape(batch_by_one).broadcast(one_by_class);

    // sum(exp(logits - max_logits)) along classes.
    scratch.device(g_device) = logit_grad.exp().sum(along_class);

    // NOTE: Eigen on GPU dispatches to an optimized implementation
    // for an expression of the form lhs = rhs.sum().
    // lhs = -rhs.sum() doesn't match the above pattern, so folding in the
    // negation before calling sum().
    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    loss.device(g_device) =
        (labels * (scratch.log().reshape(batch_by_one).broadcast(one_by_class) -
                   logit_grad))
            .sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    logit_grad.device(g_device) =
        (logit_grad.exp() /
         scratch.reshape(batch_by_one).broadcast(one_by_class)) -
        labels;
  }

  Reduction reduction_;
  std::unique_ptr<Softmax<T>> softmax_;
};

template <typename T>
struct CrossEntropy {
  enum Reduction { MEAN, SUM };

  CrossEntropy(Reduction reduction = Reduction::MEAN) : reduction_(reduction) {}

  void Forward(typename TTypes<T>::ConstMatrix probs,
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

  void Backward(typename TTypes<T>::ConstMatrix probs,
                absl::Span<const int> targets,
                typename TTypes<T>::Matrix probs_grad) {
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
