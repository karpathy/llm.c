#ifndef LLM_CPP_LLMCPP_SPAN_HPP_
#define LLM_CPP_LLMCPP_SPAN_HPP_

#include "absl/container/inlined_vector.h"
#include "tensor_types.hpp"

using floatX = float;

struct Span1D {
  using T = floatX;
  static Span1D Make(T* array, size_t size) { return Span1D(array, size); }

  size_t size() const { return flat_.size(); }

  TTypes<T, 1>::Tensor View1D() const { return {flat_.data(), flat_.size()}; }

  TTypes<T, 2>::Tensor View2D(int rows, int cols) const {
    return {flat_.data(), rows, cols};
  }

  TTypes<T, 3>::Tensor View3D(int dim0, int dim1, int dim2) const {
    CHECK_EQ(dim0 * dim1 * dim2, flat_.size());
    return {flat_.data(), dim0, dim1, dim2};
  }

  TTypes<T, 4>::Tensor View4D(int dim0, int dim1, int dim2, int dim3) const {
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, flat_.size());
    return {flat_.data(), dim0, dim1, dim2, dim3};
  }

 private:
  Span1D(T* array, size_t size) : flat_(array, size) {}

  absl::Span<T> flat_;
};

struct Span2D {
  using T = floatX;
  static Span2D Make(T* array, int rows, int cols) {
    return {array, rows, cols};
  }

  size_t size() const { return flat_.size(); }
  int rows() const { return rows_; }
  int cols() const { return cols_; }

  TTypes<T, 1>::Tensor View1D(int length) const {
    CHECK_EQ(length, flat_.size());
    return {flat_.data(), flat_.size()};
  }

  TTypes<T, 2>::Tensor View2D() const { return {flat_.data(), rows_, cols_}; }

  TTypes<T, 3>::Tensor View3D(int dim0, int dim1, int dim2) const {
    CHECK_EQ(dim0 * dim1 * dim2, flat_.size());
    return {flat_.data(), dim0, dim1, dim2};
  }

  TTypes<T, 4>::Tensor View4D(int dim0, int dim1, int dim2, int dim3) const {
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, flat_.size());
    return {flat_.data(), dim0, dim1, dim2, dim3};
  }

 private:
  Span2D(T* array, int rows, int cols)
      : flat_(array, rows * cols), rows_(rows), cols_(cols) {
    CHECK_GE(rows, 0);
    CHECK_GE(cols, 0);
  }

  absl::Span<T> flat_;
  int rows_;
  int cols_;
};

struct Span3D {
  using T = floatX;
  static Span3D Make(T* array, int dim0, int dim1, int dim2) {
    return {array, dim0, dim1, dim2};
  }

  size_t size() const { return flat_.size(); }
  int dim0() const { return dim0_; }
  int dim1() const { return dim1_; }
  int dim2() const { return dim2_; }

  TTypes<T, 1>::Tensor View1D(int length) const {
    CHECK_EQ(length, flat_.size());
    return {flat_.data(), flat_.size()};
  }

  TTypes<T, 2>::Tensor View2D(int rows, int cols) const {
    CHECK_EQ(rows * cols, flat_.size());
    return {flat_.data(), rows, cols};
  }

  TTypes<T, 3>::Tensor View3D() const {
    return {flat_.data(), dim0_, dim1_, dim2_};
  }

  TTypes<T, 4>::Tensor View4D(int dim0, int dim1, int dim2, int dim3) const {
    CHECK_EQ(dim0 * dim1 * dim2 * dim3, flat_.size());
    return {flat_.data(), dim0, dim1, dim2, dim3};
  }

 private:
  Span3D(T* array, int dim0, int dim1, int dim2)
      : flat_(array, dim0 * dim1 * dim2),
        dim0_(dim0),
        dim1_(dim1),
        dim2_(dim2) {
    CHECK_GE(dim0, 0);
    CHECK_GE(dim1, 0);
    CHECK_GE(dim2, 0);
  }

  absl::Span<T> flat_;
  int dim0_;
  int dim1_;
  int dim2_;
};

// Raw pointer -> Flat
template <typename T>
typename TTypes<T>::Flat MakeFlat(T* t, int length) {
  return {t, length};
}
template <typename T>
typename TTypes<T>::ConstFlat MakeConstFlat(T* t, int length) {
  return {t, length};
}
template <typename T>
typename TTypes<T>::ConstFlat MakeConstFlat(const T* t, int length) {
  return {t, length};
}

// Raw pointer -> Matrix
template <typename T>
typename TTypes<T>::Matrix MakeMatrix(T* t, int rows, int cols) {
  return {t, rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix MakeConstMatrix(T* t, int rows, int cols) {
  return {t, rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix MakeConstMatrix(const T* t, int rows,
                                                int cols) {
  return {t, rows, cols};
}

// Raw pointer -> 3D Tensor
template <typename T>
typename TTypes<T, 3>::Tensor Make3DTensor(T* t, int dim0, int dim1, int dim2) {
  return {t, dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor MakeConst3DTensor(T* t, int dim0, int dim1,
                                                     int dim2) {
  return {t, dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor MakeConst3DTensor(const T* t, int dim0,
                                                     int dim1, int dim2) {
  return {t, dim0, dim1, dim2};
}

// Raw pointer -> 4D Tensor
template <typename T>
typename TTypes<T, 4>::Tensor Make4DTensor(T* t, int dim0, int dim1, int dim2,
                                           int dim3) {
  return {t, dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor MakeConst4DTensor(T* t, int dim0, int dim1,
                                                     int dim2, int dim3) {
  return {t, dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor MakeConst4DTensor(const T* t, int dim0,
                                                     int dim1, int dim2,
                                                     int dim3) {
  return {t, dim0, dim1, dim2, dim3};
}

// Flat -> Matrix
template <typename T>
typename TTypes<T>::Matrix shaped(typename TTypes<T>::Flat t, int rows,
                                  int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T>::Flat t, int rows,
                                       int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T>::ConstFlat t,
                                       int rows, int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}

// Flat -> 3D Tensor
template <typename T>
typename TTypes<T, 3>::Tensor shaped(typename TTypes<T>::Flat t, int dim0,
                                     int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T>::Flat t, int dim0,
                                          int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T>::ConstFlat t,
                                          int dim0, int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}

// Flat -> 4D Tensor
template <typename T>
typename TTypes<T, 4>::Tensor shaped(typename TTypes<T>::Flat t, int dim0,
                                     int dim1, int dim2, int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T>::Flat t, int dim0,
                                          int dim1, int dim2, int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T>::ConstFlat t,
                                          int dim0, int dim1, int dim2,
                                          int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}

// Matrix -> Flat
template <typename T>
typename TTypes<T>::Flat shaped(typename TTypes<T>::Matrix t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T>::Matrix t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T>::ConstMatrix t) {
  return {t.data(), t.size()};
}

// Matrix -> 3D Tensor
template <typename T>
typename TTypes<T, 3>::Tensor shaped(typename TTypes<T>::Matrix t, int dim0,
                                     int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T>::Matrix t,
                                          int dim0, int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T>::ConstMatrix t,
                                          int dim0, int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}

// Matrix -> 4D Tensor
template <typename T>
typename TTypes<T, 4>::Tensor shaped(typename TTypes<T>::Matrix t, int dim0,
                                     int dim1, int dim2, int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T>::Matrix t,
                                          int dim0, int dim1, int dim2,
                                          int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T>::ConstMatrix t,
                                          int dim0, int dim1, int dim2,
                                          int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}

// 3D Tensor -> Flat
template <typename T>
typename TTypes<T>::Flat shaped(typename TTypes<T, 3>::Tensor t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T, 3>::Tensor t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T, 3>::ConstTensor t) {
  return {t.data(), t.size()};
}

// 3D Tensor -> Matrix
template <typename T>
typename TTypes<T>::Matrix shaped(typename TTypes<T, 3>::Tensor t, int rows,
                                  int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T, 3>::Tensor t,
                                       int rows, int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T, 3>::ConstTensor t,
                                       int rows, int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}

// 3D Tensor -> 4D Tensor
template <typename T>
typename TTypes<T, 4>::Tensor shaped(typename TTypes<T, 3>::Tensor t, int dim0,
                                     int dim1, int dim2, int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T, 3>::Tensor t,
                                          int dim0, int dim1, int dim2,
                                          int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}
template <typename T>
typename TTypes<T, 4>::ConstTensor shaped(typename TTypes<T, 3>::ConstTensor t,
                                          int dim0, int dim1, int dim2,
                                          int dim3) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2 * dim3);
  return {t.data(), dim0, dim1, dim2, dim3};
}

// 4D Tensor -> Flat
template <typename T>
typename TTypes<T>::Flat shaped(typename TTypes<T, 4>::Tensor t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T, 4>::Tensor t) {
  return {t.data(), t.size()};
}
template <typename T>
typename TTypes<T>::ConstFlat shaped(typename TTypes<T, 4>::ConstTensor t) {
  return {t.data(), t.size()};
}

// 4D Tensor -> Matrix
template <typename T>
typename TTypes<T>::Matrix shaped(typename TTypes<T, 4>::Tensor t, int rows,
                                  int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T, 4>::Tensor t,
                                       int rows, int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}
template <typename T>
typename TTypes<T>::ConstMatrix shaped(typename TTypes<T, 4>::ConstTensor t,
                                       int rows, int cols) {
  CHECK_EQ(t.size(), rows * cols);
  return {t.data(), rows, cols};
}

// 4D Tensor -> 3D Tensor
template <typename T>
typename TTypes<T, 3>::Tensor shaped(typename TTypes<T, 4>::Tensor t, int dim0,
                                     int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T, 4>::Tensor t,
                                          int dim0, int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}
template <typename T>
typename TTypes<T, 3>::ConstTensor shaped(typename TTypes<T, 4>::ConstTensor t,
                                          int dim0, int dim1, int dim2) {
  CHECK_EQ(t.size(), dim0 * dim1 * dim2);
  return {t.data(), dim0, dim1, dim2};
}

#endif  // LLM_CPP_LLMCPP_SPAN_HPP_
