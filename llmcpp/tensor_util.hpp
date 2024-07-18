#ifndef LLM_CPP_LLMCPP_TENSOR_UTIL_HPP_
#define LLM_CPP_LLMCPP_TENSOR_UTIL_HPP_

#include "absl/container/inlined_vector.h"
#include "tensor_types.hpp"

using floatX = float;

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

#endif  // LLM_CPP_LLMCPP_TENSOR_UTIL_HPP_
