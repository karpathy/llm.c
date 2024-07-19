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

#endif  // LLM_CPP_LLMCPP_TENSOR_UTIL_HPP_
