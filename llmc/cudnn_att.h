/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"
#include <cudnn_frontend.h>

// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef half floatX;
#define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
typedef __nv_bfloat16 floatX;
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

// forward declarations of functions defined in cudnn_att.cpp
void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C);

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C);

#endif // CUDNN_ATT_H