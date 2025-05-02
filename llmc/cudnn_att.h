/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             floatX* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

#endif // CUDNN_ATT_H