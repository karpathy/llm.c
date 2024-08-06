#ifndef LLM_CPP__GPT_HPP_
#define LLM_CPP__GPT_HPP_

#include "nn.hpp"

namespace gpt {

struct MLP {
  using T = floatX;

  explicit MLP(int n_embed) : n_embed_(n_embed) {
    c_fc_ = std::make_unique<nn::Linear>(n_embed, 4 * n_embed);
    gelu_ = std::make_unique<nn::NewGELU>();
    c_proj_ = std::make_unique<nn::Linear>(4 * n_embed, n_embed);

    // activation
    auto dtype = nn::DataTypeToEnum<T>::value;
    fch_ = std::make_unique<nn::Activation>(dtype);
    fch_gelu_ = std::make_unique<nn::Activation>(dtype);
  }

  void Forward(typename TTypes<T>::ConstMatrix x,
               typename TTypes<T>::Matrix y) {
    // x: [B*T, 4*n_embed], y: [B*T, 4*n_embed]
    CHECK_EQ(x.dimension(1), n_embed_);
    // x.shape == y.shape
    CHECK_EQ(x.dimension(0), y.dimension(0));
    CHECK_EQ(x.dimension(1), y.dimension(1));

    int BT = x.dimension(0);
    fch_->LazyAllocate(BT * 4 * n_embed_);
    fch_gelu_->LazyAllocate(BT * 4 * n_embed_);

    // forward
    auto fch = fch_->matrix<T>(BT, 4 * n_embed_);
    auto fch_gelu = fch_gelu_->matrix<T>(BT, 4 * n_embed_);
    c_fc_->Forward(x, fch);
    gelu_->Forward(MakeConstFlat(fch.data(), fch.size()),
                   MakeFlat(fch_gelu.data(), fch_gelu.size()));
    auto fch_gelu_const = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
    c_proj_->Forward(fch_gelu_const, y);
  }

  void Backward(typename TTypes<T>::ConstMatrix x,
                typename TTypes<T>::ConstMatrix y_grad,
                typename TTypes<T>::Matrix x_grad) {
    // x: [B*T, 4*n_embed], y_grad: [B*T, 4*n_embed]
    // x_grad: [B*T, 4*n_embed]
    CHECK_EQ(x.dimension(1), n_embed_);
    // x.shape == y_grad.shape == x_grad.shape
    CHECK_EQ(x.dimension(0), y_grad.dimension(0));
    CHECK_EQ(x.dimension(1), y_grad.dimension(1));
    CHECK_EQ(x.dimension(0), x_grad.dimension(0));
    CHECK_EQ(x.dimension(1), x_grad.dimension(1));

    // Lazily allocate the memory for activation
    int BT = x.dimension(0);
    fch_->LazyAllocateGradient();
    fch_gelu_->LazyAllocateGradient();
    fch_->ZeroGrad();
    fch_gelu_->ZeroGrad();

    auto fch_gelu = fch_gelu_->const_matrix<T>(BT, 4 * n_embed_);
    auto fch_gelu_grad = fch_gelu_->matrix_grad<T>(BT, 4 * n_embed_);
    c_proj_->Backward(fch_gelu, y_grad, fch_gelu_grad);

    auto fch = fch_->const_flat<T>();
    auto fch_gelu_grad_flat = fch_gelu_->const_flat_grad<T>();
    auto fch_grad = fch_->flat_grad<T>();
    gelu_->Backward(fch, fch_gelu_grad_flat, fch_grad);

    auto fch_grad_2d = fch_->const_matrix_grad<T>(BT, 4 * n_embed_);
    c_fc_->Backward(x, fch_grad_2d, x_grad);
  }

  size_t NumParameters() const {
    return c_fc_->NumParameters() + c_proj_->NumParameters();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    c_fc_->Parameters(parameters);
    c_proj_->Parameters(parameters);
  }

  int n_embed_;
  std::unique_ptr<nn::Linear> c_fc_;
  std::unique_ptr<nn::NewGELU> gelu_;
  std::unique_ptr<nn::Linear> c_proj_;

  // activation tensors
  std::unique_ptr<nn::Activation> fch_;       // [B*T, 4*C]
  std::unique_ptr<nn::Activation> fch_gelu_;  // [B*T, 4*C]
};

struct CausalSelfAttention {
  using Type = floatX;

  CausalSelfAttention(int block_size, int n_head, int n_embed)
      : n_head_(n_head), n_embed_(n_embed) {
    CHECK_EQ(n_embed % n_head, 0);

    // key, query, value projections for all heads, but in a batch
    c_attn_ = std::make_unique<nn::Linear>(n_embed, 3 * n_embed);

    // output projection
    c_proj_ = std::make_unique<nn::Linear>(n_embed, n_embed);

    softmax_ = std::make_unique<nn::Softmax>();

    // mask
    auto dtype = nn::DataTypeToEnum<Type>::value;
    for (int i = 0; i <= block_size; ++i) {
      bias_.emplace_back(std::make_unique<nn::Parameter>(dtype, i * i));
      auto bias_2d = bias_.back()->matrix<Type>(i, i);
      nn::UpperTriangularWithNegativeInf(bias_2d);
    }

    // activation tensors
    qkv_ = std::make_unique<nn::Activation>(dtype);     // [B, T, 3C]
    q_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, T, HS]
    k_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, HS, T]
    v_ = std::make_unique<nn::Activation>(dtype);       // [B, NH, T, HS]
    preatt_ = std::make_unique<nn::Activation>(dtype);  // [B, NH, T, T]
    preatt_softmax_ = std::make_unique<nn::Activation>(dtype);  // [B, NH, T, T]
    att_ = std::make_unique<nn::Activation>(dtype);   // [B, NH, T, HS]
    att2_ = std::make_unique<nn::Activation>(dtype);  // [B, T, NH, HS]
  }

  void Forward(typename TTypes<Type, 3>::ConstTensor x,
               typename TTypes<Type, 3>::Tensor y) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    int NH = n_head_, HS = C / n_head_;
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK(C == n_embed_ && C == y.dimension(2));

    // Lazily allocate the memory for activation
    qkv_->LazyAllocate(B * T * 3 * C);
    q_->LazyAllocate(B * NH * T * HS);
    k_->LazyAllocate(B * NH * HS * T);
    v_->LazyAllocate(B * NH * T * HS);
    preatt_->LazyAllocate(B * NH * T * T);
    preatt_softmax_->LazyAllocate(B * NH * T * T);
    att_->LazyAllocate(B * NH * T * HS);
    att2_->LazyAllocate(B * T * NH * HS);

    auto _x = MakeConstMatrix(x.data(), B * T, C);
    auto qkv = MakeMatrix(qkv_->data<Type>(), B * T, 3 * C);
    c_attn_->Forward(_x, qkv);

    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    Eigen::array<Eigen::Index, 4> shape = {B, T, NH, HS};
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 2, 3, 1};
    auto qkv3d = qkv_->tensor_3d<Type>(B, T, 3 * C);
    auto q_4d = q_->tensor_4d<Type>(B, NH, T, HS);
    auto k_4d = k_->tensor_4d<Type>(B, NH, HS, T);
    auto v_4d = v_->tensor_4d<Type>(B, NH, T, HS);
    q_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_q, extents)  // [B, T, C]
                                    .reshape(shape)       // [B, T, NH, HS]
                                    .shuffle(shuffle_qv)  // [B, NH, T, HS]
        ;
    k_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_k, extents)  // [B, T, C]
                                    .reshape(shape)      //  [B, T, NH, HS]
                                    .shuffle(shuffle_k)  //  [B, NH, HS, T]
        ;
    v_4d.device(nn::g_device) = qkv3d
                                    .slice(offsets_v, extents)  // [B, T, C]
                                    .reshape(shape)       //  [B, T, NH, HS]
                                    .shuffle(shuffle_qv)  //  [B, NH, T, HS]
        ;

    const float factor = 1.0f / std::sqrt(static_cast<float>(HS));
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < NH; ++h) {
        auto q2d =
            MakeConstMatrix(q_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto k2d =
            MakeConstMatrix(k_->data<Type>() + (b * NH + h) * HS * T, HS, T);
        auto v2d = MakeMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto preatt2d =
            MakeMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d = MakeConstMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto att2d =
            MakeMatrix(att_->data<Type>() + (b * NH + h) * T * HS, T, HS);

        nn::MatMul::Forward(q2d, k2d, preatt2d, factor);
        auto bias_2d = bias_[T]->matrix<Type>(T, T);
        preatt2d.device(nn::g_device) = preatt2d + bias_2d;

        // softmax
        auto preatt2d_tensor =
            MakeConstMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d_tensor = MakeMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        softmax_->Forward(preatt2d_tensor, preatt_softmax2d_tensor);

        // att * v
        typename TTypes<Type>::ConstMatrix v2d_const =
            MakeConstMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        nn::MatMul::Forward(preatt_softmax2d, v2d_const, att2d);
      }
    }

    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    auto att_4d = att_->tensor_4d<Type>(B, NH, T, HS);
    auto att2_4d = att2_->tensor_4d<Type>(B, T, NH, HS);
    att2_4d.device(nn::g_device) =
        att_4d.shuffle(shuffle_att);  // [B, T, NH, HS]
    auto att2_2d = MakeConstMatrix(att2_->data<Type>(), B * T, C);
    auto y2d = MakeMatrix(y.data(), B * T, C);
    c_proj_->Forward(att2_2d, y2d);
  }

  void Backward(typename TTypes<Type, 3>::ConstTensor x,
                typename TTypes<Type, 3>::ConstTensor y_grad,
                typename TTypes<Type, 3>::Tensor x_grad) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    int NH = n_head_, HS = C / n_head_;
    CHECK(B == y_grad.dimension(0) && B == x_grad.dimension(0));
    CHECK(T == y_grad.dimension(1) && T == x_grad.dimension(1));
    CHECK(C == y_grad.dimension(2) && C == x_grad.dimension(2));

    // Lazily allocate the memory for activation
    qkv_->LazyAllocateGradient();
    q_->LazyAllocateGradient();
    k_->LazyAllocateGradient();
    v_->LazyAllocateGradient();
    preatt_->LazyAllocateGradient();
    preatt_softmax_->LazyAllocateGradient();
    att_->LazyAllocateGradient();
    att2_->LazyAllocateGradient();
    qkv_->ZeroGrad();
    q_->ZeroGrad();
    k_->ZeroGrad();
    v_->ZeroGrad();
    preatt_->ZeroGrad();
    preatt_softmax_->ZeroGrad();
    att_->ZeroGrad();
    att2_->ZeroGrad();

    // attproj backward
    auto att2_2d = MakeConstMatrix(att2_->data<Type>(), B * T, C);
    auto y_grad_2d = MakeConstMatrix(y_grad.data(), B * T, C);
    auto att2_grad_2d = MakeMatrix(att2_->grad<Type>(), B * T, C);
    c_proj_->Backward(att2_2d, y_grad_2d, att2_grad_2d);

    // shuffle backward
    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    auto att_grad = att_->tensor_4d_grad<Type>(B, NH, T, HS);
    auto att2_grad = att2_->tensor_4d_grad<Type>(B, T, NH, HS);
    att_grad.device(nn::g_device) =
        att2_grad.shuffle(shuffle_att);  // [B, NH, T, HS]

    // attention backward
    float factor = 1.0f / std::sqrt(static_cast<float>(HS));
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < NH; ++h) {
        auto q2d =
            MakeConstMatrix(q_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto q_grad2d =
            MakeMatrix(q_->grad<Type>() + (b * NH + h) * T * HS, T, HS);
        auto k2d =
            MakeConstMatrix(k_->data<Type>() + (b * NH + h) * HS * T, HS, T);
        auto k_grad2d =
            MakeMatrix(k_->grad<Type>() + (b * NH + h) * HS * T, HS, T);
        auto v2d =
            MakeConstMatrix(v_->data<Type>() + (b * NH + h) * T * HS, T, HS);
        auto v_grad2d =
            MakeMatrix(v_->grad<Type>() + (b * NH + h) * T * HS, T, HS);
        auto preatt2d =
            MakeConstMatrix(preatt_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax2d = MakeConstMatrix(
            preatt_softmax_->data<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_grad2d =
            MakeMatrix(preatt_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_grad2d_const =
            MakeConstMatrix(preatt_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax_grad2d = MakeMatrix(
            preatt_softmax_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto preatt_softmax_grad2d_const = MakeConstMatrix(
            preatt_softmax_->grad<Type>() + (b * NH + h) * T * T, T, T);
        auto att_grad2d =
            MakeConstMatrix(att_->grad<Type>() + (b * NH + h) * T * HS, T, HS);

        // backward: att * v
        nn::MatMul::Backward(preatt_softmax2d, v2d, att_grad2d,
                             preatt_softmax_grad2d, v_grad2d);

        // backward: softmax
        softmax_->Backward(preatt_softmax2d, preatt_softmax_grad2d_const,
                           preatt_grad2d);

        // backward: mask
        // backward: q * k
        nn::MatMul::Backward(q2d, k2d, preatt_grad2d_const, q_grad2d, k_grad2d,
                             factor);
      }
    }

    // backward: shuffle -> reshape
    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    Eigen::array<Eigen::Index, 3> shape = {B, T, C};
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 3, 1, 2};
    // q_grad_: [B, NH, T, HS] -> [B, T, NH, HS] -> [B, T, C]
    auto qkv_grad = qkv_->tensor_3d_grad<Type>(B, T, 3 * C);
    auto q_grad = q_->tensor_4d_grad<Type>(B, NH, T, HS);
    auto k_grad = k_->tensor_4d_grad<Type>(B, NH, HS, T);
    auto v_grad = v_->tensor_4d_grad<Type>(B, NH, T, HS);
    qkv_grad.slice(offsets_q, extents).device(nn::g_device) =
        q_grad.shuffle(shuffle_qv).reshape(shape);

    // k_grad_: [B, NH, HS, T] -> [B, T, NH, HS] -> [B, T, C]
    qkv_grad.slice(offsets_k, extents).device(nn::g_device) =
        k_grad.shuffle(shuffle_k).reshape(shape);

    // v_grad_: [B, NH, T, HS] -> [B, T, NH, HS] -> [B, T, C]
    qkv_grad.slice(offsets_v, extents).device(nn::g_device) =
        v_grad.shuffle(shuffle_qv).reshape(shape);

    // backward: qkv
    auto _x = MakeConstMatrix(x.data(), B * T, C);
    auto qkv_grad_2d = MakeConstMatrix(qkv_->grad<Type>(), B * T, 3 * C);
    auto _x_grad = MakeMatrix(x_grad.data(), B * T, C);
    c_attn_->Backward(_x, qkv_grad_2d, _x_grad);
  }

  size_t NumParameters() const {
    return c_attn_->NumParameters() + c_proj_->NumParameters();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    c_attn_->Parameters(parameters);
    c_proj_->Parameters(parameters);
  }

  int n_head_;
  int n_embed_;
  std::unique_ptr<nn::Linear> c_attn_;
  std::unique_ptr<nn::Linear> c_proj_;
  std::unique_ptr<nn::Softmax> softmax_;

  // activation tensors
  std::unique_ptr<nn::Activation> qkv_;             // [B, T, 3C]
  std::unique_ptr<nn::Activation> q_;               // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> k_;               // [B, NH, HS, T]
  std::unique_ptr<nn::Activation> v_;               // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> preatt_;          // [B, NH, T, T]
  std::unique_ptr<nn::Activation> preatt_softmax_;  // [B, NH, T, T]
  std::unique_ptr<nn::Activation> att_;             // [B, NH, T, HS]
  std::unique_ptr<nn::Activation> att2_;            // [B, T, NH, HS]

  // not really a 'bias', more of a mask, but following the OpenAI/HF naming
  // though
  //  Eigen::MatrixXi bias_;
  std::vector<std::unique_ptr<nn::Activation>>
      bias_;  // [0x0], [1x1], ... [block_size, block_size]
};

struct Block {
  using Type = floatX;

  Block(int block_size, int n_head, int n_embed) {
    ln1_ = std::make_unique<nn::LayerNorm>(n_embed);
    attn_ = std::make_unique<CausalSelfAttention>(block_size, n_head, n_embed);
    ln2_ = std::make_unique<nn::LayerNorm>(n_embed);
    mlp_ = std::make_unique<MLP>(n_embed);

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    ln1_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln1_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln1_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    att_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
    residual1_ = std::make_unique<nn::Activation>(dtype);  // [B, T, C]
    ln2_y_ = std::make_unique<nn::Activation>(dtype);      // [B*T, C]
    ln2_mean_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    ln2_rstd_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    mlp_y_ = std::make_unique<nn::Activation>(dtype);      // [B, T, C]
  }

  void Forward(typename TTypes<Type, 3>::ConstTensor x,
               typename TTypes<Type, 3>::Tensor y) {
    // x: [B, T, C], y: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK_EQ(C, y.dimension(2));

    ln1_y_->LazyAllocate(B * T * C);
    ln1_mean_->LazyAllocate(B * T);
    ln1_rstd_->LazyAllocate(B * T);
    att_y_->LazyAllocate(B * T * C);
    residual1_->LazyAllocate(B * T * C);
    ln2_y_->LazyAllocate(B * T * C);
    ln2_mean_->LazyAllocate(B * T);
    ln2_rstd_->LazyAllocate(B * T);
    mlp_y_->LazyAllocate(B * T * C);

    // LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_y_2d = MakeMatrix(ln1_y_->data<Type>(), B * T, C);
    auto ln1_mean_1d = MakeFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeFlat(ln1_rstd_->data<Type>(), B * T);
    ln1_->Forward(x_2d, ln1_y_2d, ln1_mean_1d, ln1_rstd_1d);

    // Attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_2d.data(), B, T, C);
    auto att_y_3d = Make3DTensor(att_y_->data<Type>(), B, T, C);
    attn_->Forward(ln1_y_3d, att_y_3d);

    // Residual
    auto x_1d = MakeConstFlat(x.data(), B * T * C);
    auto att_y_1d = MakeConstFlat(att_y_->data<Type>(), B * T * C);
    auto residual1_1d = MakeFlat(residual1_->data<Type>(), residual1_->size());
    nn::Residual::Forward(x_1d, att_y_1d, residual1_1d);

    // LN2
    auto ln2_y_2d = MakeMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_2d_const = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_mean_1d = MakeFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    ln2_->Forward(residual1_2d, ln2_y_2d, ln2_mean_1d, ln2_rstd_1d);

    // MLP
    auto mlp_y_2d = MakeMatrix(mlp_y_->data<Type>(), B * T, C);
    mlp_->Forward(ln2_y_2d_const, mlp_y_2d);

    // Residual
    auto residual1_1d_const =
        MakeConstFlat(residual1_->data<Type>(), residual1_->size());
    auto mlp_y_1d = MakeConstFlat(mlp_y_->data<Type>(), B * T * C);
    auto y_1d = MakeFlat(y.data(), y.size());
    nn::Residual::Forward(residual1_1d_const, mlp_y_1d, y_1d);
  }

  void Backward(typename TTypes<Type, 3>::ConstTensor x,
                typename TTypes<Type, 3>::ConstTensor y_grad,
                typename TTypes<Type, 3>::Tensor x_grad) {
    // x: [B, T, C], y_grad: [B, T, C], x_grad: [B, T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y_grad.dimension(0));
    CHECK_EQ(T, y_grad.dimension(1));
    CHECK_EQ(C, y_grad.dimension(2));
    CHECK_EQ(B, x_grad.dimension(0));
    CHECK_EQ(T, x_grad.dimension(1));
    CHECK_EQ(C, x_grad.dimension(2));

    ln1_y_->LazyAllocateGradient();
    att_y_->LazyAllocateGradient();
    residual1_->LazyAllocateGradient();
    ln2_y_->LazyAllocateGradient();
    mlp_y_->LazyAllocateGradient();
    ln1_y_->ZeroGrad();
    att_y_->ZeroGrad();
    residual1_->ZeroGrad();
    ln2_y_->ZeroGrad();
    mlp_y_->ZeroGrad();

    // backward residual
    auto y_grad_1d = MakeConstFlat(y_grad.data(), y_grad.size());
    auto residual1_grad_1d =
        MakeFlat(residual1_->grad<Type>(), residual1_->size());
    auto mlp_y_grad_1d = MakeFlat(mlp_y_->grad<Type>(), mlp_y_->size());
    nn::Residual::Backward(y_grad_1d, residual1_grad_1d, mlp_y_grad_1d);

    // backward MLP
    auto ln2_y_2d = MakeConstMatrix(ln2_y_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d = MakeMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto mlp_y_grad_2d = MakeConstMatrix(mlp_y_->grad<Type>(), B * T, C);
    mlp_->Backward(ln2_y_2d, mlp_y_grad_2d, ln2_y_grad_2d);

    // backward LN2
    auto ln2_mean_1d = MakeConstFlat(ln2_mean_->data<Type>(), B * T);
    auto ln2_rstd_1d = MakeConstFlat(ln2_rstd_->data<Type>(), B * T);
    auto residual1_2d = MakeConstMatrix(residual1_->data<Type>(), B * T, C);
    auto ln2_y_grad_2d_const = MakeConstMatrix(ln2_y_->grad<Type>(), B * T, C);
    auto residual1_grad_2d = MakeMatrix(residual1_->grad<Type>(), B * T, C);
    ln2_->Backward(residual1_2d, ln2_y_grad_2d_const, ln2_mean_1d, ln2_rstd_1d,
                   residual1_grad_2d);

    // backward residual
    auto residual1_grad_1d_const =
        MakeConstFlat(residual1_->grad<Type>(), residual1_->size());
    auto x_grad_1d = MakeFlat(x_grad.data(), x_grad.size());
    auto att_y_grad_1d = MakeFlat(att_y_->grad<Type>(), att_y_->size());
    nn::Residual::Backward(residual1_grad_1d_const, x_grad_1d, att_y_grad_1d);

    // backward attention
    auto ln1_y_3d = MakeConst3DTensor(ln1_y_->data<Type>(), B, T, C);
    auto ln1_y_grad_3d = Make3DTensor(ln1_y_->grad<Type>(), B, T, C);
    auto att_y_grad_3d = MakeConst3DTensor(att_y_->grad<Type>(), B, T, C);
    attn_->Backward(ln1_y_3d, att_y_grad_3d, ln1_y_grad_3d);

    // backward LN1
    auto x_2d = MakeConstMatrix(x.data(), B * T, C);
    auto ln1_mean_1d = MakeConstFlat(ln1_mean_->data<Type>(), B * T);
    auto ln1_rstd_1d = MakeConstFlat(ln1_rstd_->data<Type>(), B * T);
    auto ln1_y_grad_2d = MakeConstMatrix(ln1_y_->grad<Type>(), B * T, C);
    auto x_grad_2d = MakeMatrix(x_grad.data(), B * T, C);
    ln1_->Backward(x_2d, ln1_y_grad_2d, ln1_mean_1d, ln1_rstd_1d, x_grad_2d);
  }

  size_t NumParameters() const {
    return ln1_->NumParameters() + attn_->NumParameters() +
           ln2_->NumParameters() + mlp_->NumParameters();
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    ln1_->Parameters(parameters);
    attn_->Parameters(parameters);
    ln2_->Parameters(parameters);
    mlp_->Parameters(parameters);
  }

  std::unique_ptr<nn::LayerNorm> ln1_;
  std::unique_ptr<CausalSelfAttention> attn_;
  std::unique_ptr<nn::LayerNorm> ln2_;
  std::unique_ptr<MLP> mlp_;

  // activation tensors
  std::unique_ptr<nn::Activation> ln1_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln1_mean_, ln1_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> att_y_;                // [B, T, C]
  std::unique_ptr<nn::Activation> residual1_;            // [B, T, C]
  std::unique_ptr<nn::Activation> ln2_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> ln2_mean_, ln2_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> mlp_y_;                // [B, T, C]
};

struct GPT {
  using Type = floatX;

  GPT(int block_size, int vocab_size, int padded_vocab_size, int n_layer,
      int n_head, int n_embed)
      : block_size_(block_size),
        vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        n_layer_(n_layer),
        n_embed_(n_embed),
        lm_head_(nullptr),
        lm_head_grad_(nullptr) {
    CHECK_GT(n_layer, 0);

    wte_ = std::make_unique<nn::Embedding>(padded_vocab_size, n_embed);
    wpe_ = std::make_unique<nn::Embedding>(block_size, n_embed);
    for (int i = 0; i < n_layer; ++i) {
      h_.emplace_back(std::make_unique<Block>(block_size, n_head, n_embed));
    }
    lnf_ = std::make_unique<nn::LayerNorm>(n_embed);

    lm_head_unused_ = std::make_unique<nn::Linear>(n_embed, vocab_size);
    // https://paperswithcode.com/method/weight-tying
    nn::g_device.memcpy(wte_->weight_->data<Type>(),
                        lm_head_unused_->weight_->template data<Type>(),
                        sizeof(float) * vocab_size * n_embed);
    nn::g_device.memset(
        wte_->weight_->data<Type>() + vocab_size * n_embed, 0,
        sizeof(float) * (padded_vocab_size - vocab_size) * n_embed);
    lm_head_ = wte_->weight_->data<Type>();
    softmax_cross_entropy_ = std::make_unique<nn::SoftmaxCrossEntropy>();

    // activation
    auto dtype = nn::DataTypeToEnum<Type>::value;
    tok_emb_ = std::make_unique<nn::Activation>(dtype);   // [B, T, C]
    pos_emb_ = std::make_unique<nn::Activation>(dtype);   // [T, C]
    encoded_ = std::make_unique<nn::Activation>(dtype);   // [B, T, C]
    block_y_ = std::make_unique<nn::Activation>(dtype);   // [L, B, T, C]
    lnf_y_ = std::make_unique<nn::Activation>(dtype);     // [B*T, C]
    lnf_mean_ = std::make_unique<nn::Activation>(dtype);  // [B*T]
    lnf_rstd_ = std::make_unique<nn::Activation>(dtype);  // [B*T]
    scratch_ = std::make_unique<nn::Activation>(dtype);   // [B*T]
    loss_ = std::make_unique<nn::Activation>(dtype);      // [B*T]
    probs_ = std::make_unique<nn::Activation>(dtype);     // [B*T, vocab_size]
    logits_grad_ =
        std::make_unique<nn::Activation>(dtype);  // [B*T, vocab_size]
  }

  void __Forward(typename TTypes<int>::ConstMatrix idx) {
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    CHECK_LE(T, block_size_) << "Cannot forward sequence of length " << T
                             << ", block size is only " << block_size_;
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);

    // Lazily allocate memory
    tok_emb_->LazyAllocate(B * T * C);
    pos_emb_->LazyAllocate(T * C);
    encoded_->LazyAllocate(B * T * C);
    block_y_->LazyAllocate(L * B * T * C);
    lnf_y_->LazyAllocate(BT * C);
    lnf_mean_->LazyAllocate(BT);
    lnf_rstd_->LazyAllocate(BT);

    wte_->Forward(idx,
                  absl::MakeSpan(tok_emb_->data<Type>(), tok_emb_->size()));
    wpe_->Forward(pos,
                  absl::MakeSpan(pos_emb_->data<Type>(), pos_emb_->size()));

    auto tok_emb = tok_emb_->matrix<Type>(B, TC);
    auto pos_emb = pos_emb_->flat<Type>();
    auto encoded = encoded_->matrix<Type>(B, TC);
    Eigen::array<Eigen::Index, 2> batch_by_one = {B, 1};
    Eigen::array<Eigen::Index, 2> one_by_class = {1, TC};
    encoded.device(nn::g_device) =
        tok_emb + pos_emb.reshape(one_by_class).broadcast(batch_by_one);

    for (int l = 0; l < n_layer_; ++l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* y = block_y_->data<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_y_3d = Make3DTensor(y, B, T, C);
      block->Forward(block_x_3d, block_y_3d);
    }

    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_mean = MakeFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeFlat(lnf_rstd_->data<Type>(), BT);
    lnf_->Forward(block_out_2d, lnf_y, lnf_mean, lnf_rstd);
  }

  void Forward(typename TTypes<int>::ConstMatrix idx,
               typename TTypes<Type, 3>::Tensor logits) {
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    // OPTIMIZE:
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    //    auto lnf_y_3d = Eigen::TensorMap<nn::Tensor3D>(lnf_y_.data(), B, T,
    //    C); nn::Tensor2D lnf_y_last_t = lnf_y_3d.chip(T - 1, 1);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    //    nn::MatMul::Forward(lnf_y, lm_head, logits_2d);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);
  }

  void SoftmaxForwardCPU(typename TTypes<Type>::ConstMatrix logits,
                         absl::Span<const int> targets, float* loss) {
    int BT = logits.dimension(0);
    CHECK_EQ(BT, targets.size());
    CHECK_EQ(vocab_size_, logits.dimension(1));
    probs_->LazyAllocate(BT * vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Forward(logits, targets, probs_2d, loss);
  }

  void SoftmaxForwardGPU(typename TTypes<Type>::ConstMatrix logits,
                         typename TTypes<Type>::ConstMatrix labels,
                         float* loss) {
    int BT = logits.dimension(0);
    CHECK_EQ(BT, labels.dimension(0));
    CHECK_EQ(vocab_size_, logits.dimension(1));
    CHECK_EQ(vocab_size_, labels.dimension(1));
    scratch_->LazyAllocate(BT);
    loss_->LazyAllocate(BT);
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto logits_grad = MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    nn::SoftmaxCrossEntropy::ForwardAndBackward(
        logits, labels, scratch_->template flat<Type>(),
        loss_->template flat<Type>(), logits_grad);
    logits_grad.device(nn::g_device) = logits_grad * (1.0f / BT);
    TTypes<float>::Scalar loss_scalar(loss);
    loss_scalar.device(nn::g_device) = loss_->template flat<Type>().mean();
  }

  void ForwardCPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<int>::ConstMatrix targets,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(targets.dimension(0) == B && targets.dimension(1) == T);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);
    auto probs_2d = MakeMatrix(probs_->data<Type>(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    SoftmaxForwardCPU(logits_2d_const, targets, loss);
  }

  void ForwardGPU(typename TTypes<int>::ConstMatrix idx,
                  typename TTypes<Type, 3>::ConstTensor labels,
                  typename TTypes<Type, 3>::Tensor logits, float* loss) {
    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_;
    const int BT = B * T;
    CHECK(labels.dimension(0) == B && labels.dimension(1) == T &&
          labels.dimension(2) == vocab_size_);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    __Forward(idx);

    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = MakeMatrix(logits.data(), BT, vocab_size_);

    // [BT, C] x [C, vocab_size] -> [BT, vocab_size]
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    logits_2d.device(nn::g_device) = lnf_y.contract(lm_head, product_dims);

    auto logits_2d_const = MakeConstMatrix(logits.data(), BT, vocab_size_);
    auto labels_2d_const = MakeConstMatrix(labels.data(), BT, vocab_size_);
    SoftmaxForwardGPU(logits_2d_const, labels_2d_const, loss);
  }

  void SoftmaxBackwardCPU(absl::Span<const int> targets) {
    int BT = targets.size();
    logits_grad_->LazyAllocate(BT * vocab_size_);
    logits_grad_->ZeroData();
    auto probs_2d = MakeConstMatrix(probs_->data<Type>(), BT, vocab_size_);
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    softmax_cross_entropy_->Backward(probs_2d, targets, logits_grad_2d);
  }

  void BackwardCPU(typename TTypes<int>::ConstMatrix idx,
                   typename TTypes<int>::ConstMatrix targets) {
    SoftmaxBackwardCPU(targets);
    BackwardGPU(idx);
  }

  void BackwardGPU(typename TTypes<int>::ConstMatrix idx) {
    // idx: [B, T], targets: [B, T]
    const int B = idx.dimension(0), T = idx.dimension(1), C = n_embed_,
              L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;

    wte_->weight_->LazyAllocateGradient();
    if (lm_head_grad_ == nullptr) {
      lm_head_grad_ = wte_->weight_->grad<Type>();
    }

    tok_emb_->LazyAllocateGradient();
    pos_emb_->LazyAllocateGradient();
    encoded_->LazyAllocateGradient();
    block_y_->LazyAllocateGradient();
    lnf_y_->LazyAllocateGradient();

    tok_emb_->ZeroGrad();
    pos_emb_->ZeroGrad();
    encoded_->ZeroGrad();
    block_y_->ZeroGrad();
    lnf_y_->ZeroGrad();

    // backward lm_head
    auto logits_grad_2d =
        MakeMatrix(logits_grad_->data<Type>(), BT, vocab_size_);
    auto lnf_y = MakeMatrix(lnf_y_->data<Type>(), BT, C);
    auto lnf_y_grad = MakeMatrix(lnf_y_->grad<Type>(), BT, C);
    auto lm_head = MakeMatrix(lm_head_, vocab_size_, C);
    auto lm_head_grad = MakeMatrix(lm_head_grad_, vocab_size_, C);
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims2 = {
        Eigen::IndexPair<int>(0, 0)};
    lnf_y_grad.device(nn::g_device) += logits_grad_2d.contract(
        lm_head, product_dims);  // [BT, vocab_size] x [vocab_size, C]
    lm_head_grad.device(nn::g_device) += logits_grad_2d.contract(
        lnf_y, product_dims2);  // [vocab_size, BT] x [BT, C]

    // backward LNF
    auto block_out_2d =
        MakeConstMatrix(block_y_->data<Type>() + (L - 1) * BTC, BT, C);
    auto block_out_grad_2d =
        MakeMatrix(block_y_->grad<Type>() + (L - 1) * BTC, BT, C);
    auto lnf_mean = MakeConstFlat(lnf_mean_->data<Type>(), BT);
    auto lnf_rstd = MakeConstFlat(lnf_rstd_->data<Type>(), BT);
    auto lnf_y_grad_2d = MakeConstMatrix(lnf_y_->grad<Type>(), BT, C);
    lnf_->Backward(block_out_2d, lnf_y_grad_2d, lnf_mean, lnf_rstd,
                   block_out_grad_2d);

    // backward blocks
    for (int l = n_layer_ - 1; l >= 0; --l) {
      const auto& block = h_[l];
      Type* x = l == 0 ? encoded_->data<Type>()
                       : block_y_->data<Type>() + (l - 1) * BTC;
      Type* x_grad = l == 0 ? encoded_->grad<Type>()
                            : block_y_->grad<Type>() + (l - 1) * BTC;
      Type* y_grad = block_y_->grad<Type>() + l * BTC;
      auto block_x_3d = MakeConst3DTensor(x, B, T, C);
      auto block_x_grad_3d = Make3DTensor(x_grad, B, T, C);
      auto block_y_grad_3d = MakeConst3DTensor(y_grad, B, T, C);
      block->Backward(block_x_3d, block_y_grad_3d, block_x_grad_3d);
    }

    // backward tok_emb, pos_emb
    auto encoded_grad = encoded_->matrix_grad<Type>(B, TC);
    auto tok_emb_grad = tok_emb_->matrix_grad<Type>(B, TC);
    auto pos_emb_grad = pos_emb_->flat_grad<Type>();
    Eigen::array<Eigen::Index, 1> along_batch = {0};
    tok_emb_grad.device(nn::g_device) = encoded_grad;
    pos_emb_grad.device(nn::g_device) = tok_emb_grad.sum(along_batch);

    // backward wte, wpe
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);
    wte_->Backward(idx, tok_emb_grad);
    wpe_->Backward(pos, pos_emb_grad);
  }

  size_t NumParameters() const {
    size_t num_parameters = 0;
    num_parameters += wte_->NumParameters();
    num_parameters += wpe_->NumParameters();
    for (const auto& b : h_) {
      num_parameters += b->NumParameters();
    }
    num_parameters += lnf_->NumParameters();
    return num_parameters;
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    wte_->Parameters(parameters);
    wpe_->Parameters(parameters);
    for (const auto& b : h_) {
      b->Parameters(parameters);
    }
    lnf_->Parameters(parameters);
  }

 public:
  int block_size_;
  int vocab_size_;
  int padded_vocab_size_;
  int n_layer_;
  int n_embed_;

  // transformer
  std::unique_ptr<nn::Embedding> wte_;
  std::unique_ptr<nn::Embedding> wpe_;
  std::vector<std::unique_ptr<Block>> h_;
  std::unique_ptr<nn::LayerNorm> lnf_;
  std::unique_ptr<nn::SoftmaxCrossEntropy> softmax_cross_entropy_;

  // head
  std::unique_ptr<nn::Linear> lm_head_unused_;
  Type *lm_head_, *lm_head_grad_;  // [vocal_size, C]

  // activation tensors and gradients
  std::unique_ptr<nn::Activation> tok_emb_;              // [B, T, C]
  std::unique_ptr<nn::Activation> pos_emb_;              // [T, C]
  std::unique_ptr<nn::Activation> encoded_;              // [B, T, C]
  std::unique_ptr<nn::Activation> block_y_;              // [L, B, T, C]
  std::unique_ptr<nn::Activation> lnf_y_;                // [B*T, C]
  std::unique_ptr<nn::Activation> lnf_mean_, lnf_rstd_;  // [B*T]
  std::unique_ptr<nn::Activation> probs_;                // [B*T, vocab_size]
  std::unique_ptr<nn::Activation> scratch_;              // [B*T]
  std::unique_ptr<nn::Activation> loss_;                 // [B*T]
  std::unique_ptr<nn::Activation> logits_grad_;          // [B*T, vocab_size]
};

}  // namespace gpt

#endif  // LLM_CPP__GPT_HPP_
