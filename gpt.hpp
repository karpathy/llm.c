#ifndef LLM_CPP__GPT_HPP_
#define LLM_CPP__GPT_HPP_

#include "nn.hpp"

namespace gpt {

#define LAZY_ALLOCATE_VECTOR(v, X) \
  do {                             \
    if (v.size() == 0) {           \
      v.resize(X);                 \
    }                              \
    CHECK(v.size() == X);          \
  } while (false)

#define LAZY_ALLOCATE_MATRIX(m, X, Y)      \
  do {                                     \
    if (m.size() == 0) {                   \
      m.resize(X, Y);                      \
    }                                      \
    CHECK(m.rows() == X && m.cols() == Y); \
  } while (false)

#define LAZY_ALLOCATE_TENSOR3D(t, X, Y, Z)                                    \
  do {                                                                        \
    if (t.size() == 0) {                                                      \
      t.resize(X, Y, Z);                                                      \
    }                                                                         \
    CHECK(t.dimension(0) == X && t.dimension(1) == Y && t.dimension(2) == Z); \
  } while (false)

#define LAZY_ALLOCATE_TENSOR4D(t, A, B, C, D)                                  \
  do {                                                                         \
    if (t.size() == 0) {                                                       \
      t.resize(A, B, C, D);                                                    \
    }                                                                          \
    CHECK(t.dimension(0) == A && t.dimension(1) == B && t.dimension(2) == C && \
          t.dimension(3) == D);                                                \
  } while (false)

struct MLP {
  explicit MLP(int n_embed) : n_embed_(n_embed) {
    c_fc_ = std::make_unique<nn::Linear>(n_embed, 4 * n_embed);
    gelu_ = std::make_unique<nn::NewGELU>();
    c_proj_ = std::make_unique<nn::Linear>(4 * n_embed, n_embed);
  }

  void Forward(const Eigen::Map<nn::Matrix>& x, Eigen::Map<nn::Matrix>& y) {
    // x: [B*T, n_embed], y: [B*T, n_embed]
    CHECK_EQ(x.cols(), n_embed_);
    // x.shape == y.shape
    CHECK_EQ(x.rows(), y.rows());
    CHECK_EQ(x.cols(), y.cols());

    LAZY_ALLOCATE_MATRIX(fch_, x.rows(), 4 * n_embed_);
    LAZY_ALLOCATE_MATRIX(fch_gelu_, x.rows(), 4 * n_embed_);

    // forward
    Eigen::Map<nn::Matrix> fch(fch_.data(), fch_.rows(), fch_.cols());
    Eigen::Map<nn::Matrix> fch_gelu(fch_gelu_.data(), fch_gelu_.rows(),
                                    fch_gelu_.cols());
    c_fc_->Forward(x, fch);
    gelu_->Forward(absl::MakeSpan(fch.data(), fch.size()),
                   absl::MakeSpan(fch_gelu.data(), fch_gelu.size()));
    c_proj_->Forward(fch_gelu, y);
  }

  void Backward(const Eigen::Map<nn::Matrix>& x,
                const Eigen::Map<nn::Matrix>& y_grad,
                Eigen::Map<nn::Matrix>& x_grad) {
    // x: [B*T, n_embed], y_grad: [B*T, n_embed], x_grad: [B*T, n_embed]
    CHECK_EQ(x.cols(), n_embed_);
    // x.shape == y_grad.shape == x_grad.shape
    CHECK_EQ(x.rows(), y_grad.rows());
    CHECK_EQ(x.cols(), y_grad.cols());
    CHECK_EQ(x.rows(), x_grad.rows());
    CHECK_EQ(x.cols(), x_grad.cols());

    // Lazily allocate the memory for activation
    if (fch_grad_.size() == 0) {
      fch_grad_ = nn::Matrix::Zero(x.rows(), 4 * n_embed_);
    }
    if (fch_gelu_grad_.size() == 0) {
      fch_gelu_grad_ = nn::Matrix::Zero(x.rows(), 4 * n_embed_);
    }
    CHECK_EQ(x.rows(), fch_grad_.rows());
    CHECK_EQ(x.rows(), fch_gelu_grad_.rows());

    Eigen::Map<nn::Matrix> fch_gelu = nn::GetMatrixMap(fch_gelu_);
    Eigen::Map<nn::Matrix> fch_gelu_grad = nn::GetMatrixMap(fch_gelu_grad_);
    c_proj_->Backward(fch_gelu, y_grad, fch_gelu_grad);

    auto fch = absl::MakeSpan(fch_);
    Eigen::Map<nn::Matrix> fch_grad = nn::GetMatrixMap(fch_grad_);
    gelu_->Backward(fch, fch_gelu_grad, absl::MakeSpan(fch_grad));

    c_fc_->Backward(x, fch_grad, x_grad);
  }

  size_t NumParameters() const {
    return c_fc_->NumParameters() + c_proj_->NumParameters();
  }

  int n_embed_;
  std::unique_ptr<nn::Linear> c_fc_;
  std::unique_ptr<nn::NewGELU> gelu_;
  std::unique_ptr<nn::Linear> c_proj_;

  // activation tensors
  nn::Matrix fch_, fch_grad_;            // [B*T, 4*C]
  nn::Matrix fch_gelu_, fch_gelu_grad_;  // [B*T, 4*C]
};

struct CausalSelfAttention {
  CausalSelfAttention(int block_size, int n_head, int n_embed)
      : n_head_(n_head), n_embed_(n_embed) {
    CHECK_EQ(n_embed % n_head, 0);

    // key, query, value projections for all heads, but in a batch
    c_attn_ = std::make_unique<nn::Linear>(n_embed, 3 * n_embed);

    // output projection
    c_proj_ = std::make_unique<nn::Linear>(n_embed, n_embed);

    softmax_ = std::make_unique<nn::Softmax>(true);

    // mask
    bias_ = Eigen::MatrixXi::Ones(block_size, block_size)
                .triangularView<Eigen::Lower>();
  }

  void Forward(const Eigen::TensorMap<nn::Tensor3D>& x,
               Eigen::TensorMap<nn::Tensor3D>& y) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    int NH = n_head_, HS = C / n_head_;
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK(C == n_embed_ && C == y.dimension(2));

    // Lazily allocate the memory for activation
    LAZY_ALLOCATE_TENSOR3D(qkv_, B, T, 3 * C);
    LAZY_ALLOCATE_TENSOR4D(q_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(k_, B, NH, HS, T);
    LAZY_ALLOCATE_TENSOR4D(v_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(preatt_, B, NH, T, T);
    LAZY_ALLOCATE_TENSOR4D(att_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(att2_, B, T, NH, HS);

    auto _x = Eigen::Map<nn::Matrix>(x.data(), B * T, C);
    auto qkv = Eigen::Map<nn::Matrix>(qkv_.data(), B * T, 3 * C);
    c_attn_->Forward(_x, qkv);

    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    Eigen::array<Eigen::Index, 4> shape = {B, T, NH, HS};
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 2, 3, 1};
    q_ = qkv_.slice(offsets_q, extents)  // [B, T, C]
             .reshape(shape)             // [B, T, NH, HS]
             .shuffle(shuffle_qv)        // [B, NH, T, HS]
        ;
    k_ = qkv_.slice(offsets_k, extents)  // [B, T, C]
             .reshape(shape)             //  [B, T, NH, HS]
             .shuffle(shuffle_k)         //  [B, NH, T, HS]
        ;
    v_ = qkv_.slice(offsets_v, extents)  // [B, T, C]
             .reshape(shape)             //  [B, T, NH, HS]
             .shuffle(shuffle_qv)        //  [B, NH, T, HS]
        ;

    const float factor = 1.0f / std::sqrt(static_cast<float>(HS));
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < NH; ++h) {
        auto q2d =
            Eigen::Map<nn::Matrix>(q_.data() + (b * NH + h) * T * HS, T, HS);
        auto k2d =
            Eigen::Map<nn::Matrix>(k_.data() + (b * NH + h) * HS * T, HS, T);
        auto v2d =
            Eigen::Map<nn::Matrix>(v_.data() + (b * NH + h) * T * HS, T, HS);
        auto preatt2d =
            Eigen::Map<nn::Matrix>(preatt_.data() + (b * NH + h) * T * T, T, T);
        auto att2d =
            Eigen::Map<nn::Matrix>(att_.data() + (b * NH + h) * T * HS, T, HS);

        preatt2d.noalias() = (q2d * k2d) * factor;
        for (int i = 0; i < T; ++i) {
          for (int j = 0; j < T; ++j) {
            if (!bias_(i, j)) {
              preatt2d(i, j) = -std::numeric_limits<float>::infinity();
            }
          }
        }

        // softmax
        //        preatt2d = preatt2d.array().exp();
        //        preatt2d =
        //            preatt2d.array().colwise() /
        //            preatt2d.rowwise().sum().array();
        softmax_->Forward(preatt2d, preatt2d);

        // att * v
        att2d = preatt2d * v2d;
      }
    }

    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    att2_ = att_.shuffle(shuffle_att);  // [B, T, NH, HS]
    auto att2_2d = Eigen::Map<nn::Matrix>(att2_.data(), B * T, C);

    auto y2d = Eigen::Map<nn::Matrix>(y.data(), B * T, C);
    c_proj_->Forward(att2_2d, y2d);
  }

  void Backward(const Eigen::TensorMap<nn::Tensor3D>& x,
                const Eigen::TensorMap<nn::Tensor3D>& y,
                const Eigen::TensorMap<nn::Tensor3D>& y_grad,
                Eigen::TensorMap<nn::Tensor3D>& x_grad) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    int NH = n_head_, HS = C / n_head_;
    CHECK(B == y.dimension(0) && B == y_grad.dimension(0) &&
          B == x_grad.dimension(0));
    CHECK(T == y.dimension(1) && T == y_grad.dimension(1) &&
          T == x_grad.dimension(1));
    CHECK(C == y.dimension(2) && C == y_grad.dimension(2) &&
          C == x_grad.dimension(2));

    // Lazily allocate the memory for activation
    LAZY_ALLOCATE_TENSOR3D(qkv_grad_, B, T, 3 * C);
    LAZY_ALLOCATE_TENSOR4D(q_grad_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(k_grad_, B, NH, HS, T);
    LAZY_ALLOCATE_TENSOR4D(v_grad_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(preatt_grad_, B, NH, T, T);
    LAZY_ALLOCATE_TENSOR4D(att_grad_, B, NH, T, HS);
    LAZY_ALLOCATE_TENSOR4D(att2_grad_, B, T, NH, HS);

    // attproj backward
    auto att2_2d = Eigen::Map<nn::Matrix>(att2_.data(), B * T, C);
    auto y_grad_2d = Eigen::Map<nn::Matrix>(y_grad.data(), B * T, C);
    auto att2_grad_2d = Eigen::Map<nn::Matrix>(att2_grad_.data(), B * T, C);
    c_proj_->Backward(att2_2d, y_grad_2d, att2_grad_2d);

    // shuffle backward
    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    att_grad_ = att2_grad_.shuffle(shuffle_att);  // [B, NH, T, HS]

    //
  }

  size_t NumParameters() const {
    return c_attn_->NumParameters() + c_proj_->NumParameters();
  }

  int n_head_;
  int n_embed_;
  std::unique_ptr<nn::Linear> c_attn_;
  std::unique_ptr<nn::Linear> c_proj_;
  std::unique_ptr<nn::Softmax> softmax_;

  // activation tensors
  nn::Tensor3D qkv_, qkv_grad_;        // [B, T, 3C]
  nn::Tensor4D q_, q_grad_;            // [B, NH, T, HS]
  nn::Tensor4D k_, k_grad_;            // [B, NH, HS, T]
  nn::Tensor4D v_, v_grad_;            // [B, NH, T, HS]
  nn::Tensor4D preatt_, preatt_grad_;  // [B, NH, T, T]
  nn::Tensor4D att_, att_grad_;        // [B, NH, T, HS]
  nn::Tensor4D att2_, att2_grad_;      // [B, T, NH, HS]

  // not really a 'bias', more of a mask, but following the OpenAI/HF naming
  // though
  Eigen::MatrixXi bias_;
};

struct Block {
  Block(int block_size, int n_head, int n_embed) {
    ln1_ = std::make_unique<nn::LayerNorm>(n_embed);
    attn_ = std::make_unique<CausalSelfAttention>(block_size, n_head, n_embed);
    ln2_ = std::make_unique<nn::LayerNorm>(n_embed);
    mlp_ = std::make_unique<MLP>(n_embed);
  }

  void Forward(const Eigen::TensorMap<nn::Tensor3D>& x,
               Eigen::TensorMap<nn::Tensor3D>& y) {
    // x: [B * T, C], y: [B * T, C]
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK_EQ(C, y.dimension(2));

    LAZY_ALLOCATE_MATRIX(ln1_y_, B * T, C);
    LAZY_ALLOCATE_VECTOR(ln1_mean_, B * T);
    LAZY_ALLOCATE_VECTOR(ln1_rstd_, B * T);
    LAZY_ALLOCATE_TENSOR3D(att_y_, B, T, C);
    LAZY_ALLOCATE_MATRIX(ln2_y_, B * T, C);
    LAZY_ALLOCATE_VECTOR(ln2_mean_, B * T);
    LAZY_ALLOCATE_VECTOR(ln2_rstd_, B * T);

    // LN1
    auto x_2d = Eigen::Map<nn::Matrix>(x.data(), B * T, C);
    auto ln1_y_2d = Eigen::Map<nn::Matrix>(ln1_y_.data(), B * T, C);
    auto ln1_mean_1d = Eigen::Map<Eigen::RowVectorXf>(ln1_mean_.data(), B * T);
    auto ln1_rstd_1d = Eigen::Map<Eigen::RowVectorXf>(ln1_rstd_.data(), B * T);
    ln1_->Forward(x_2d, ln1_y_2d, ln1_mean_1d, ln1_rstd_1d);

    // Attention
    auto ln1_y_3d = Eigen::TensorMap<nn::Tensor3D>(ln1_y_2d.data(), B, T, C);
    auto att_y_3d = Eigen::TensorMap<nn::Tensor3D>(att_y_.data(), B, T, C);
    attn_->Forward(ln1_y_3d, att_y_3d);

    // Residual
    auto att_y_2d = Eigen::Map<nn::Matrix>(att_y_.data(), B * T, C);
    att_y_2d.array() += x_2d.array();

    // LN2
    auto ln2_y_2d = Eigen::Map<nn::Matrix>(ln2_y_.data(), B * T, C);
    auto ln2_mean_1d = Eigen::Map<Eigen::RowVectorXf>(ln2_mean_.data(), B * T);
    auto ln2_rstd_1d = Eigen::Map<Eigen::RowVectorXf>(ln2_rstd_.data(), B * T);
    ln2_->Forward(att_y_2d, ln2_y_2d, ln2_mean_1d, ln2_rstd_1d);

    // MLP
    auto y_2d = Eigen::Map<nn::Matrix>(y.data(), B * T, C);
    mlp_->Forward(ln2_y_2d, y_2d);

    // Residual
    y_2d.array() += att_y_2d.array();
  }

  size_t NumParameters() const {
    return ln1_->NumParameters() + attn_->NumParameters() +
           ln2_->NumParameters() + mlp_->NumParameters();
  }

  std::unique_ptr<nn::LayerNorm> ln1_;
  std::unique_ptr<CausalSelfAttention> attn_;
  std::unique_ptr<nn::LayerNorm> ln2_;
  std::unique_ptr<MLP> mlp_;

  // activation tensors
  nn::Matrix ln1_y_;                        // [B*T, C]
  Eigen::RowVectorXf ln1_mean_, ln1_rstd_;  // [B*T]
  nn::Tensor3D att_y_;                      // [B, T, C]
  nn::Matrix ln2_y_;                        // [B*T, C]
  Eigen::RowVectorXf ln2_mean_, ln2_rstd_;  // [B*T]
};

struct GPT {
  GPT(int block_size, int vocab_size, int padded_vocab_size, int n_layer,
      int n_head, int n_embed)
      : block_size_(block_size),
        vocab_size_(vocab_size),
        padded_vocab_size_(padded_vocab_size),
        n_layer_(n_layer),
        n_embed_(n_embed) {
    CHECK_GT(n_layer, 0);

    wte_ = std::make_unique<nn::Embedding>(padded_vocab_size, n_embed);
    wpe_ = std::make_unique<nn::Embedding>(block_size, n_embed);
    for (int i = 0; i < n_layer; ++i) {
      h_.emplace_back(std::make_unique<Block>(block_size, n_head, n_embed));
    }
    lnf_ = std::make_unique<nn::LayerNorm>(n_embed);

    lm_head_unused_ = std::make_unique<nn::Linear>(n_embed, vocab_size);
    // https://paperswithcode.com/method/weight-tying
    std::memcpy(wte_->weight_.get(), lm_head_unused_->weight_.data(),
                sizeof(float) * vocab_size * n_embed);
    lm_head_ = wte_->weight_.get();
    cross_entropy_ =
        std::make_unique<nn::SoftmaxCrossEntropy>(nn::SoftmaxCrossEntropy::MEAN, true);
  }

  void Forward(const Eigen::Map<nn::MatrixInt>& idx,
               Eigen::TensorMap<nn::Tensor3D>& logits) {
    const int B = idx.rows(), T = idx.cols(), C = n_embed_;
    const int BT = B * T;
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    DoForward(idx);

    // OPTIMIZE:
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    //    auto lnf_y_3d = Eigen::TensorMap<nn::Tensor3D>(lnf_y_.data(), B, T,
    //    C); nn::Tensor2D lnf_y_last_t = lnf_y_3d.chip(T - 1, 1);
    auto lnf_y = Eigen::Map<nn::Matrix>(lnf_y_.data(), BT, C);
    auto lm_head = Eigen::Map<nn::Matrix>(lm_head_, vocab_size_, C);
    auto logits_2d = Eigen::Map<nn::Matrix>(logits.data(), BT, vocab_size_);
    nn::MatMul::Forward(lnf_y, lm_head, logits_2d);
  }

  void Forward(const Eigen::Map<nn::MatrixInt>& idx,
               const Eigen::Map<nn::MatrixInt>& targets,
               Eigen::TensorMap<nn::Tensor3D>& logits, float* loss) {
    // idx: [B, T], targets: [B, T]
    // logits: [B, T, vocab_size]
    const int B = idx.rows(), T = idx.cols(), C = n_embed_;
    const int BT = B * T;
    CHECK(targets.rows() == B && targets.cols() == T);
    CHECK(logits.dimension(0) == B && logits.dimension(1) == T &&
          logits.dimension(2) == vocab_size_)
        << "B: " << B << ", T: " << T << ", vocab_size: " << vocab_size_;
    DoForward(idx);

    if (probs_.size() < BT * vocab_size_) {
      probs_.resize(BT * vocab_size_);
    }

    auto lnf_y = Eigen::Map<nn::Matrix>(lnf_y_.data(), BT, C);
    auto lm_head = Eigen::Map<nn::Matrix>(lm_head_, vocab_size_, n_embed_);
    auto logits_2d = Eigen::Map<nn::Matrix>(logits.data(), BT, vocab_size_);
    auto probs_2d = Eigen::Map<nn::Matrix>(probs_.data(), BT, vocab_size_);
    nn::MatMul::Forward(lnf_y, lm_head, logits_2d);
    cross_entropy_->Forward(logits_2d, targets, probs_2d, loss);
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

 private:
  void DoForward(const Eigen::Map<nn::MatrixInt>& idx) {
    const int B = idx.rows(), T = idx.cols(), C = n_embed_, L = n_layer_;
    const int BT = B * T, TC = T * C;
    const int BTC = BT * C;
    const int LBTC = L * BTC;

    CHECK_LE(T, block_size_) << "Cannot forward sequence of length " << T
                             << ", block size is only " << block_size_;
    std::vector<int> pos(T);
    std::iota(pos.begin(), pos.end(), 0);

    // Lazily allocate memory
    if (tok_emb_.size() < BTC) {
      tok_emb_.resize(BTC);
    }
    if (pos_emb_.size() < TC) {
      pos_emb_.resize(TC);
    }
    if (block_y_.size() < LBTC) {
      block_y_.resize(LBTC);
    }
    if (lnf_y_.size() < BTC) {
      lnf_y_.resize(BTC);
    }
    if (lnf_mean_.size() < BT) {
      lnf_mean_.resize(BT);
    }
    if (lnf_rstd_.size() < BT) {
      lnf_rstd_.resize(BT);
    }

    wte_->Forward(idx, absl::MakeSpan(tok_emb_));
    wpe_->Forward(pos, absl::MakeSpan(pos_emb_));

    auto tok_w = Eigen::Map<nn::Matrix>(wte_->weight_.get(), vocab_size_, C);
    auto pos_w = Eigen::Map<nn::Matrix>(wpe_->weight_.get(), block_size_, C);
    auto tok_emb = Eigen::Map<nn::Matrix>(tok_emb_.data(), B, TC);
    auto pos_emb = Eigen::Map<Eigen::RowVectorXf>(pos_emb_.data(), TC);
    tok_emb.rowwise() += pos_emb;

    for (int l = 0; l < n_layer_; ++l) {
      const auto& block = h_[l];
      float* x = l == 0 ? tok_emb_.data() : block_y_.data() + (l - 1) * BTC;
      float* y = block_y_.data() + l * BTC;
      auto block_x_3d = Eigen::TensorMap<nn::Tensor3D>(x, B, T, C);
      auto block_y_3d = Eigen::TensorMap<nn::Tensor3D>(y, B, T, C);
      block->Forward(block_x_3d, block_y_3d);
    }

    auto block_out_2d =
        Eigen::Map<nn::Matrix>(block_y_.data() + (L - 1) * BTC, BT, C);
    auto lnf_y = Eigen::Map<nn::Matrix>(lnf_y_.data(), BT, C);
    auto lnf_mean = Eigen::Map<Eigen::RowVectorXf>(lnf_mean_.data(), BT);
    auto lnf_rstd = Eigen::Map<Eigen::RowVectorXf>(lnf_rstd_.data(), BT);
    lnf_->Forward(block_out_2d, lnf_y, lnf_mean, lnf_rstd);
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
  std::unique_ptr<nn::SoftmaxCrossEntropy> cross_entropy_;

  // head
  std::unique_ptr<nn::Linear> lm_head_unused_;
  float* lm_head_;  // [vocal_size, C]

  // activation tensors
  std::vector<float> tok_emb_;              // [B, T, C]
  std::vector<float> pos_emb_;              // [T, C]
  std::vector<float> block_y_;              // [L, B, T, C]
  std::vector<float> lnf_y_;                // [B*T, C]
  std::vector<float> lnf_mean_, lnf_rstd_;  // [B*T]
  std::vector<float> probs_;                // [B*T, vocab_size]
};

}  // namespace gpt

#endif  // LLM_CPP__GPT_HPP_
