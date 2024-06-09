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

#define LAZY_ALLOCATE_TENSOR(t, X, Y, Z)                                      \
  do {                                                                        \
    if (t.size() == 0) {                                                      \
      t.resize(X, Y, Z);                                                      \
    }                                                                         \
    CHECK(t.dimension(0) == X && t.dimension(1) == Y && t.dimension(2) == Z); \
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

    // mask
    bias_ = Eigen::MatrixXi::Ones(block_size, block_size)
                .triangularView<Eigen::Lower>();
  }

  void Forward(const Eigen::TensorMap<nn::Tensor3D>& x,
               Eigen::TensorMap<nn::Tensor3D>& y) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y.dimension(0));
    CHECK_EQ(T, y.dimension(1));
    CHECK(C == n_embed_ && C == y.dimension(2));

    // Lazily allocate the memory for activation
    LAZY_ALLOCATE_TENSOR(qkv_, B, T, 3 * C);

    auto _x = Eigen::Map<nn::Matrix>(x.data(), B * T, C);
    auto qkv = Eigen::Map<nn::Matrix>(qkv_.data(), B * T, 3 * C);
    c_attn_->Forward(_x, qkv);

    Eigen::array<Eigen::Index, 3> offsets_q = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> offsets_k = {0, 0, C};
    Eigen::array<Eigen::Index, 3> offsets_v = {0, 0, 2 * C};
    Eigen::array<Eigen::Index, 3> extents = {B, T, C};
    nn::Tensor3D q3d = qkv_.slice(offsets_q, extents);  // [B, T, C]
    nn::Tensor3D k3d = qkv_.slice(offsets_k, extents);  // [B, T, C]
    nn::Tensor3D v3d = qkv_.slice(offsets_v, extents);  // [B, T, C]

    int nh = n_head_, hs = C / n_head_;
    const float factor = 1.0f / std::sqrt(static_cast<float>(hs));
    auto q4d = Eigen::TensorMap<nn::Tensor4D>(q3d.data(), B, T, nh, hs);
    auto k4d = Eigen::TensorMap<nn::Tensor4D>(k3d.data(), B, T, nh, hs);
    auto v4d = Eigen::TensorMap<nn::Tensor4D>(v3d.data(), B, T, nh, hs);
    Eigen::array<Eigen::Index, 4> shuffle_qv = {0, 2, 1, 3},
                                  shuffle_k = {0, 2, 3, 1};
    nn::Tensor4D q = q4d.shuffle(shuffle_qv);  // (B, nh, T, hs)
    nn::Tensor4D k = k4d.shuffle(shuffle_k);   // (B, nh, hs, T)
    nn::Tensor4D v = v4d.shuffle(shuffle_qv);  // (B, nh, T, hs)
    nn::Tensor4D att(B, nh, T, T), out(B, nh, T, hs);
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < nh; ++h) {
        auto q2d =
            Eigen::Map<nn::Matrix>(q.data() + (b * nh + h) * T * hs, T, hs);
        auto k2d =
            Eigen::Map<nn::Matrix>(k.data() + (b * nh + h) * hs * T, hs, T);
        auto v2d =
            Eigen::Map<nn::Matrix>(v.data() + (b * nh + h) * T * hs, T, hs);
        auto att2d =
            Eigen::Map<nn::Matrix>(att.data() + (b * nh + h) * T * T, T, T);
        auto out2d =
            Eigen::Map<nn::Matrix>(out.data() + (b * nh + h) * T * hs, T, hs);

        att2d.noalias() = (q2d * k2d) * factor;
        for (int i = 0; i < T; ++i) {
          for (int j = 0; j < T; ++j) {
            if (!bias_(i, j)) {
              att2d(i, j) = -std::numeric_limits<float>::infinity();
            }
          }
        }

        // softmax
        att2d = att2d.array().exp();
        att2d = att2d.array().colwise() / att2d.rowwise().sum().array();

        // att * v
        out2d = att2d * v2d;
      }
    }

    Eigen::array<Eigen::Index, 4> shuffle_att = {0, 2, 1, 3};
    auto y4d = Eigen::TensorMap<nn::Tensor4D>(y.data(), B, T, nh, hs);
    y4d = out.shuffle(shuffle_qv);  // (B, T, nh, hs)

    auto y2d = Eigen::Map<nn::Matrix>(y4d.data(), B * T, C);
    c_proj_->Forward(y2d, y2d);
  }

  void Backward(const Eigen::TensorMap<nn::Tensor3D>& x,
                const Eigen::TensorMap<nn::Tensor3D>& y_grad,
                Eigen::TensorMap<nn::Tensor3D>& x_grad) {
    const int B = x.dimension(0);  // batch size
    const int T = x.dimension(1);  // sequence length
    const int C = x.dimension(2);  // embedding dimensionality (n_embd)
    CHECK_EQ(B, y_grad.dimension(0));
    CHECK_EQ(T, y_grad.dimension(1));
    CHECK_EQ(C, y_grad.dimension(2));
    CHECK_EQ(B, x_grad.dimension(0));
    CHECK_EQ(T, x_grad.dimension(1));
    CHECK_EQ(C, x_grad.dimension(2));

    // TODO:
  }

  int n_head_;
  int n_embed_;
  std::unique_ptr<nn::Linear> c_attn_;
  std::unique_ptr<nn::Linear> c_proj_;

  // activation tensors
  nn::Tensor3D qkv_;

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
    LAZY_ALLOCATE_TENSOR(att_y_, B, T, C);
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
  GPT(int block_size, int vocab_size, int n_layer, int n_head, int n_embed)
      : block_size_(block_size),
        vocab_size_(vocab_size),
        n_layer_(n_layer),
        n_embed_(n_embed) {
    CHECK_GT(n_layer, 0);
    wte_ = std::make_unique<nn::Embedding>(vocab_size, n_embed);
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
  }

  void Forward(const Eigen::Map<nn::MatrixInt>& idx,
               Eigen::Map<nn::Matrix>& logits) {
    const int B = idx.rows(), T = idx.cols(), C = n_embed_;
    CHECK_EQ(logits.rows(), B*T);
    CHECK_EQ(logits.cols(), vocab_size_);
    DoForward(idx);


    // OPTIMIZE:
    // inference-time mini-optimization: only forward the lm_head on the very
    // last position
    //    auto lnf_y_3d = Eigen::TensorMap<nn::Tensor3D>(lnf_y_.data(), B, T,
    //    C); nn::Tensor2D lnf_y_last_t = lnf_y_3d.chip(T - 1, 1);
    auto lnf_y = Eigen::Map<nn::Matrix>(lnf_y_.data(), B * T, C);
    auto lm_head = Eigen::Map<nn::Matrix>(lm_head_, vocab_size_, C);

    nn::MatMul::Forward(lnf_y, lm_head, logits);
    std::cout << "logits:\n" << logits << std::endl;
  }

  void Forward(const Eigen::Map<nn::MatrixInt>& idx,
               const Eigen::Map<nn::MatrixInt>& targets,
               Eigen::Map<nn::Matrix>& logits, float* loss) {
    const int B = idx.rows(), T = idx.cols(), C = n_embed_;
    const int BT = B * T;
    CHECK(targets.rows() == B && targets.cols() == T);
    DoForward(idx);

    auto lnf_y = Eigen::Map<nn::Matrix>(lnf_y_.data(), BT, C);
    auto lm_head = Eigen::Map<nn::Matrix>(lm_head_, vocab_size_, n_embed_);
    nn::MatMul::Forward(lnf_y, lm_head, logits);
    std::cout << "logits:\n" << logits << std::endl;
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
    std::cout << "tok_emb:\n" << tok_emb << std::endl;
    std::cout << "pos_emb:\n" << pos_emb << std::endl;
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
    std::cout << "LNF:\n" << lnf_y << std::endl;

    //    auto lm_head = Eigen::Map<nn::Matrix>(lm_head_, vocab_size_, C);
    //    nn::MatMul::Forward(lnf_y, lm_head, logits);
    //    std::cout << "logits:\n" << logits << std::endl;
  }

 public:
  int block_size_;
  int vocab_size_;
  int n_layer_;
  int n_embed_;

  // transformer
  std::unique_ptr<nn::Embedding> wte_;
  std::unique_ptr<nn::Embedding> wpe_;
  std::vector<std::unique_ptr<Block>> h_;
  std::unique_ptr<nn::LayerNorm> lnf_;

  // head
  std::unique_ptr<nn::Linear> lm_head_unused_;
  float* lm_head_;  // [vocal_size, C]

  // activation tensors
  std::vector<float> tok_emb_;              // [B, T, C]
  std::vector<float> pos_emb_;              // [T, C]
  std::vector<float> block_y_;              // [L, B, T, C]
  std::vector<float> lnf_y_;                // [B*T, C]
  std::vector<float> lnf_mean_, lnf_rstd_;  // [B*T]
};

}  // namespace gpt2

#endif  // LLM_CPP__GPT_HPP_
