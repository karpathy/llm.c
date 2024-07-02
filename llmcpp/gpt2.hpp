#ifndef LLM_CPP__GPT2_HPP_
#define LLM_CPP__GPT2_HPP_

#include "absl/strings/string_view.h"
#include "gpt.hpp"
#include "llmc/utils.h"

namespace gpt2 {
struct GPT2Config {
  int max_seq_len;        // max sequence length, e.g. 1024
  int vocab_size;         // vocab size, e.g. 50257
  int padded_vocab_size;  // padded to e.g. %128==0, 50304
  int num_layers;         // number of layers, e.g. 12
  int num_heads;          // number of heads in attention, e.g. 12
  int channels;           // number of channels, e.g. 768
};

struct GPT2 {
  void BuildFromCheckpoint(absl::string_view checkpoint_path) {
    // read in model from a checkpoint file
    FILE* model_file = fopenCheck(checkpoint_path.data(), "rb");
    if (model_file == nullptr) {
      printf("Error opening model file\n");
      exit(1);
    }

    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
      printf("Bad magic model file\n");
      exit(1);
    }
    if (model_header[1] != 3) {
      printf("Bad version in model file\n");
      printf("---> HINT: try to re-run `python train_gpt2.py`\n");
      exit(1);
    }

    // read in hyperparameters
    int maxT, V, Vp, L, NH, C;
    config.max_seq_len = maxT = model_header[2];
    config.vocab_size = V = model_header[3];
    config.num_layers = L = model_header[4];
    config.num_heads = NH = model_header[5];
    config.channels = C = model_header[6];
    config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    gpt2_ = std::make_unique<gpt::GPT>(
        config.max_seq_len, config.vocab_size, config.padded_vocab_size,
        config.num_layers, config.num_heads, config.channels);
    // allocate space for all the parameters and read them in
    printf("num_parameters: %zu\n", gpt2_->NumParameters());

    auto restore_fn = [&](nn::Parameter* p, const std::string& name) {
      freadCheck(p->data(), sizeof(float), p->size(), model_file);
    };
    ApplyFn(restore_fn, L);
    fcloseCheck(model_file);
  }

  void Parameters(std::vector<nn::Parameter*>* parameters) const {
    auto collect_fn = [&](nn::Parameter* p, const std::string& name) {
      parameters->push_back(p);
    };
    ApplyFn(collect_fn, config.num_layers);
  }

  void Parameters(std::unordered_map<std::string , nn::Parameter*>* parameters) const {
    auto collect_fn = [&](nn::Parameter* p, const std::string& name) {
      parameters->insert({name, p});
    };
    ApplyFn(collect_fn, config.num_layers);
  }

  void ApplyFn(
      const std::function<void(nn::Parameter*, const std::string&)>& apply_fn,
      int L) const {
    apply_fn(gpt2_->wte_->weight_.get(), "wte");
    apply_fn(gpt2_->wpe_->weight_.get(), "wpe");

    auto name_with_layer = [](const std::string& name, int l) {
      return name + "-L" + std::to_string(l);
    };

    // ln1w
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->ln1_->weight_.get();
      apply_fn(p, name_with_layer("ln1w", l));
    }

    // ln1b
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->ln1_->bias_.get();
      apply_fn(p, name_with_layer("ln1b", l));
    }

    // qkvw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->attn_->c_attn_->weight_.get();
      apply_fn(p, name_with_layer("qkvw", l));
    }

    // qkvb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->attn_->c_attn_->bias_.get();
      apply_fn(p, name_with_layer("qkvb", l));
    }

    // attprojw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->attn_->c_proj_->weight_.get();
      apply_fn(p, name_with_layer("attprojw", l));
    }

    // attprojb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->attn_->c_proj_->bias_.get();
      apply_fn(p, name_with_layer("attprojb", l));
    }

    // ln2w
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->ln2_->weight_.get();
      apply_fn(p, name_with_layer("ln2w", l));
    }

    // ln2b
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->ln2_->bias_.get();
      apply_fn(p, name_with_layer("ln2b", l));
    }

    // fcw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->mlp_->c_fc_->weight_.get();
      apply_fn(p, name_with_layer("fcw", l));
    }

    // fcb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->mlp_->c_fc_->bias_.get();
      apply_fn(p, name_with_layer("fcb", l));
    }

    // fcprojw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->mlp_->c_proj_->weight_.get();
      apply_fn(p, name_with_layer("fcprojw", l));
    }

    // fcprojb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      nn::Parameter* p = block->mlp_->c_proj_->bias_.get();
      apply_fn(p, name_with_layer("fcprojb", l));
    }

    // lnfw
    apply_fn(gpt2_->lnf_->weight_.get(), "lnfw");
    // lnfb
    apply_fn(gpt2_->lnf_->bias_.get(), "lnfb");
  }

  GPT2Config config;
  std::unique_ptr<gpt::GPT> gpt2_;
};
}  // namespace gpt2

#endif  // LLM_CPP__GPT2_HPP_
