#ifndef LLM_CPP__GPT2_HPP_
#define LLM_CPP__GPT2_HPP_

#include "absl/strings/string_view.h"
#include "gpt.hpp"
#include "llmc/utils.h"

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

    // read in all the parameters from file
    freadCheck(gpt2_->wte_->weight_->data(), sizeof(float),
               gpt2_->wte_->NumParameters(), model_file);
    freadCheck(gpt2_->wpe_->weight_->data(), sizeof(float),
               gpt2_->wpe_->NumParameters(), model_file);
    // ln1w
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->ln1_->weight_->data();
      int num = block->ln1_->normalized_shape_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // ln1b
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->ln1_->bias_->data();
      int num = block->ln1_->normalized_shape_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // qkvw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->attn_->c_attn_->weight_->data();
      int num = block->attn_->c_attn_->out_features_ *
                block->attn_->c_attn_->in_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // qkvb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->attn_->c_attn_->bias_->data();
      int num = block->attn_->c_attn_->out_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // attprojw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->attn_->c_proj_->weight_->data();
      int num = block->attn_->c_proj_->out_features_ *
                block->attn_->c_proj_->in_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // attprojb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->attn_->c_proj_->bias_->data();
      int num = block->attn_->c_proj_->out_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // ln2w
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->ln2_->weight_->data();
      int num = block->ln2_->normalized_shape_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // ln2b
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->ln2_->bias_->data();
      int num = block->ln2_->normalized_shape_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // fcw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->mlp_->c_fc_->weight_->data();
      int num =
          block->mlp_->c_fc_->out_features_ * block->mlp_->c_fc_->in_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // fcb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->mlp_->c_fc_->bias_->data();
      int num = block->mlp_->c_fc_->out_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // fcprojw
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->mlp_->c_proj_->weight_->data();
      int num = block->mlp_->c_proj_->out_features_ *
                block->mlp_->c_proj_->in_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // fcprojb
    for (int l = 0; l < L; ++l) {
      const auto& block = gpt2_->h_[l];
      float* p = block->mlp_->c_proj_->bias_->data();
      int num = block->mlp_->c_proj_->out_features_;
      freadCheck(p, sizeof(float), num, model_file);
    }

    // lnfw
    freadCheck(gpt2_->lnf_->weight_->data(), sizeof(float),
               gpt2_->lnf_->normalized_shape_, model_file);
    // lnfb
    freadCheck(gpt2_->lnf_->bias_->data(), sizeof(float),
               gpt2_->lnf_->normalized_shape_, model_file);

    fcloseCheck(model_file);
  }

  GPT2Config config;
  std::unique_ptr<gpt::GPT> gpt2_;
};

#endif  // LLM_CPP__GPT2_HPP_
