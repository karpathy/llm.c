#include <unistd.h>
#include <iostream>
#include <memory>

#include "Eigen/Core"
#include "absl/types/span.h"
#include "gpt2.hpp"
#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"
#include "llmc/utils.h"
#include "optim.hpp"

// sampler

unsigned int random_u32(unsigned long long* state) {
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long* state) {  // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1;  // in case of rounding errors
}

int main(int argc, char** argv) {
  gpt2::GPT2<float> model;
  model.BuildFromCheckpoint("gpt2_124M.bin");

  // build the DataLoaders from tokens files. for now use tiny_shakespeare if
  // available, else tiny_stories
  const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
  const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
  const char* tiny_shakespeare_train =
      "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
  const char* tiny_shakespeare_val =
      "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
  const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1
                                 ? tiny_shakespeare_train
                                 : tiny_stories_train;
  const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1
                               ? tiny_shakespeare_val
                               : tiny_stories_val;
  int B = 4;   // batch size 4 (i.e. 4 independent token sequences will be
               // trained on)
  int T = 64;  // sequence length 64 (i.e. each sequence is 64 tokens long).
               // must be <= maxT, which is 1024 for GPT-2
  DataLoader train_loader, val_loader;
  dataloader_init(&train_loader, train_tokens, B, T, 0, 1);
  dataloader_init(&val_loader, val_tokens, B, T, 0, 1);
  printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B * T));
  printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B * T));
  int val_num_batches = 5;

  // build the Tokenizer
  Tokenizer tokenizer;
  tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

  // some memory for generating samples from the model
  unsigned long long rng_state = 1337;
  int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
  const int genT = 64;  // number of steps of inference we will do

  // train
  struct timespec start, end;
  int V = model.config.vocab_size;
  std::unique_ptr<float[]> logit = std::make_unique<float[]>(B * T * V);
  std::unique_ptr<float[]> prob = std::make_unique<float[]>(B * T * V);
  nn::Softmax<float> softmax;
  std::vector<nn::Parameter*> parameters;
  model.Parameters(&parameters);
  optim::AdamW optimizer(parameters, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f);
  for (int step = 0; step <= 10; step++) {
    // once in a while estimate the validation loss
    if (step % 10 == 0) {
      float val_loss = 0.0f;
      dataloader_reset(&val_loader);
      for (int i = 0; i < val_num_batches; i++) {
        dataloader_next_batch(&val_loader);
        float loss = 0.0f;
        auto idx = TTypes<int>::ConstMatrix(val_loader.inputs, B, T);
        auto target = TTypes<int>::ConstMatrix(val_loader.targets, B, T);
        auto logit_3d = Make3DTensor(logit.get(), B, T, V);
        model.gpt2_->Forward(idx, target, logit_3d, &loss);
        val_loss += loss;
      }
      val_loss /= val_num_batches;
      printf("val loss %f\n", val_loss);
    }

    // once in a while do model inference to print generated text
    if (step > 0 && step % 20 == 0) {
      // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
      for (int i = 0; i < B * T; ++i) {
        gen_tokens[i] = tokenizer.eot_token;
      }
      // now sample from the model autoregressively
      printf("generating:\n---\n");
      for (int t = 1; t < genT; t++) {
        // note that inference is very wasteful here because for each token
        // we re-calculate the forward pass for all of (B,T) positions from
        // scratch but the inference here is just for sanity checking anyway and
        // we can maybe optimize a bit more later, with careful tests
        auto gen_tokens_2d = TTypes<int>::ConstMatrix(gen_tokens, B, T);
        auto logit_3d = Make3DTensor(logit.get(), B, T, V);
        model.gpt2_->Forward(gen_tokens_2d, logit_3d);
        auto logit_2d = MakeConstMatrix(logit.get(), B * T, V);
        auto prob_2d = MakeMatrix(prob.get(), B * T, V);
        softmax.Forward(logit_2d, prob_2d);
        // furthermore, below we're only using b=0 (i.e. the first row) of all B
        // rows we're in principle running B "inference streams" in parallel
        // here but only using position 0 get the Vp-dimensional vector probs[0,
        // t-1, :]
        float* probs = prob.get() + (t - 1) * V;
        float coin = random_f32(&rng_state);
        // note we're only sampling from the first V elements, ignoring padding
        // (the probabilities in the padded region should be zero anyway)
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        // print the generated token, either using the Tokenizer or a fallback
        if (tokenizer.init_ok) {
          const char* token_str = tokenizer_decode(&tokenizer, next_token);
          safe_printf(token_str);
        } else {
          // fall back to printing the token id
          printf("%d ", next_token);
        }
        fflush(stdout);
      }
      printf("\n---\n");
    }

    // do a training step
    clock_gettime(CLOCK_MONOTONIC, &start);
    dataloader_next_batch(&train_loader);
    float loss = 0.0f;
    auto idx = TTypes<int>::ConstMatrix(train_loader.inputs, B, T);
    auto target = TTypes<int>::ConstMatrix(train_loader.targets, B, T);
    auto logit_3d = Make3DTensor(logit.get(), B, T, V);
    model.gpt2_->Forward(idx, target, logit_3d, &loss);
    optimizer.ZeroGrad();
    model.gpt2_->Backward(idx, target);
    optimizer.Step(step + 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("step %d: train loss %f (took %f ms)\n", step, loss,
           time_elapsed_s * 1000);
  }

  // free
  dataloader_free(&train_loader);
  dataloader_free(&val_loader);
  tokenizer_free(&tokenizer);
  free(gen_tokens);
  return 0;
}
