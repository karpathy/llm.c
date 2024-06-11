#include <unistd.h>
#include <iostream>
#include <memory>

#include "gpt2.hpp"

#include "Eigen/Core"
#include "absl/types/span.h"
#include "glog/logging.h"

int main(int argc, char** argv) {
  GPT2 model;
  model.BuildFromCheckpoint("gpt2_124M.bin");

  int C = model.config.channels;
  int V = model.config.vocab_size;
  int Vp = model.config.padded_vocab_size;
  int maxT = model.config.max_seq_len;
  int L = model.config.num_layers;

  // load additional information that we will use for debugging and error
  // checking
  FILE* state_file = fopen("gpt2_124M_debug_state.bin", "rb");
  if (state_file == nullptr) {
    printf("Error opening state file\n");
    return 1;
  }
  int state_header[256];
  fread(state_header, sizeof(int), 256, state_file);
  if (state_header[0] != 20240327) {
    printf("Bad magic state file\n");
    return 1;
  }
  if (state_header[1] != 2) {
    printf("Bad version in state file\n");
    printf("---> HINT: try to re-run `python train_gpt2.py`\n");
    return 1;
  }
  int B = state_header[2];  // batch size, e.g. 4
  int T = state_header[3];  // time / sequence length (e.g. 64, up to maxT)
  printf("[State]\n");
  printf("batch_size: %d\n", B);
  printf("seq_len: %d\n", T);

  // inputs and expected outputs, only used for error checking
  auto x = std::make_unique<int[]>(B * T);
  auto y = std::make_unique<int[]>(B * T);
  auto calculated_logits = std::make_unique<float[]>(B * T * V);
  float loss = 0.0;
  auto expected_logits = std::make_unique<float[]>(B * T * V);
  float expected_loss = 0.0;

  // read reference information from Python
  freadCheck(x.get(), sizeof(int), B * T, state_file);
  freadCheck(y.get(), sizeof(int), B * T, state_file);
  freadCheck(expected_logits.get(), sizeof(float), B * T * V, state_file);
  freadCheck(&expected_loss, sizeof(float), 1, state_file);
  fcloseCheck(state_file);

  // overall OK signal for the test
  int allok = 1;
  auto idx = Eigen::Map<nn::MatrixInt>(x.get(), B, T);
  auto target = Eigen::Map<nn::MatrixInt>(y.get(), B, T);
  auto logit_3d =
      Eigen::TensorMap<nn::Tensor3D>(calculated_logits.get(), B, T, V);

  for (int step = 0; step < 1; step++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    model.gpt2_->Forward(idx, target, logit_3d, &loss);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (step == 0) {
      // error checking at step 0 for reference activations/gradients
      // at this point, target should be equal to expected_logits, let's compare
      int logits_ok = 1;
      float max_diff = 0.0f;
      for (int bt = 0; bt < B * T; bt++) {
        for (int v = 0; v < V; v++) {
          int i = bt * V + v;
          if (i < 10) {
            printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
          }
          float diff =
              fabsf(expected_logits[bt * V + v] - calculated_logits[i]);
          max_diff = fmaxf(max_diff, diff);
          if (diff >= 1e-2f) {
            printf("MISMATCH AT INDEX %d,%d: ", bt, v);
            printf("%f %f\n", expected_logits[bt * V + v],
                   calculated_logits[i]);
            logits_ok = 0;
            bt = B * T;  // to break out of both loops
            break;
          }
        }
      }
      if (!logits_ok) {
        printf("NOT ");
      }
      printf("OK (LOGITS), max_diff = %e\n", max_diff);
      allok = allok && logits_ok;

      // compare the achieved loss
      if (fabsf(loss - expected_loss) >= 1e-2) {
        printf("LOSS MISMATCH: %f %f\n", loss, expected_loss);
        allok = 0;
      } else {
        printf("LOSS OK: %f %f\n", loss, expected_loss);
      }
    }
  }

  return 0;
}