#include <unistd.h>
#include <iostream>
#include <memory>

#include "gpt2.hpp"
#include "optim.hpp"

#include "Eigen/Core"
#include "absl/types/span.h"

float* restore_tensor(const std::string& name, int* num) {
  FILE* fp = fopen(name.c_str(), "rb");
  fread(num, sizeof(int), 1, fp);

  float* data = new float[*num];
  fread(data, sizeof(float), *num, fp);
  fclose(fp);
  return data;
}

void diff_tensor(absl::Span<const float> value, float* gt, int len,
                 const char* desc) {
  CHECK_EQ(value.size(), len);
  int diff_count5 = 0, diff_count6 = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    float val = value[i], val_gt = gt[i];
    if (std::abs(val - val_gt) > 1e-5) {
      if (diff_count5 < 10) {
        fprintf(stdout, "--- diff5(%s): %d(%d) %.6f %.6f\n", desc, i, len, val,
                val_gt);
      }
      diff_count5++;
    }

    if (std::abs(val - val_gt) > 1e-6) {
      if (diff_count6 < 10) {
        fprintf(stdout, "--- diff6(%s): %d(%d) %.6f %.6f\n", desc, i, len, val,
                val_gt);
      }
      diff_count6++;
    }
  }
  fprintf(stdout, "--- diff5 count(%s): %d\n", desc, diff_count5);
  fprintf(stdout, "--- diff6 count(%s): %d\n", desc, diff_count6);
}

void diff_tensor(nn::Parameter* p, float* gt, int len, const char* desc) {
  diff_tensor(p->View(nn::Parameter::kGrad), gt, len, desc);
}

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
  // let's do 10 training iterations, following the pytorch code
  float expected_losses[10] = {5.270007133483887,  4.059706687927246,
                               3.3751230239868164, 2.8007826805114746,
                               2.315382242202759,  1.8490285873413086,
                               1.3946564197540283, 0.9991465210914612,
                               0.6240804195404053, 0.37651097774505615};

  auto idx = Eigen::Map<nn::MatrixInt>(x.get(), B, T);
  auto target = Eigen::Map<nn::MatrixInt>(y.get(), B, T);
  auto logit_3d =
      Eigen::TensorMap<nn::Tensor3D>(calculated_logits.get(), B, T, V);
  std::vector<nn::Parameter*> parameters;
  model.Parameters(&parameters);
  optim::AdamW optimizer(parameters, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f);

  for (int step = 0; step < 10; step++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    model.gpt2_->Forward(idx, target, logit_3d, &loss);
    optimizer.ZeroGrad();
    model.gpt2_->Backward(idx, target);

    /*
    int num_restore = 0;
    auto wte = restore_tensor("wte.dat", &num_restore);
    auto wte_span = model.gpt2_->wte_->weight_->View();
    diff_tensor(wte_span, wte, num_restore, "wte");
    auto logit_grad = restore_tensor("logit_grad.dat", &num_restore);
    diff_tensor(model.gpt2_->logits_grad_, logit_grad, num_restore,
                "logit_grad");
    auto lnf = restore_tensor("lnf.dat", &num_restore);
    diff_tensor(model.gpt2_->lnf_y_grad_, lnf, num_restore, "lnf_y_grad");
    auto lnf_mean = restore_tensor("lnf_mean.dat", &num_restore);
    diff_tensor(model.gpt2_->lnf_mean_, lnf_mean, num_restore, "lnf_mean");
    auto lnf_rstd = restore_tensor("lnf_rstd.dat", &num_restore);
    diff_tensor(model.gpt2_->lnf_rstd_, lnf_rstd, num_restore, "lnf_rstd");

    auto lnfb = restore_tensor("lnfb.dat", &num_restore);
    diff_tensor(model.gpt2_->lnf_->bias_.get(), lnfb, num_restore, "lnfb");
    */

    optimizer.Step(step + 1);
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

    float expected_loss = expected_losses[step];
    int step_loss_ok = fabsf(expected_loss - loss) < 1e-2;
    allok = allok && step_loss_ok;
    // print the timing information at the end
    printf("step %d: loss %f (took %f ms) OK = %d\n", step, loss,
           time_elapsed_s * 1000, step_loss_ok);
  }

  return 0;
}