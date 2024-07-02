#define TESTING
#include <unistd.h>
#include <iostream>
#include <memory>

#include "gpt2.hpp"
#include "optim.hpp"

#include "Eigen/Core"
#include "absl/types/span.h"
#include "train_gpt2.c"

// poor man's tensor checker
int check_tensor(float* a, float* b, int n, const char* label) {
  int print_upto = 5;
  int ok = 1;
  float maxdiff = 0.0f;
  float tol = 2e-2;
  printf("%s\n", label);
  for (int i = 0; i < n; i++) {
    // look at the diffence at position i of these two tensors
    float diff = fabsf(a[i] - b[i]);

    // keep track of the overall error
    ok = ok && (diff <= tol);
    if (diff > maxdiff) {
      maxdiff = diff;
    }

    // for the first few elements of each tensor, pretty print
    // the actual numbers, so we can do a visual, qualitative proof/assessment
    if (i < print_upto) {
      if (diff <= tol) {
        if (i < print_upto) {
          printf("OK ");
        }
      } else {
        if (i < print_upto) {
          printf("NOT OK ");
        }
      }
      printf("%f %f\n", a[i], b[i]);
    }
  }
  // print the final result for this tensor
  if (ok) {
    printf("TENSOR OK, maxdiff = %e\n", maxdiff);
  } else {
    printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
  }
  return ok;
}

int main(int argc, char** argv) {
  Eigen::setNbThreads(4);
  GPT2 model;
  gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

  gpt2::GPT2 model_cpp;
  model_cpp.BuildFromCheckpoint("gpt2_124M.bin");

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

  ParameterTensors expected_grads;
  float* expected_grads_memory =
      malloc_and_point_parameters(&expected_grads, model.param_sizes);

  // inputs and expected outputs, only used for error checking
  int* x = (int*)malloc(B * T * sizeof(int));
  int* y = (int*)malloc(B * T * sizeof(int));
  float* expected_logits = (float*)malloc(B * T * V * sizeof(float));
  float* expected_loss = (float*)malloc(1 * sizeof(float));

  // read reference information from Python
  fread(x, sizeof(int), B * T, state_file);
  fread(y, sizeof(int), B * T, state_file);
  fread(expected_logits, sizeof(float), B * T * V, state_file);
  fread(expected_loss, sizeof(float), 1, state_file);
  fread(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
  fclose(state_file);

  // overall OK signal for the test
  int allok = 1;
  // let's do 10 training iterations, following the pytorch code
  float expected_losses[10] = {5.270007133483887,  4.059706687927246,
                               3.3751230239868164, 2.8007826805114746,
                               2.315382242202759,  1.8490285873413086,
                               1.3946564197540283, 0.9991465210914612,
                               0.6240804195404053, 0.37651097774505615};

  auto idx = Eigen::Map<nn::MatrixInt>(x, B, T);
  auto target = Eigen::Map<nn::MatrixInt>(y, B, T);
  float* calculated_logits = (float*)malloc(B * T * V * sizeof(float));
  auto logit_3d = Eigen::TensorMap<nn::Tensor3D>(calculated_logits, B, T, V);
  std::vector<nn::Parameter*> parameters;
  model_cpp.Parameters(&parameters);
  size_t num_parameters = 0;
  for (const auto& p : parameters) {
    num_parameters += p->size();
  }
  CHECK_EQ(num_parameters, model.num_parameters);
  optim::AdamW optimizer(parameters, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f);
  float loss = 0;

  for (int step = 0; step < 10; step++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    model_cpp.gpt2_->Forward(idx, target, logit_3d, &loss);
    optimizer.ZeroGrad();
    model_cpp.gpt2_->Backward(idx, target);

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
      if (fabsf(loss - *expected_loss) >= 1e-2) {
        printf("LOSS MISMATCH: %f %f\n", loss, expected_loss);
        allok = 0;
      } else {
        printf("LOSS OK: %f %f\n", loss, expected_loss);
      }

      std::unordered_map<std::string, nn::Parameter*> name_to_parameter;
      model_cpp.Parameters(&name_to_parameter);
      auto get_concated_grad = [&name_to_parameter, L](const std::string& name,
                                                       int len) {
        CHECK_EQ(len % L, 0);
        int C = len / L;
        float* concated = new float[len];  // memory leak, but we don't care
        int total = 0;
        for (int l = 0; l < L; ++l) {
          auto key = name + "-L" + std::to_string(l);
          auto grad = name_to_parameter[key]->grad();
          int num = name_to_parameter[key]->size();
          CHECK_EQ(C, num);
          std::memcpy(concated + l * C, grad, sizeof(float) * C);
          total += num;
        }
        CHECK_EQ(total, len);
        return concated;
      };
      // finally check all the gradients
      int gradoks[16];
      gradoks[0] = check_tensor(model_cpp.gpt2_->wte_->weight_->grad(),
                                expected_grads.wte, V * C, "dwte");
      gradoks[1] = check_tensor(model_cpp.gpt2_->wpe_->weight_->grad(),
                                expected_grads.wpe, maxT * C, "dwpe");
      auto ln1w = get_concated_grad("ln1w", L * C);
      gradoks[2] = check_tensor(ln1w, expected_grads.ln1w, L * C, "dln1w");
      auto ln1b = get_concated_grad("ln1b", L * C);
      gradoks[3] = check_tensor(ln1b, expected_grads.ln1b, L * C, "dln1b");
      auto qkvw = get_concated_grad("qkvw", L * 3 * C * C);
      gradoks[4] =
          check_tensor(qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw");
      auto qkvb = get_concated_grad("qkvb", L * 3 * C);
      gradoks[5] = check_tensor(qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb");
      auto attprojw = get_concated_grad("attprojw", L * C * C);
      gradoks[6] = check_tensor(attprojw, expected_grads.attprojw, L * C * C,
                                "dattprojw");
      auto attprojb = get_concated_grad("attprojb", L * C);
      gradoks[7] =
          check_tensor(attprojb, expected_grads.attprojb, L * C, "dattprojb");
      auto ln2w = get_concated_grad("ln2w", L * C);
      gradoks[8] = check_tensor(ln2w, expected_grads.ln2w, L * C, "dln2w");
      auto ln2b = get_concated_grad("ln2b", L * C);
      gradoks[9] = check_tensor(ln2b, expected_grads.ln2b, L * C, "dln2b");
      auto fcw = get_concated_grad("fcw", L * 4 * C * C);
      gradoks[10] =
          check_tensor(fcw, expected_grads.fcw, L * 4 * C * C, "dfcw");
      auto fcb = get_concated_grad("fcb", L * 4 * C);
      gradoks[11] = check_tensor(fcb, expected_grads.fcb, L * 4 * C, "dfcb");
      auto fcprojw = get_concated_grad("fcprojw", L * C * 4 * C);
      gradoks[12] = check_tensor(fcprojw, expected_grads.fcprojw, L * C * 4 * C,
                                 "dfcprojw");

      auto fcprojb = get_concated_grad("fcprojb", L * C);
      gradoks[13] =
          check_tensor(fcprojb, expected_grads.fcprojb, L * C, "dfcprojb");
      gradoks[14] = check_tensor(model_cpp.gpt2_->lnf_->weight_->grad(),
                                 expected_grads.lnfw, C, "dlnfw");
      gradoks[15] = check_tensor(model_cpp.gpt2_->lnf_->bias_->grad(),
                                 expected_grads.lnfb, C, "dlnfb");
      for (int i = 0; i < 16; i++) {
        allok = allok && gradoks[i];
      }
    }

    optimizer.Step(step + 1);

    // compare the losses
    float expected_loss = expected_losses[step];
    int step_loss_ok = fabsf(expected_loss - loss) < 1e-2;
    allok = allok && step_loss_ok;
    // print the timing information at the end
    printf("step %d: loss %f (took %f ms) OK = %d\n", step, loss,
           time_elapsed_s * 1000, step_loss_ok);
  }

  // final judgement
  printf("overall okay: %d\n", allok);

  // free everything
  free(x);
  free(y);
  free(expected_logits);
  free(expected_loss);
  free(expected_grads_memory);
  gpt2_free(&model);

  return 0;
}