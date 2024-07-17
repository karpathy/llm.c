#include "gpt.hpp"
#include "optim.hpp"

int main(int argc, char** argv) {
  /*
torch.set_printoptions(precision=6)
torch.manual_seed(42)
config = GPTConfig(block_size=8, n_embd=16, n_head=4, n_layer=8, vocab_size=100)
gpt2 = GPT(config=config)
B, T, C = 4, 8, 16
idx = torch.LongTensor([[35, 28, 51,  9, 81, 41, 30, 22],
                        [99, 91, 96, 20, 99, 46, 85, 63],
                        [ 0, 78, 75, 43, 94, 99, 78, 93],
                        [14, 42, 54, 11, 63, 42, 99, 48]])
targets = torch.LongTensor([[28, 51,  9, 81, 41, 30, 22, 99],
                            [91, 96, 20, 99, 46, 85, 63, 0],
                            [78, 75, 43, 94, 99, 78, 93, 14],
                            [42, 54, 11, 63, 42, 99, 48, 0]])
optimizer = torch.optim.SGD(gpt2.parameters(),
                            lr=1e-2)
for i in range(10):
  logit, loss = gpt2(idx, targets)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print('loss', loss)
  */

  Eigen::setNbThreads(4);
  nn::ManualSeed(42);
  int block_size = 8, n_embd = 16, n_head = 4, n_layer = 8, vocab_size = 100;
  int B = 4, T = block_size, C = n_embd, nh = n_head, hs = n_embd / nh;
  gpt::GPT<float> gpt(block_size, vocab_size, vocab_size, n_layer, n_head,
                      n_embd);

  std::vector<int> idx = {35, 28, 51, 9,  81, 41, 30, 22, 99, 91, 96,
                          20, 99, 46, 85, 63, 0,  78, 75, 43, 94, 99,
                          78, 93, 14, 42, 54, 11, 63, 42, 99, 48};
  auto idx_m = TTypes<int>::ConstMatrix(idx.data(), B, T);
  std::vector<float> logits(B * T * vocab_size);
  auto logits_3d = Make3DTensor(logits.data(), B, T, vocab_size);

  std::vector<int> target = {28, 51, 9,  81, 41, 30, 22, 99, 91, 96, 20,
                             99, 46, 85, 63, 0,  78, 75, 43, 94, 99, 78,
                             93, 14, 42, 54, 11, 63, 42, 99, 48, 0};
  auto target_m = TTypes<int>::ConstMatrix(target.data(), B, T);

  std::vector<nn::Parameter*> parameters;
  gpt.Parameters(&parameters);
  auto optimizer = optim::SGD(parameters, 1e-2f);
  float expected_loss[] = {
      4.691669, 4.668904, 4.646729, 4.625142, 4.604129,
      4.583667, 4.563725, 4.544271, 4.525268, 4.506680,
  };
  for (int step = 0; step < 10; ++step) {
    float loss = 0.0;
    gpt.Forward(idx_m, target_m, logits_3d, &loss);
    optimizer.ZeroGrad();
    gpt.Backward(idx_m, target_m);
    optimizer.Step();
    fprintf(stdout, "Step %d, loss = %.6f\n", step, loss);
    CHECK(std::abs(loss - expected_loss[step]) < 1e-5);
  }
}
