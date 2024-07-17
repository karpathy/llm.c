#include "gpt.hpp"
#include "optim.hpp"

using Tensor1D = Eigen::Tensor<float, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

int main(int argc, char** argv) {
  std::cout << "sizeof Tensor1D : " << sizeof(Tensor1D) << std::endl;
  std::cout << "sizeof Tensor2D : " << sizeof(Tensor2D) << std::endl;
  std::cout << "sizeof Tensor3D : " << sizeof(Tensor3D) << std::endl;
  std::cout << "sizeof Tensor4D : " << sizeof(Tensor4D) << std::endl;

  std::cout << "sizeof map Tensor1D : " << sizeof(Eigen::TensorMap<Tensor1D>)
            << std::endl;
  std::cout << "sizeof map Tensor2D : " << sizeof(Eigen::TensorMap<Tensor2D>)
            << std::endl;
  std::cout << "sizeof map Tensor3D : " << sizeof(Eigen::TensorMap<Tensor3D>)
            << std::endl;
  std::cout << "sizeof map Tensor4D : " << sizeof(Eigen::TensorMap<Tensor4D>)
            << std::endl;

  Eigen::setNbThreads(4);
  nn::ManualSeed(42);
  int B = 4, T = 64, C = 768, vocab_size = 50304;
  std::vector<float> x(B * T * C), lm_head(C * vocab_size),
      y(B * T * vocab_size);
  nn::NormalFill(absl::MakeSpan(x));
  nn::NormalFill(absl::MakeSpan(lm_head));

  auto xm = MakeConstMatrix(x.data(), B * T, C);
  auto lm_headm = MakeConstMatrix(lm_head.data(), C, vocab_size);
  auto ym = MakeMatrix(y.data(), B * T, vocab_size);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i) {
    nn::MatMul<float>::Forward(xm, lm_headm, ym);
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "avg: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   (end - start))
                       .count() /
                   10
            << std::endl;

  return 0;
}
