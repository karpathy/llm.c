#define EIGEN_USE_GPU

#include "gpt.hpp"
// #include "optim.hpp"

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

using Tensor1D = Eigen::Tensor<float, 1, Eigen::RowMajor>;
using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;

int main(int argc, char** argv) {
  nn::ManualSeed(42);
  int B = 4, T = 64, C = 768, vocab_size = 50304;
  std::vector<float> x(B * T * C), lm_head(C * vocab_size),
      y(B * T * vocab_size);
  nn::NormalFill(absl::MakeSpan(x));
  nn::NormalFill(absl::MakeSpan(lm_head));
  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);
  //  Eigen::ThreadPool thread_pool(16);
  //  Eigen::ThreadPoolDevice gpu_device(&thread_pool, 12);

  float *dx, *dy, *dlm_head;
  dx = static_cast<float*>(gpu_device.allocate(sizeof(float) * B * T * C));
  dlm_head =
      static_cast<float*>(gpu_device.allocate(sizeof(float) * C * vocab_size));
  dy = static_cast<float*>(
      gpu_device.allocate(sizeof(float) * B * T * vocab_size));
  gpu_device.memcpyHostToDevice(dx, x.data(), sizeof(float) * B * T * C);
  gpu_device.memcpyHostToDevice(dlm_head, lm_head.data(),
                                sizeof(float) * C * vocab_size);
  gpu_device.memcpyHostToDevice(dy, y.data(),
                                sizeof(float) * B * T * vocab_size);

  auto xm = Eigen::TensorMap<Tensor2D>(dx, B * T, C);
  auto lm_headm = Eigen::TensorMap<Tensor2D>(dlm_head, C, vocab_size);
  auto ym = Eigen::TensorMap<Tensor2D>(dy, B * T, vocab_size);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i) {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
        Eigen::IndexPair<int>(1, 0)};
    ym.device(gpu_device) = xm.contract(lm_headm, product_dims);
    //    nn::MatMul::Forward(xm, lm_headm, ym);
  }
  gpu_device.synchronize();
  auto end = std::chrono::steady_clock::now();
  std::cout << "avg: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   (end - start))
                       .count() /
                   10
            << std::endl;

  return 0;
}
