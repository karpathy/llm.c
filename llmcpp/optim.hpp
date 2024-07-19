#ifndef LLM_CPP__OPTIM_HPP_
#define LLM_CPP__OPTIM_HPP_

#include "nn.hpp"

namespace optim {

struct SGD {
  SGD(std::vector<nn::Parameter*> parameters, float lr)
      : parameters_(std::move(parameters)), lr_(lr) {}

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step() {
    for (nn::Parameter* parameter : parameters_) {
      auto param = parameter->flat<float>();
      auto grad = parameter->flat_grad<float>();
      param.device(nn::g_cpu_device) -= lr_ * grad;
    }
  }

 private:
  std::vector<nn::Parameter*> parameters_;
  float lr_;
};

struct AdamW {
  AdamW(std::vector<nn::Parameter*> parameters, float lr, float beta1 = 0.9f,
        float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.0f)
      : parameters_(std::move(parameters)),
        lr_(lr),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps),
        weight_decay_(weight_decay) {
    for (const auto& parameter : parameters_) {
      m_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
      v_.emplace_back(
          std::make_unique<nn::Parameter>(nn::DT_FLOAT, parameter->size()));
    }
  }

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step(int t) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto parameter = parameters_[i]->flat<float>();
      auto grad = parameters_[i]->flat_grad<float>();
      auto m = m_[i]->flat<float>();
      auto v = v_[i]->flat<float>();

      // update the first moment (momentum)
      m.device(nn::g_cpu_device) = beta1_ * m + (1.0f - beta1_) * grad;
      // update the second moment (RMSprop)
      v.device(nn::g_cpu_device) = beta2_ * v + (1.0f - beta2_) * grad * grad;
      // bias-correct both moments
      auto m_hat = m / (1.0f - static_cast<float>(std::pow(beta1_, t)));
      auto v_hat = v / (1.0f - static_cast<float>(std::pow(beta2_, t)));

      // update
      parameter.device(nn::g_cpu_device) -=
          lr_ * (m_hat / (v_hat.sqrt() + eps_) + weight_decay_ * parameter);
    }
  }

 private:
  std::vector<nn::Parameter*> parameters_;
  std::vector<std::unique_ptr<nn::Parameter>> m_;
  std::vector<std::unique_ptr<nn::Parameter>> v_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
};

}  // namespace optim

#endif  // LLM_CPP__OPTIM_HPP_
