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
      auto param = parameter->View();
      auto grad = parameter->View(nn::Parameter::kGrad);
      for (size_t i = 0; i < param.size(); ++i) {
        param[i] -= lr_ * grad[i];
      }
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
      m_.emplace_back(parameter->size());
      v_.emplace_back(parameter->size());
    }
  }

  void ZeroGrad() {
    for (nn::Parameter* parameter : parameters_) {
      parameter->ZeroGrad();
    }
  }

  void Step(int t) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto parameter = parameters_[i]->View();
      auto gradient = parameters_[i]->View(nn::Parameter::kGrad);
      auto momentum = m_[i].View();
      auto velocity = v_[i].View();
      for (size_t j = 0; j < parameter.size(); ++j) {
        float grad = gradient[j];
        float param = parameter[j];
        float& m = momentum[j];
        float& v = velocity[j];
        // update the first moment (momentum)
        m = beta1_ * m + (1.0f - beta1_) * grad;
        // update the second moment (RMSprop)
        v = beta2_ * v + (1.0f - beta2_) * grad * grad;

        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1_, t));
        float v_hat = v / (1.0f - powf(beta2_, t));

        // update
        parameter[j] -=
            lr_ * (m_hat / (std::sqrt(v_hat) + eps_) + weight_decay_ * param);
      }
    }
  }

 private:
  std::vector<nn::Parameter*> parameters_;
  std::vector<nn::Parameter> m_;
  std::vector<nn::Parameter> v_;
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
};

}  // namespace optim

#endif  // LLM_CPP__OPTIM_HPP_
