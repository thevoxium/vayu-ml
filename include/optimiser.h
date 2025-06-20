#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>
#ifdef __APPLE__
#include <arm_neon.h>
#endif //  __APPLE__

class Optimiser {
protected:
  std::vector<std::shared_ptr<Tensor>> params;

public:
  virtual ~Optimiser() = default;
  virtual void step() = 0;

  void zero_grad() {
    for (auto &param : params) {
      param->zero_grad();
    }
  }
};

class SGD : public Optimiser {
private:
  float learning_rate;

public:
  SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr)
      : learning_rate(lr) {
    params = parameters;
  }

  void step() override {
    for (auto &param : params) {

#ifdef __APPLE__
      /*#if defined(__APPLE__) && 0*/ const size_t simd_elements =
          (param->numel() / 4) * 4;
      float32x4_t lr_vec = vdupq_n_f32(learning_rate);
      for (size_t i = 0; i < simd_elements; i += 4) {
        float32x4_t grad_vec = vmulq_f32(lr_vec, vld1q_f32(&param->grad[i]));
        float32x4_t data_vec = vld1q_f32(&param->data[i]);
        float32x4_t result = vsubq_f32(data_vec, grad_vec);
        vst1q_f32(&param->data[i], result);
      }

      for (size_t i = simd_elements; i < param->numel(); i++) {
        param->data[i] -= learning_rate * param->grad[i];
      }
#else // __APPLE__
      for (size_t i = 0; i < param->numel(); ++i) {
        param->data[i] -= learning_rate * param->grad[i];
      }
#endif
    }
  }
};

class Adam : public Optimiser {
private:
  float learning_rate, beta1, beta2, eps;
  float time_step;

public:
  Adam(std::vector<std::shared_ptr<Tensor>> parameters, float lr = 3e-4,
       float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
      : learning_rate(lr), beta1(beta1), beta2(beta2), eps(eps), time_step(0) {
    params = parameters;
  }

  void step() override {
    time_step++;
    for (auto &param : params) {
      std::vector<float> m, v;
      m.resize(param->numel(), 0.0f);
      v.resize(param->numel(), 0.0f);
      for (size_t i = 0; i < param->numel(); i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * param->grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * param->grad[i] * param->grad[i];
        float m_hat = m[i] / (1.0f - std::pow(beta1, time_step));
        float v_hat = v[i] / (1.0f - std::pow(beta2, time_step));
        param->data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
      }
    }
  }
};

#endif
