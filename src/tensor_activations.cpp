#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#ifdef __APPLE__
#include <arm_neon.h>
#endif // __APPLE__

std::shared_ptr<Tensor> Tensor::relu() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::max(0.0f, this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "relu";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += (out->data[i] > 0.0f ? 1.0f : 0.0f) * out->grad[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a) { return a->relu(); }

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

#ifdef __APPLE__

  const size_t simd_elements = (this->numel() / 4) * 4;
  const float32x4_t ones = vdupq_n_f32(1.0f);
  const float32x4_t halves = vdupq_n_f32(0.5f);
  const float32x4_t sixths = vdupq_n_f32(1.0f / 6.0f);

  for (size_t i = 0; i < simd_elements; i += 4) {
    float32x4_t x = vnegq_f32(vld1q_f32(&this->data[i]));
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t exp_x = vaddq_f32(ones, x);
    exp_x = vaddq_f32(exp_x, vmulq_f32(x2, halves));
    exp_x = vaddq_f32(exp_x, vmulq_f32(x3, sixths));
    float32x4_t sigmoid = vdivq_f32(ones, vaddq_f32(ones, exp_x));
    vst1q_f32(&out->data[i], sigmoid);
  }
  for (size_t i = simd_elements; i < this->numel(); i++) {
    out->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }

#else
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }
#endif
  out->_prev = {shared_from_this()};
  out->_op = "sigmoid";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
#ifdef __APPLE__
      const size_t simd_elements = (self_ptr->numel() / 4) * 4;
      float32x4_t ones = vdupq_n_f32(1.0f);

      for (size_t i = 0; i < simd_elements; i += 4) {
        float32x4_t input_vec = vld1q_f32(&out->data[i]);
        float32x4_t out_grad = vld1q_f32(&out->grad[i]);
        float32x4_t self_grad = vld1q_f32(&self_ptr->grad[i]);
        float32x4_t result = vaddq_f32(
            self_grad,
            vmulq_f32(vmulq_f32(vsubq_f32(ones, input_vec), input_vec),
                      out_grad));

        vst1q_f32(&self_ptr->grad[i], result);
      }

      for (size_t i = simd_elements; i < self_ptr->numel(); i++) {
        auto sig = out->data[i];
        self_ptr->grad[i] += (sig * (1.0f - sig)) * out->grad[i];
      }

#else
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        auto sig = out->data[i];
        self_ptr->grad[i] += (sig * (1.0f - sig)) * out->grad[i];
      }
#endif
    }
  };

  return out;
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a) {
  return a->sigmoid();
}

std::shared_ptr<Tensor> Tensor::tanh() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (std::abs(this->data[i]) < 3) {
      float x2 = this->data[i] * this->data[i];
      out->data[i] = (this->data[i] * (27.0f + x2)) / (27.0f + 9.0f * x2);
    } else
      out->data[i] = std::tanh(this->data[i]);
  }
  out->_prev = {shared_from_this()};
  out->_op = "tanh";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += (1 - out->data[i] * out->data[i]) * out->grad[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a) { return a->tanh(); }

std::shared_ptr<Tensor> Tensor::softmax() {
  assert(this->shape.size() == 2);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  size_t batch_size = this->shape[0], num_classes = this->shape[1];
  for (size_t b = 0; b < batch_size; b++) {
    float max_val = -std::numeric_limits<float>::infinity();
    size_t base = b * num_classes;
    for (size_t c = 0; c < num_classes; c++) {
      max_val = std::max(max_val, this->data[base + c]);
    }

    float sum = 0.0f;

    for (size_t c = 0; c < num_classes; c++) {
      float exp_val = std::exp(this->data[base + c] - max_val);
      sum += exp_val;
      out->data[base + c] = exp_val;
    }
    for (size_t c = 0; c < num_classes; c++) {
      out->data[base + c] /= sum;
    }
  }

  auto self_ptr = shared_from_this();
  out->_prev = {shared_from_this()};
  out->_op = "softmax";
  out->_backward = [self_ptr, out, batch_size, num_classes]() {
    if (self_ptr->requires_grad) {
      for (size_t b = 0; b < batch_size; b++) {
        size_t base = b * num_classes;
        float dot_prod = 0.0f;
        for (size_t i = 0; i < num_classes; i++) {
          dot_prod += (out->data[base + i] * out->grad[base + i]);
        }

        for (size_t i = 0; i < num_classes; i++) {
          float s_i = out->data[base + i];
          self_ptr->grad[base + i] += (s_i * (out->grad[base + i] - dot_prod));
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> a) {
  return a->softmax();
}
