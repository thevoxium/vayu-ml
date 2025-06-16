#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>

#ifdef __APPLE__
#include <arm_neon.h>
#endif // __APPLE__

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);

#ifdef __APPLE__
  if (this->numel() == other->numel() && this->numel() == out->numel()) {
    const size_t simd_elements = (out->numel() / 4) * 4;
    for (size_t i = 0; i < simd_elements; i += 4) {
      float32x4_t first_vec = vld1q_f32(&this->data[i]);
      float32x4_t second_vec = vld1q_f32(&other->data[i]);
      float32x4_t result = vaddq_f32(first_vec, second_vec);
      vst1q_f32(&out->data[i], result);
    }
    for (size_t i = simd_elements; i < out->numel(); i++) {
      size_t idx1 = i % this->numel(), idx2 = i % other->numel();
      out->data[i] = this->data[idx1] + other->data[idx2];
    }
  } else {
    for (size_t i = 0; i < out->numel(); i++) {
      size_t idx1 = i % this->numel(), idx2 = i % other->numel();
      out->data[i] = this->data[idx1] + other->data[idx2];
    }
  }
#else
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] + other->data[idx2];
  }
#endif
  out->_prev = {shared_from_this(), other};
  out->_op = "+";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {

#ifdef __APPLE__
      if (self_ptr->numel() == other->numel() &&
          other->numel() == out->numel()) {
        const size_t simd_elements = (out->numel() / 4) * 4;
        for (size_t i = 0; i < simd_elements; i += 4) {
          float32x4_t current_grad = vld1q_f32(&other->grad[i]);
          float32x4_t out_grad = vld1q_f32(&out->grad[i]);
          float32x4_t result = vaddq_f32(current_grad, out_grad);
          vst1q_f32(&other->grad[i], result);
        }
        for (size_t i = simd_elements; i < out->numel(); i++) {
          self_ptr->grad[i] += out->grad[i];
        }

      } else {

        for (size_t i = 0; i < out->numel(); i++) {
          size_t idx = i % self_ptr->numel();
          self_ptr->grad[idx] += out->grad[i];
        }
      }
#else
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % self_ptr->numel();
        self_ptr->grad[idx] += out->grad[i];
      }
#endif // __APPLE__
    }
    if (other->requires_grad) {

#ifdef __APPLE__
      if (other->numel() == self_ptr->numel() &&
          self_ptr->numel() == out->numel()) {
        const size_t simd_elements = (out->numel() / 4) * 4;
        for (size_t i = 0; i < simd_elements; i += 4) {
          float32x4_t current_grad = vld1q_f32(&self_ptr->grad[i]);
          float32x4_t out_grad = vld1q_f32(&out->grad[i]);
          float32x4_t result = vaddq_f32(current_grad, out_grad);
          vst1q_f32(&self_ptr->grad[i], result);
        }
        for (size_t i = simd_elements; i < out->numel(); i++) {
          other->grad[i] += out->grad[i];
        }

      } else {

        for (size_t i = 0; i < out->numel(); i++) {
          size_t idx = i % other->numel();
          other->grad[idx] += out->grad[i];
        }
      }
#else
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % other->numel();
        other->grad[idx] += out->grad[i];
      }
#endif // __APPLE__
    }
  };

  return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator+(b);
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] * other->data[idx2];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "*";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx1 = i % self_ptr->numel(), idx2 = i % other->numel();
        self_ptr->grad[idx1] += (out->grad[i] * other->data[idx2]);
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx2 = i % other->numel(), idx1 = i % self_ptr->numel();
        other->grad[idx2] += (out->grad[i] * self_ptr->data[idx1]);
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);

  // Direct subtraction without intermediate tensors
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] - other->data[idx2];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "-";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % self_ptr->numel();
        self_ptr->grad[idx] += out->grad[i]; // +1 for subtraction
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % other->numel();
        other->grad[idx] -= out->grad[i]; // -1 for subtraction
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator-(b);
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator*(b);
}
