#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <memory>

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] + other->data[idx2];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "+";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % self_ptr->numel();
        self_ptr->grad[idx] += out->grad[i];
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % other->numel();
        other->grad[idx] += out->grad[i];
      }
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
