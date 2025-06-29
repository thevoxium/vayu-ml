#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>

#ifdef __APPLE__
#include <arm_neon.h>
#endif // __APPLE__

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  // Since we're using explicit broadcasting, both tensors should have same
  // shape
  assert(this->shape == other->shape &&
         "Tensors must have same shape. Use broadcast() first.");

  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad ||
                                                       other->requires_grad);

  // Simple element-wise addition
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = this->data[i] + other->data[i];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "+";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        self_ptr->grad[i] += out->grad[i]; // Simple 1:1 mapping
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        other->grad[i] += out->grad[i]; // Simple 1:1 mapping
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::operator-(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape &&
         "Tensors must have same shape. Use broadcast() first.");

  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad ||
                                                       other->requires_grad);

  // Simple element-wise subtraction
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = this->data[i] - other->data[i];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "-";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        self_ptr->grad[i] += out->grad[i]; // +1 for subtraction
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        other->grad[i] -= out->grad[i]; // -1 for subtraction
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape &&
         "Tensors must have same shape. Use broadcast() first.");
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    out->data[i] = this->data[i] * other->data[i];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "*";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        self_ptr->grad[i] += (out->grad[i] * other->data[i]);
      }
    }
    if (other->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        other->grad[i] += (out->grad[i] * self_ptr->data[i]);
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator+(b);
}
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator-(b);
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator*(b);
}

std::shared_ptr<Tensor> Tensor::gt(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] > other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->gt(b);
}

std::shared_ptr<Tensor> Tensor::eq(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] == other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->eq(b);
}

std::shared_ptr<Tensor> Tensor::ne(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] != other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->ne(b);
}

std::shared_ptr<Tensor> Tensor::lt(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] < other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->lt(b);
}

std::shared_ptr<Tensor> Tensor::le(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] <= other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->le(b);
}

std::shared_ptr<Tensor> Tensor::ge(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad |
                                                       other->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] >= other->data[i])
      out->data[i] = 1.0f;
    else
      out->data[i] = 0.0f;
  }
  return out;
}

std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
  return a->ge(b);
}
