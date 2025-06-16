#include "../include/tensor.h"
#include <algorithm>
#include <cmath>
#include <memory>

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
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }

  out->_prev = {shared_from_this()};
  out->_op = "sigmoid";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        auto sig = out->data[i];
        self_ptr->grad[i] += (sig * (1.0f - sig)) * out->grad[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a) {
  return a->sigmoid();
}
