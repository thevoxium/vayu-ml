#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Tensor::pow(float exponent) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::pow(this->data[i], exponent);
  }

  out->_prev = {shared_from_this()};
  out->_op = "pow";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, exponent]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (exponent == 0.0f)
          self_ptr->grad[i] = 0.0f;
        else {
          self_ptr->grad[i] += out->grad[i] * exponent *
                               std::pow(self_ptr->data[i], exponent - 1.0f);
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::exp() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::exp(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "exp";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += out->grad[i] * out->data[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::log() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  const float epsilon = 1e-8f;
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::log(std::max(this->data[i], epsilon));
  }

  out->_prev = {shared_from_this()};
  out->_op = "log";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] > 0.0f)
          self_ptr->grad[i] += out->grad[i] * (1.0f / self_ptr->data[i]);
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> base, float exponent) {
  return base->pow(exponent);
}

std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> num) { return num->log(); }
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> base) {
  return base->exp();
}
