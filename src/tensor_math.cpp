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

std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> base, float exponent) {
  return base->pow(exponent);
}
