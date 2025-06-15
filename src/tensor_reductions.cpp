#include "../include/tensor.h"
#include <algorithm>
#include <cmath>
#include <numeric>

std::shared_ptr<Tensor> Tensor::sum() {
  auto out =
      std::make_shared<Tensor>(std::vector<size_t>{1}, this->requires_grad);
  float total_sum = 0.0;
  for (size_t i = 0; i < this->numel(); i++) {
    total_sum += this->data[i];
  }
  out->data[0] = total_sum;

  out->_prev = {shared_from_this()};
  out->_op = "sum";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); ++i) {
        self_ptr->grad[i] += out->grad[0];
      }
    }
  };
  return out;
}
