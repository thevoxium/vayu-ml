#include "../include/tensor.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Tensor::mse_loss(std::shared_ptr<Tensor> target) {
  assert(this->shape == target->shape);
  auto self_ptr = shared_from_this();
  auto diff = self_ptr - target;
  auto diff_sq = diff * diff;
  auto sum_loss = diff_sq->sum();
  auto out = std::make_shared<Tensor>(std::vector<size_t>{1, 1},
                                      sum_loss->requires_grad);
  float n = static_cast<float>(this->numel());
  out->data[0] = sum_loss->data[0] / n;
  out->_prev = {sum_loss};
  out->_op = "mse_loss";
  out->_backward = [sum_loss, out, n]() {
    if (sum_loss->requires_grad) {
      sum_loss->grad[0] += out->grad[0] / n;
    }
  };

  return out;
}

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> predicted,
                                 std::shared_ptr<Tensor> target) {
  return predicted->mse_loss(target);
}

std::shared_ptr<Tensor>
Tensor::cross_entropy_loss(std::shared_ptr<Tensor> target) {
  assert(this->shape == target->shape);
  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{1, 1}, this->requires_grad || target->requires_grad);
  size_t batch_size = this->shape[0], num_classes = this->shape[1];
  float sum_loss = 0.0f;
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t i = 0; i < num_classes; i++) {
      sum_loss += (-1.0f * std::log(1e-9f + this->data[b * num_classes + i]) *
                   target->data[b * num_classes + i]);
    }
  }
  out->data[0] = 1.0f * sum_loss / static_cast<float>(batch_size);
  out->_prev = {shared_from_this(), target};
  out->_op = "cross_entropy_loss";
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, target, batch_size, num_classes, out]() {
    if (self_ptr->requires_grad) {
      for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < num_classes; i++) {
          size_t idx = b * num_classes + i;
          self_ptr->grad[idx] +=
              out->grad[0] *
              (-target->data[idx] / (self_ptr->data[idx] + 1e-9f)) /
              static_cast<float>(batch_size);
        }
      }
    }
  };
  return out;
}

std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> predicted,
                                           std::shared_ptr<Tensor> target) {
  return predicted->cross_entropy_loss(target);
}
