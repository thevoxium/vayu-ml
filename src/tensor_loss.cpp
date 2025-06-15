#include "../include/tensor.h"
#include <cassert>
#include <memory>

std::shared_ptr<Tensor> Tensor::mse_loss(std::shared_ptr<Tensor> target) {
  assert(this->shape == target->shape);
  auto self_ptr = shared_from_this();
  auto diff = self_ptr - target;
  auto diff_sq = diff * diff;
  auto sum_loss = diff_sq->sum();
  auto out =
      std::make_shared<Tensor>(std::vector<size_t>{1}, sum_loss->requires_grad);
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
