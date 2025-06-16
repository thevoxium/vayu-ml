#include "../include/linear.h"
#include <cstddef>
#include <memory>
#include <vector>

Linear::Linear(size_t in_dim, size_t out_dim)
    : in_dim(in_dim), out_dim(out_dim) {
  float limit = std::sqrt(6.0f / (in_dim + out_dim));
  weights = random_tensor({in_dim, out_dim}, true, -limit, limit);
  bias = tensor({out_dim}, true);
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
  auto f = input->mm(weights);
  auto out = f + bias;
  return out;
}

std::shared_ptr<Tensor> Linear::operator()(std::shared_ptr<Tensor> input) {
  return forward(input);
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
  return {weights, bias};
}

void Linear::zero_grad() {
  weights->zero_grad();
  bias->zero_grad();
}

std::shared_ptr<Tensor> Linear::get_weights() const { return weights; }

std::shared_ptr<Tensor> Linear::get_bias() const { return bias; }
