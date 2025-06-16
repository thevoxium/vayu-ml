#include "../include/activations.h"
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input) {
  return input->relu();
}
void Relu::zero_grad() {}
std::vector<std::shared_ptr<Tensor>> Relu::parameters() { return {}; }

std::shared_ptr<Tensor> Sigmoid::forward(std::shared_ptr<Tensor> input) {
  return input->sigmoid();
}
void Sigmoid::zero_grad() {}
std::vector<std::shared_ptr<Tensor>> Sigmoid::parameters() { return {}; }
