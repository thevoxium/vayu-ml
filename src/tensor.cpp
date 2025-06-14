#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <memory>
#include <random>
#include <vector>

Tensor::Tensor(const std::vector<float> &data, const std::vector<size_t> &shape,
               bool requires_grad)
    : data(data), shape(shape), requires_grad(requires_grad), _op("") {
  size_t total_size = 1;
  for (auto dim : shape)
    total_size *= dim;
  assert(data.size() == total_size);

  if (requires_grad) {
    grad.resize(total_size, 0.0f);
  }
  _backward = []() {};
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(data: [";
  for (size_t i = 0; i < t.data.size(); i++) {
    os << t.data[i];
    if (i != t.data.size() - 1)
      os << ", ";
  }
  os << "], requires_grad: " << std::boolalpha << t.requires_grad
     << " shape: (";

  for (size_t i = 0; i < t.shape.size(); i++) {
    os << t.shape[i];
    if (i != t.shape.size() - 1)
      os << ", ";
  }
  os << "))";
  return os;
}

std::shared_ptr<Tensor> tensor(const std::vector<float> &data,
                               const std::vector<size_t> &shape,
                               bool requires_grad) {
  return std::make_shared<Tensor>(data, shape, requires_grad);
}
