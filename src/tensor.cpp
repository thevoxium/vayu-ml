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

Tensor::Tensor(const std::vector<size_t> &shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad), _op("") {
  size_t total_size = 1;
  for (auto dim : shape)
    total_size *= dim;
  data.resize(total_size, 0.0f);
  if (requires_grad) {
    grad.resize(total_size, 0.0f);
  }
  _backward = []() {};
}
size_t Tensor::numel() const { return data.size(); }

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto result_shape = other->shape;
  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    out->data[i] = this->data[i] + other->data[i];
  }
  return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator+(b);
}

std::shared_ptr<Tensor> Tensor::mm(std::shared_ptr<Tensor> other) {
  assert(this->shape.size() == 2 && other->shape.size() == 2);
  assert(this->shape[1] == other->shape[0]);

  size_t m = this->shape[0], k = this->shape[1], n = other->shape[1];

  auto out = std::make_shared<Tensor>(std::vector<size_t>{m, n},
                                      requires_grad = this->requires_grad ||
                                                      other->requires_grad);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t kk = 0; kk < k; ++kk) {
        sum += this->data[i * k + kk] * other->data[kk * n + j];
      }
      out->data[i * n + j] = sum;
    }
  }
  return out;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(data: [";
  for (size_t i = 0; i < t.data.size(); i++) {
    os << t.data[i];
    if (i != t.data.size() - 1)
      os << ", ";
  }
  os << "], requires_grad: " << std::boolalpha << t.requires_grad
     << ", shape: (";

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

std::shared_ptr<Tensor> tensor(const std::vector<size_t> &shape,
                               bool requires_grad) {
  return std::make_shared<Tensor>(shape, requires_grad);
}
