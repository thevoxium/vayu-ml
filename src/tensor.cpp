#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
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

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
  assert(this->shape == other->shape);
  auto result_shape = other->shape;
  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    out->data[i] = this->data[i] * other->data[i];
  }
  return out;
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator*(b);
}
std::shared_ptr<Tensor> Tensor::mm(std::shared_ptr<Tensor> other, bool fast) {
  assert(this->shape.size() == 2 && other->shape.size() == 2);
  assert(this->shape[1] == other->shape[0]);

  int m = static_cast<int>(this->shape[0]);
  int k = static_cast<int>(this->shape[1]);
  int n = static_cast<int>(other->shape[1]);

  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      requires_grad = this->requires_grad || other->requires_grad);
  if (fast) {
    cblas_sgemm(CblasRowMajor,         // Matrix storage order
                CblasNoTrans,          // Don't transpose A
                CblasNoTrans,          // Don't transpose B
                m, n, k,               // Matrix dimensions
                1.0f,                  // alpha = 1.0
                this->data.data(), k,  // Matrix A and leading dimension
                other->data.data(), n, // Matrix B and leading dimension
                0.0f,                  // beta = 0.0 (don't add to C)
                out->data.data(), n);  // Matrix C and leading dimension
  } else {

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t kk = 0; kk < k; ++kk) {
          sum += this->data[i * k + kk] * other->data[kk * n + j];
        }
        out->data[i * n + j] = sum;
      }
    }
  }

  return out;
}

std::shared_ptr<Tensor> Tensor::relu() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::max(0.0f, this->data[i]);
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }
  return out;
}

std::shared_ptr<Tensor> Tensor::sum() {
  auto out =
      std::make_shared<Tensor>(std::vector<size_t>{1}, this->requires_grad);
  float total_sum = 0.0;
  for (size_t i = 0; i < this->numel(); i++) {
    total_sum += this->data[i];
  }
  out->data[0] = total_sum;
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
