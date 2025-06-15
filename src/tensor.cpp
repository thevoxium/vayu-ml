#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <ios>
#include <iostream>
#include <memory>
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

std::shared_ptr<Tensor> make_ones(const std::vector<size_t> &shape,
                                  bool requires_grad) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::vector<float> data(total_size, 1.0);
  return std::make_shared<Tensor>(data, shape, requires_grad);
}
std::shared_ptr<Tensor> random_tensor(const std::vector<size_t> &shape,
                                      bool requires_grad, float min_val,
                                      float max_val) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  std::vector<float> data(total_size);
  for (size_t i = 0; i < total_size; ++i) {
    data[i] = dis(gen);
  }

  return std::make_shared<Tensor>(data, shape, requires_grad);
}

size_t Tensor::numel() const { return data.size(); }

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] + other->data[idx2];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "+";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % self_ptr->numel();
        self_ptr->grad[idx] += out->grad[i];
      }
    }
    if (out->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx = i % other->numel();
        other->grad[idx] += out->grad[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b) {
  return a->operator+(b);
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
  assert(can_broadcast(this->shape, other->shape));
  auto result_shape = broadcast_shape(this->shape, other->shape);

  auto out = std::make_shared<Tensor>(result_shape, this->requires_grad ||
                                                        other->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    size_t idx1 = i % this->numel(), idx2 = i % other->numel();
    out->data[i] = this->data[idx1] * other->data[idx2];
  }

  out->_prev = {shared_from_this(), other};
  out->_op = "*";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx1 = i % self_ptr->numel(), idx2 = i % other->numel();
        self_ptr->grad[idx1] += (out->grad[i] * other->data[idx2]);
      }
    }
    if (out->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        size_t idx2 = i % other->numel(), idx1 = i % self_ptr->numel();
        other->grad[idx2] += (out->grad[i] * self_ptr->data[idx1]);
      }
    }
  };

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

  out->_prev = {shared_from_this()};
  out->_op = "relu";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += (out->data[i] > 0.0f ? 1.0f : 0.0f) * out->grad[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = 1.0f / (1.0f + std::exp(-this->data[i]));
  }

  out->_prev = {shared_from_this()};
  out->_op = "sigmoid";

  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        auto sig = out->data[i];
        self_ptr->grad[i] += (sig * (1.0f - sig)) * out->grad[i];
      }
    }
  };

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

std::shared_ptr<Tensor> Tensor::transpose() {
  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{this->shape[1], this->shape[0]}, this->requires_grad);
  size_t rows = this->shape[0], cols = this->shape[1];

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[j * rows + i] = this->data[i * cols + j];
    }
  }

  out->_prev = {shared_from_this()};
  out->_op = "transpose";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, rows, cols]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
          self_ptr->grad[i * cols + j] += out->grad[j * rows + i];
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::reshape(const std::vector<size_t> &shape) {
  size_t newsize = 1;
  for (auto dim : shape)
    newsize *= dim;
  assert(newsize == this->numel());

  auto out = std::make_shared<Tensor>(shape, this->requires_grad);
  out->data = this->data;
  out->_prev = {shared_from_this()};
  out->_op = "reshape";
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += out->grad[i];
      }
    }
  };
  return out;
}

float Tensor::operator[](size_t idx) { return data[idx]; }

bool Tensor::can_broadcast(const std::vector<size_t> shape1,
                           const std::vector<size_t> shape2) {
  int max_size = std::max(shape1.size(), shape2.size());
  for (int i = 0; i < max_size; ++i) {
    int dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
    int dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }
  return true;
}

std::vector<size_t> Tensor::broadcast_shape(const std::vector<size_t> shape1,
                                            const std::vector<size_t> shape2) {
  int max_size = std::max(shape1.size(), shape2.size());
  std::vector<size_t> result_shape(max_size);
  for (int i = 0; i < max_size; ++i) {
    size_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
    size_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
    result_shape[max_size - 1 - i] = std::max(dim1, dim2);
  }
  return result_shape;
}

void Tensor::init_grad() {
  if (requires_grad && grad.empty()) {
    grad.resize(data.size(), 0.0f);
  }
}

void Tensor::zero_grad() {
  if (requires_grad) {
    std::fill(grad.begin(), grad.end(), 0.0f);
  }
}

void Tensor::backward() {
  std::vector<std::shared_ptr<Tensor>> topo;
  std::set<std::shared_ptr<Tensor>> visited;

  std::function<void(std::shared_ptr<Tensor>)> build_topo =
      [&](std::shared_ptr<Tensor> v) {
        if (visited.find(v) == visited.end()) {
          visited.insert(v);
          for (auto child : v->_prev) {
            build_topo(child);
          }
          topo.push_back(v);
        }
      };

  build_topo(shared_from_this());
  this->init_grad();
  std::fill(this->grad.begin(), this->grad.end(), 1.0f);

  std::reverse(topo.begin(), topo.end());
  for (auto tensor : topo) {
    tensor->_backward();
  }
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(data: \n";
  size_t rows = t.shape[0];
  size_t cols = t.shape[1];

  os << "[";
  for (size_t i = 0; i < rows; i++) {
    if (i > 0)
      os << " "; // Indentation for subsequent rows
    os << "[";
    for (size_t j = 0; j < cols; j++) {
      size_t index = i * cols + j;
      os << t.data[index];
      if (j != cols - 1)
        os << ", ";
    }
    os << "]";
    if (i != rows - 1)
      os << ",\n";
  }
  os << "]";

  os << ",\nrequires_grad: " << std::boolalpha << t.requires_grad
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
