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

size_t Tensor::numel() const { return data.size(); }

float Tensor::operator[](size_t idx) { return data[idx]; }

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

void Tensor::clear_graph() {
  std::set<std::shared_ptr<Tensor>> visited;

  std::function<void(std::shared_ptr<Tensor>)> clear_recursive =
      [&](std::shared_ptr<Tensor> node) {
        if (visited.find(node) != visited.end())
          return;
        visited.insert(node);

        // Recursively clear children first
        for (auto child : node->_prev) {
          clear_recursive(child);
        }

        // Clear this node
        node->_prev.clear();
        node->_backward = []() {};
      };

  clear_recursive(shared_from_this());
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

std::shared_ptr<Tensor> make_ones(const std::vector<size_t> &shape,
                                  bool requires_grad) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::vector<float> data(total_size, 1.0);
  return std::make_shared<Tensor>(data, shape, requires_grad);
}

std::shared_ptr<Tensor> make_const(const std::vector<size_t> &shape, float val,
                                   bool requires_grad) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::vector<float> data(total_size, val);
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

std::shared_ptr<Tensor> asvector(const std::vector<std::vector<float>> &input,
                                 bool requires_grad) {
  if (input.empty() || input[0].empty()) {
    return std::make_shared<Tensor>(std::vector<size_t>{0, 0}, requires_grad);
  }
  size_t rows = input.size(), cols = input[0].size();
  for (const auto &row : input) {
    if (row.size() != cols) {
      throw std::invalid_argument("All rows must have the same length");
    }
  }
  auto out =
      std::make_shared<Tensor>(std::vector<size_t>{rows, cols}, requires_grad);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      out->data[i * cols + j] = input[i][j];
    }
  }
  return out;
}

std::shared_ptr<Tensor> arrange(int start, int end, size_t step,
                                bool requires_grad) {
  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{1, (end - start + step - 1) / step}, requires_grad);
  int curr = start;
  for (int i = 0; i < out->numel(); i++) {
    out->data[i] = curr;
    curr += step;
  }
  return out;
}
