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
  cached_size = 1;
  ndim = shape.size();
  for (auto dim : shape)
    cached_size *= dim;
  assert(data.size() == cached_size);
  compute_strides();
  if (requires_grad) {
    grad.reserve(cached_size);
    grad.resize(cached_size, 0.0f);
  }
  _backward = []() {};
}

Tensor::Tensor(const std::vector<size_t> &shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad), _op("") {
  cached_size = 1;
  ndim = shape.size();
  for (auto dim : shape)
    cached_size *= dim;
  compute_strides();
  data.reserve(cached_size);
  data.resize(cached_size, 0.0f);
  if (requires_grad) {
    grad.reserve(cached_size);
    grad.resize(cached_size, 0.0f);
  }

  _backward = []() {};
}

std::shared_ptr<Tensor> empty(const std::vector<size_t> &shape,
                              bool requires_grad) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::vector<float> data(total_size, 0.0f);
  return std::make_shared<Tensor>(data, shape, requires_grad);
}

std::shared_ptr<Tensor> Tensor::operator[](size_t idx) {
  auto out = std::make_shared<Tensor>(std::vector<size_t>{1, this->shape[1]},
                                      this->requires_grad);
  for (size_t i = 0; i < this->shape[1]; i++) {
    out->data[i] = this->data[idx * this->shape[0] + i];
  }
  return out;
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

std::shared_ptr<Tensor>
Tensor::broadcast(const std::vector<size_t> &target_shape) {
  if (this->shape == target_shape) {
    return shared_from_this();
  }
  assert(can_broadcast(this->shape, target_shape) &&
         "Cannot broadcast tensor to target shape");

  auto out = std::make_shared<Tensor>(target_shape, this->requires_grad);

  for (size_t i = 0; i < out->numel(); i++) {
    auto output_indices = out->unflatten_index(i);
    size_t source_idx = get_broadcast_source_index(output_indices, this->shape);
    out->data[i] = this->data[source_idx];
  }

  out->_prev = {shared_from_this()};
  out->_op = "broadcast";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, target_shape]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        auto output_indices = out->unflatten_index(i);
        size_t source_idx = self_ptr->get_broadcast_source_index(
            output_indices, self_ptr->shape);
        self_ptr->grad[source_idx] += out->grad[i];
      }
    }
  };

  return out;
}

size_t Tensor::get_broadcast_source_index(
    const std::vector<size_t> &output_indices,
    const std::vector<size_t> &source_shape) const {
  size_t source_idx = 0;
  int output_dim = output_indices.size();
  int source_dim = source_shape.size();

  // Handle dimension alignment (broadcasting from right)
  for (int i = 0; i < output_dim; i++) {
    int source_axis = i - (output_dim - source_dim);

    if (source_axis >= 0) {
      // This dimension exists in source tensor
      size_t coord;
      if (source_shape[source_axis] == 1) {
        // Broadcasted dimension - always use index 0
        coord = 0;
      } else {
        // Normal dimension - use the output coordinate
        coord = output_indices[i];
      }
      source_idx = source_idx * source_shape[source_axis] + coord;
    }
    // If source_axis < 0, this dimension doesn't exist in source (implicit size
    // 1)
  }

  return source_idx;
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

std::shared_ptr<Tensor> ones(const std::vector<size_t> &shape,
                             bool requires_grad) {
  size_t total_size = 1;
  for (auto dim : shape) {
    total_size *= dim;
  }

  std::vector<float> data(total_size, 1.0);
  return std::make_shared<Tensor>(data, shape, requires_grad);
}

std::shared_ptr<Tensor> full(const std::vector<size_t> &shape, float val,
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

std::shared_ptr<Tensor> arange(int start, int end, size_t step,
                               bool requires_grad) {
  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{(end - start + step - 1) / step}, requires_grad);
  int curr = start;
  for (int i = 0; i < out->numel(); i++) {
    out->data[i] = curr;
    curr += step;
  }
  return out;
}

std::shared_ptr<Tensor> linspace(double start, double stop, size_t num_steps,
                                 bool requires_grad) {
  auto out =
      std::make_shared<Tensor>(std::vector<size_t>{num_steps}, requires_grad);

  if (num_steps == 1) {
    out->data[0] = start;
    return out;
  }

  double step_size = (stop - start) / (num_steps - 1);

  for (size_t i = 0; i < num_steps; i++) {
    out->data[i] = start + i * step_size;
  }

  return out;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(";

  std::function<void(size_t, size_t, const std::vector<size_t> &)>
      print_recursive =
          [&](size_t offset, size_t dim, const std::vector<size_t> &coords) {
            if (dim == t.shape.size() - 1) {
              // Print the innermost dimension
              os << "[";
              for (size_t i = 0; i < t.shape[dim]; ++i) {
                os << t.data[offset + i];
                if (i < t.shape[dim] - 1)
                  os << ", ";
              }
              os << "]";
            } else {
              os << "[";
              for (size_t i = 0; i < t.shape[dim]; ++i) {
                if (i > 0)
                  os << ",\n" << std::string(dim + 1, ' ');
                size_t new_offset = offset + i * t.strides[dim];
                print_recursive(new_offset, dim + 1, coords);
              }
              os << "]";
            }
          };

  print_recursive(0, 0, {});
  os << ", shape: (";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    os << t.shape[i];
    if (i < t.shape.size() - 1)
      os << ", ";
  }
  os << "), requires_grad: " << t.requires_grad << ")";
  return os;
}
