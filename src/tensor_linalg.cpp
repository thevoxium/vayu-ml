
#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>

std::shared_ptr<Tensor> Tensor::mm(std::shared_ptr<Tensor> other, bool fast) {
  assert(this->shape.size() == 2 && other->shape.size() == 2);
  assert(this->shape[1] == other->shape[0]);

  int m = static_cast<int>(this->shape[0]);
  int k = static_cast<int>(this->shape[1]);
  int n = static_cast<int>(other->shape[1]);

  auto out = std::make_shared<Tensor>(
      std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
      this->requires_grad || other->requires_grad);

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

  // Set up backward pass
  out->_prev = {shared_from_this(), other};
  out->_op = "matmul";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, other, out, m, k, n, fast]() {
    if (self_ptr->requires_grad) {
      // dA = dC @ B^T
      if (fast) {
        cblas_sgemm(CblasRowMajor,         // Matrix storage order
                    CblasNoTrans,          // Don't transpose dC
                    CblasTrans,            // Transpose B
                    m, k, n,               // Matrix dimensions (m x k result)
                    1.0f,                  // alpha = 1.0
                    out->grad.data(), n,   // dC matrix and leading dimension
                    other->data.data(), n, // B matrix and leading dimension
                    1.0f,                  // beta = 1.0 (accumulate gradients)
                    self_ptr->grad.data(),
                    k); // dA matrix and leading dimension
      } else {
        // Manual implementation: dA = dC @ B^T
        for (size_t i = 0; i < m; ++i) {
          for (size_t kk = 0; kk < k; ++kk) {
            float sum = 0.0f;
            for (size_t j = 0; j < n; ++j) {
              sum += out->grad[i * n + j] * other->data[kk * n + j];
            }
            self_ptr->grad[i * k + kk] += sum;
          }
        }
      }
    }

    if (other->requires_grad) {
      // dB = A^T @ dC
      if (fast) {
        cblas_sgemm(CblasRowMajor, // Matrix storage order
                    CblasTrans,    // Transpose A
                    CblasNoTrans,  // Don't transpose dC
                    k, n, m,       // Matrix dimensions (k x n result)
                    1.0f,          // alpha = 1.0
                    self_ptr->data.data(), k, // A matrix and leading dimension
                    out->grad.data(), n,      // dC matrix and leading dimension
                    1.0f,                   // beta = 1.0 (accumulate gradients)
                    other->grad.data(), n); // dB matrix and leading dimension
      } else {
        // Manual implementation: dB = A^T @ dC
        for (size_t kk = 0; kk < k; ++kk) {
          for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < m; ++i) {
              sum += self_ptr->data[i * k + kk] * out->grad[i * n + j];
            }
            other->grad[kk * n + j] += sum;
          }
        }
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
