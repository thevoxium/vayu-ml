#ifndef TENSOR_H
#define TENSOR_H
#include <cassert>
#include <cstddef>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#ifdef USE_OPENBLAS
#include <cblas.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#elif defined(USE_MKL)
#include <mkl.h>
#endif

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<size_t> shape;

  std::function<void()> _backward;
  std::set<std::shared_ptr<Tensor>> _prev;
  std::string _op;
  bool requires_grad;

  Tensor(const std::vector<size_t> &shape, bool requires_grad = true);
  Tensor(const std::vector<float> &data, const std::vector<size_t> &shape,
         bool requires_grad = true);

  size_t numel() const;

  std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> other, bool fast = true);
  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> transpose();

  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};

std::shared_ptr<Tensor> tensor(const std::vector<float> &data,
                               const std::vector<size_t> &shape,
                               bool requires_grad);

std::shared_ptr<Tensor> tensor(const std::vector<size_t> &shape,
                               bool requires_grad);
std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b);
#endif // !TENSOR_H
