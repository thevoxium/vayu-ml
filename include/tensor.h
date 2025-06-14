#ifndef TENSOR_H
#define TENSOR_H
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<size_t> shape;

  std::function<void()> _backward;
  std::set<std::shared_ptr<Tensor>> _prev;
  std::string _op;
  bool requires_grad;

  Tensor(const std::vector<float> &data, const std::vector<size_t> &shape,
         bool requires_grad = true);

  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};

std::shared_ptr<Tensor> tensor(const std::vector<float> &data,
                               const std::vector<size_t> &shape,
                               bool requires_grad);

#endif // !TENSOR_H
