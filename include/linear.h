#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

class Linear {
private:
  std::shared_ptr<Tensor> weights;
  std::shared_ptr<Tensor> bias;
  size_t in_dim, out_dim;

public:
  Linear(size_t in_dim, size_t out_dim);
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

  std::shared_ptr<Tensor> get_weights() const;
  std::shared_ptr<Tensor> get_bias() const;

  std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);

  void zero_grad();
  std::vector<std::shared_ptr<Tensor>> parameters();
};

#endif // !LINEAR_H
