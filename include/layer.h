#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include <memory>
#include <vector>

class Layer {
public:
  virtual ~Layer() = default;
  virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;
  virtual std::vector<std::shared_ptr<Tensor>> parameters() = 0;
  virtual void zero_grad() = 0;
  std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) {
    return forward(input);
  }
};

#endif // !LAYER_H
