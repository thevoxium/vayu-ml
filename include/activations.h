#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "layer.h"
#include "tensor.h"
#include <memory>
#include <vector>

class Relu : public Layer {
public:
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
  void zero_grad() override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
};

class Softmax : public Layer {
public:
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
  void zero_grad() override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
};

class Sigmoid : public Layer {
public:
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
  void zero_grad() override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
};

class Tanh : public Layer {
public:
  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
  void zero_grad() override;
  std::vector<std::shared_ptr<Tensor>> parameters() override;
};

#endif // ACTIVATIONS_H
