#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <memory>
#include <vector>

class Optimiser {
protected:
  std::vector<std::shared_ptr<Tensor>> params;

public:
  virtual ~Optimiser() = default;
  virtual void step() = 0;

  void zero_grad() {
    for (auto &param : params) {
      param->zero_grad();
    }
  }
};

class SGD : public Optimiser {
private:
  float learning_rate;

public:
  SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr)
      : learning_rate(lr) {
    params = parameters;
  }

  void step() override {
    for (auto &param : params) {
      for (size_t i = 0; i < param->numel(); ++i) {
        param->data[i] -= learning_rate * param->grad[i];
      }
    }
  }
};

#endif
