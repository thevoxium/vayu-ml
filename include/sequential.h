#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "layer.h"
#include "tensor.h"
#include <memory>
#include <utility>
#include <vector>

class Sequential {
private:
  std::vector<std::unique_ptr<Layer>> layers;

public:
  template <typename LayerType> void add(LayerType &&layer) {
    layers.push_back(
        std::make_unique<LayerType>(std::forward<LayerType>(layer)));
  }

  std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
  std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);
  std::vector<std::shared_ptr<Tensor>> parameters();

  void zero_grad();
  size_t size() const;
};

#endif // SEQUENTIAL_H
