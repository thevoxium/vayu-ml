#include "../include/sequential.h"
#include <cassert>

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
  auto output = input;
  for (auto &layer : layers) {
    output = layer->forward(output);
  }
  return output;
}

std::shared_ptr<Tensor> Sequential::operator()(std::shared_ptr<Tensor> input) {
  return forward(input);
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
  std::vector<std::shared_ptr<Tensor>> all_params;

  for (auto &layer : layers) {
    auto layer_params = layer->parameters();
    all_params.insert(all_params.end(), layer_params.begin(),
                      layer_params.end());
  }

  return all_params;
}

void Sequential::zero_grad() {
  for (auto &layer : layers) {
    layer->zero_grad();
  }
}

size_t Sequential::size() const { return layers.size(); }
