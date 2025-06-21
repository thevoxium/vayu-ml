#ifndef MODEL_SAVE_LOAD_H
#define MODEL_SAVE_LOAD_H

#include "sequential.h"
#include "tensor.h"
#include <cstddef>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>
inline void save_tensor(std::shared_ptr<Tensor> t, std::ofstream &out) {
  size_t n_dim = t->shape.size();
  out.write(reinterpret_cast<char *>(&n_dim), sizeof(size_t));
  out.write(reinterpret_cast<char *>(t->shape.data()), n_dim * sizeof(float));
  size_t data_size = t->data.size();
  out.write(reinterpret_cast<char *>(&data_size), sizeof(size_t));
  out.write(reinterpret_cast<char *>(t->data.data()), sizeof(float));
}

inline std::shared_ptr<Tensor> load_tensor(std::ifstream &in,
                                           bool requires_grad = true) {
  size_t n_dim;
  in.read(reinterpret_cast<char *>(n_dim), sizeof(size_t));
  std::vector<size_t> shape;
  in.read(reinterpret_cast<char *>(shape.data()), sizeof(float) * n_dim);
  size_t data_size;
  in.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
  std::vector<float> data(data_size);
  in.read(reinterpret_cast<char *>(data.data()), data_size * sizeof(float));

  return std::make_shared<Tensor>(data, shape, requires_grad);
}

inline void save_model(Sequential &model, const std::string &path) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Failed to open file for saving model");

  auto params = model.parameters();
  size_t count = params.size();
  out.write(reinterpret_cast<char *>(&count), sizeof(count));

  for (auto &p : params) {
    save_tensor(p, out);
  }

  out.close();
}

inline void load_model(Sequential &model, const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("Failed to open file for loading model");

  auto params = model.parameters(); // assumes layers are already added
  size_t count;
  in.read(reinterpret_cast<char *>(&count), sizeof(count));

  if (count != params.size()) {
    throw std::runtime_error("Parameter count mismatch during model loading");
  }

  for (size_t i = 0; i < count; ++i) {
    auto t = load_tensor(in, params[i]->requires_grad);
    params[i]->data = t->data;
    params[i]->shape = t->shape;
  }
  in.close();
}

#endif
