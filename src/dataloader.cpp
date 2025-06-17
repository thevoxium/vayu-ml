#include "../include/dataloader.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

Dataloader::Dataloader(std::shared_ptr<Tensor> X, std::shared_ptr<Tensor> Y,
                       size_t batch_size, bool use_shuffle)
    : X(X), Y(Y), batch_size(batch_size), shuffle(use_shuffle) {
  assert(X->shape[0] == Y->shape[0]);
  num_samples = X->shape[0];

  indices.resize(num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    indices[i] = i;
  }

  if (shuffle) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
  }
}

size_t Dataloader::num_batch() const {
  return (num_samples + batch_size - 1) / batch_size;
}

void Dataloader::reset() {
  current_batch = 0;
  if (shuffle) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
  }
}

bool Dataloader::has_next() const { return current_batch < num_batch(); }

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> Dataloader::next() {
  assert(has_next());
  auto batch = get_batch(current_batch);
  current_batch++;
  return batch;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
Dataloader::get_batch(size_t batch_id) {
  assert(batch_id < num_batch());
  size_t start_id = batch_size * batch_id,
         end_id = std::min(num_samples, batch_size + batch_id * batch_size);

  size_t actual_batch_size = end_id - start_id;
  std::vector<size_t> X_batch_shape = X->shape;
  std::vector<size_t> Y_batch_shape = Y->shape;
  X_batch_shape[0] = actual_batch_size;
  Y_batch_shape[0] = actual_batch_size;

  auto batch_X = std::make_shared<Tensor>(X_batch_shape, X->requires_grad);
  auto batch_Y = std::make_shared<Tensor>(Y_batch_shape, Y->requires_grad);

  size_t X_sample_size = X->numel() / num_samples;
  size_t Y_sample_size = Y->numel() / num_samples;

  for (size_t i = 0; i < actual_batch_size; ++i) {
    size_t sample_idx = indices[start_id + i];

    // Copy X data
    for (size_t j = 0; j < X_sample_size; ++j) {
      batch_X->data[i * X_sample_size + j] =
          X->data[sample_idx * X_sample_size + j];
    }

    // Copy Y data
    for (size_t j = 0; j < Y_sample_size; ++j) {
      batch_Y->data[i * Y_sample_size + j] =
          Y->data[sample_idx * Y_sample_size + j];
    }
  }

  return {batch_X, batch_Y};
}
