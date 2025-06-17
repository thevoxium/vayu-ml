#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

class Dataloader {
private:
  std::shared_ptr<Tensor> X, Y;
  bool shuffle;
  size_t batch_size, num_samples;
  std::vector<size_t> indices;
  size_t current_batch;

public:
  Dataloader(std::shared_ptr<Tensor> X, std::shared_ptr<Tensor> Y,
             size_t batch_size, bool shuffle = true);

  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> next();

  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
  get_batch(size_t batch_id);

  size_t num_batch() const;
  void reset();
  bool has_next() const;
};

#endif // DATALOADER_H
