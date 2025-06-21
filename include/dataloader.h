#ifndef DATALOADER_H
#define DATALOADER_H

#include "tensor.h"
#include "tokenizer.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
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

class TextDataloader {
private:
  std::string text;
  size_t context_length;
  BPE tokenizer;
  std::vector<std::vector<float>> x, y;
  size_t train_steps;

public:
  TextDataloader(std::string text, size_t context_length, size_t train_steps)
      : text(text), context_length(context_length), train_steps(train_steps) {

    tokenizer.train(text, train_steps);
    auto tokens = tokenizer.encode(text);
    for (size_t i = 0; i < tokens.size() - (context_length - 1); ++i) {
      std::vector<float> context_inp;
      for (size_t j = i; j < i + 4; ++j) {
        context_inp.push_back(1.0f * tokens[j]);
      }
      x.push_back(context_inp);
      if (i >= 1) {
        y.push_back(context_inp);
      }
    }
  }
  size_t len() { return x.size(); }
  std::pair<std::vector<float>, std::vector<float>> get_item(int id) {
    return std::make_pair(x[id], y[id]);
  }
  const std::vector<std::vector<float>> get_input() { return x; }
  const std::vector<std::vector<float>> get_target() { return y; }
};

#endif // DATALOADER_H
