#include "../include/tensor.h"
#include <iostream>
#include <ostream>

int main() {
  auto X = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1}, false);
  auto y = tensor({3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {5, 1}, false);

  auto W = tensor({0.1f}, {1, 1}, true);
  auto b = tensor({0.1f}, {1, 1}, true);

  float learning_rate = 0.01f;
  int epochs = 200;

  std::cout << "Training Linear Regression..." << std::endl;
  std::cout << "Target: y = 2x + 1" << std::endl;

  for (int epoch = 0; epoch < epochs; epoch++) {
    W->zero_grad();
    b->zero_grad();

    auto y_pred = X->mm(W) + b;

    auto diff = y_pred - y;
    auto squared_error = diff * diff;
    auto total_loss = squared_error->sum();

    auto loss = total_loss * tensor({1.0f / y->numel()}, {1, 1}, false);

    loss->backward();

    W->data[0] -= learning_rate * W->grad[0];
    b->data[0] -= learning_rate * b->grad[0];

    if (epoch % 20 == 0) {
      std::cout << "Epoch " << epoch << " | Loss: " << loss->data[0]
                << " | W: " << W->data[0] << " | b: " << b->data[0]
                << std::endl;
    }
  }

  std::cout << "W = " << W->data[0] << " (target: 2.0)" << std::endl;
  std::cout << "b = " << b->data[0] << " (target: 1.0)" << std::endl;

  return 0;
}
