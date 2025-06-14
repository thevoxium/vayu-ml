#include "../include/tensor.h"
#include <iostream>
#include <ostream>

int main() {
  // Simple synthetic data: y = 2x + 1
  auto X = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1}, false);
  auto y = tensor({3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {5, 1}, false);

  // Initialize parameters
  auto W = tensor({0.1f}, {1, 1}, true); // weight
  auto b = tensor({0.1f}, {1, 1}, true); // bias

  float learning_rate = 0.01f;
  int epochs = 200;

  std::cout << "Training Linear Regression..." << std::endl;
  std::cout << "Target: y = 2x + 1" << std::endl;

  for (int epoch = 0; epoch < epochs; epoch++) {
    // Clear gradients
    std::fill(W->grad.begin(), W->grad.end(), 0.0f);
    std::fill(b->grad.begin(), b->grad.end(), 0.0f);

    // Manual forward pass and gradient computation
    float total_loss = 0.0f;

    // Compute loss and gradients manually for each sample
    for (size_t i = 0; i < X->numel(); i++) {
      // Forward: pred = W * X[i] + b
      float pred = W->data[0] * X->data[i] + b->data[0];

      // Loss: (pred - y[i])^2
      float error = pred - y->data[i];
      total_loss += error * error;

      // Gradients: d_loss/d_W = 2 * error * X[i], d_loss/d_b = 2 * error
      W->grad[0] += 2.0f * error * X->data[i];
      b->grad[0] += 2.0f * error;
    }

    // Average the loss and gradients
    total_loss /= X->numel();
    W->grad[0] /= X->numel();
    b->grad[0] /= X->numel();

    // Gradient descent update
    W->data[0] -= learning_rate * W->grad[0];
    b->data[0] -= learning_rate * b->grad[0];

    // Print progress
    if (epoch % 20 == 0) {
      std::cout << "Epoch " << epoch << " | Loss: " << total_loss
                << " | W: " << W->data[0] << " | b: " << b->data[0]
                << std::endl;
    }
  }

  std::cout << "\nFinal Results:" << std::endl;
  std::cout << "W = " << W->data[0] << " (target: 2.0)" << std::endl;
  std::cout << "b = " << b->data[0] << " (target: 1.0)" << std::endl;

  // Test prediction
  float test_x = 6.0f;
  float prediction = W->data[0] * test_x + b->data[0];
  std::cout << "Prediction for x=6: " << prediction << " (expected: 13)"
            << std::endl;

  return 0;
}
