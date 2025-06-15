#include "../include/tensor.h"
#include <iomanip>
#include <ios>
#include <iostream>
#include <ostream>

int main() {
  // XOR dataset
  // Inputs: [x1, x2]
  auto X = tensor({0.0f, 0.0f,  // XOR(0,0) = 0
                   0.0f, 1.0f,  // XOR(0,1) = 1
                   1.0f, 0.0f,  // XOR(1,0) = 1
                   1.0f, 1.0f}, // XOR(1,1) = 0
                  {4, 2}, false);

  // Target outputs
  auto y = tensor({0.0f,  // XOR(0,0) = 0
                   1.0f,  // XOR(0,1) = 1
                   1.0f,  // XOR(1,0) = 1
                   0.0f}, // XOR(1,1) = 0
                  {4, 1}, false);

  // Network architecture: 2 -> 4 -> 1
  // Input to hidden layer weights (2x4) and bias (1x4)
  auto W1 = random_tensor({2, 4}, true, -1.0f, 1.0f);
  auto b1 = tensor({0.0f, 0.0f, 0.0f, 0.0f}, {1, 4}, true);

  // Hidden to output layer weights (4x1) and bias (1x1)
  auto W2 = random_tensor({4, 1}, true, -1.0f, 1.0f);
  auto b2 = tensor({0.0f}, {1, 1}, true);

  float learning_rate = 0.5f;
  int epochs = 2000;

  std::cout << "Training Neural Network for XOR problem..." << std::endl;
  std::cout << "Architecture: 2 -> 4 -> 1" << std::endl;
  std::cout << "Dataset: XOR truth table" << std::endl << std::endl;

  for (int epoch = 0; epoch < epochs; epoch++) {
    // Clear gradients
    W1->zero_grad();
    b1->zero_grad();
    W2->zero_grad();
    b2->zero_grad();

    // Forward pass
    // Hidden layer: h = ReLU(X @ W1 + b1)
    auto z1 = X->mm(W1) + b1; // Linear transformation
    auto h = z1->relu();      // ReLU activation

    // Output layer: y_pred = sigmoid(h @ W2 + b2)
    auto z2 = h->mm(W2) + b2;    // Linear transformation
    auto y_pred = z2->sigmoid(); // Sigmoid activation

    // Compute binary cross-entropy loss
    // For simplicity, we'll use MSE loss instead of BCE
    auto diff = y_pred - y;
    auto squared_error = diff * diff;
    auto total_loss = squared_error->sum();
    auto loss =
        total_loss * tensor({0.25f}, {1, 1}, false); // Mean over 4 samples

    // Backward pass
    loss->backward();

    // Gradient descent update
    for (size_t i = 0; i < W1->numel(); i++) {
      W1->data[i] -= learning_rate * W1->grad[i];
    }
    for (size_t i = 0; i < b1->numel(); i++) {
      b1->data[i] -= learning_rate * b1->grad[i];
    }
    for (size_t i = 0; i < W2->numel(); i++) {
      W2->data[i] -= learning_rate * W2->grad[i];
    }
    for (size_t i = 0; i < b2->numel(); i++) {
      b2->data[i] -= learning_rate * b2->grad[i];
    }

    // Print progress
    if (epoch % 200 == 0) {
      std::cout << "Epoch " << std::setw(4) << epoch
                << " | Loss: " << std::fixed << std::setprecision(6)
                << loss->data[0] << std::endl;
    }
  }

  std::cout << "\nTraining completed!" << std::endl;
  std::cout << "\nTesting the trained network:" << std::endl;
  std::cout << "Input -> Target | Prediction | Rounded" << std::endl;
  std::cout << "--------------------------------" << std::endl;

  // Test the trained network
  for (int i = 0; i < 4; i++) {
    // Create individual test inputs
    auto test_input =
        tensor({X->data[i * 2], X->data[i * 2 + 1]}, {1, 2}, false);

    // Forward pass
    auto z1_test = test_input->mm(W1) + b1;
    auto h_test = z1_test->relu();
    auto z2_test = h_test->mm(W2) + b2;
    auto pred_test = z2_test->sigmoid();

    float prediction = pred_test->data[0];
    int rounded = (prediction > 0.5) ? 1 : 0;

    std::cout << "(" << X->data[i * 2] << "," << X->data[i * 2 + 1] << ") -> "
              << y->data[i] << "   | " << std::fixed << std::setprecision(4)
              << prediction << "     | " << rounded << std::endl;
  }

  // Calculate accuracy
  int correct = 0;
  for (int i = 0; i < 4; i++) {
    auto test_input =
        tensor({X->data[i * 2], X->data[i * 2 + 1]}, {1, 2}, false);
    auto z1_test = test_input->mm(W1) + b1;
    auto h_test = z1_test->relu();
    auto z2_test = h_test->mm(W2) + b2;
    auto pred_test = z2_test->sigmoid();

    int predicted = (pred_test->data[0] > 0.5) ? 1 : 0;
    int target = static_cast<int>(y->data[i]);

    if (predicted == target)
      correct++;
  }

  std::cout << "\nAccuracy: " << correct << "/4 (" << (correct * 100.0f / 4.0f)
            << "%)" << std::endl;

  return 0;
}
