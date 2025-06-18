#include "../include/vayu.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Path to your MNIST CSV files
const std::string MNIST_DIR = "/Users/anshul/Downloads/MNIST_CSV/";
const std::string TRAIN_CSV = "mnist_train.csv";
const std::string TEST_CSV = "mnist_test.csv";

// Helper function to format time duration
std::string format_duration(std::chrono::duration<double> duration) {
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() %
      1000;
  return std::to_string(seconds) + "." + std::to_string(milliseconds) + "s";
}

// Helper function to print performance metrics
void print_metrics(const std::string &prefix, size_t samples,
                   std::chrono::duration<double> duration, float loss) {
  double seconds = duration.count();
  std::cout << prefix << ": "
            << "Loss = " << std::fixed << std::setprecision(6) << loss
            << ", Time = " << format_duration(duration)
            << ", Samples/sec = " << std::fixed << std::setprecision(2)
            << (samples / seconds) << std::endl;
}

// Function to load MNIST dataset from CSV files
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
load_mnist_csv(const std::string &filename, bool normalize = true) {
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    exit(1);
  }

  std::cout << "Loading data from " << filename << "..." << std::endl;

  std::string line;
  std::vector<float> feature_data;
  std::vector<float> label_data;

  size_t num_samples = 0;
  size_t num_features = 0;
  size_t num_classes = 10; // MNIST has 10 classes (digits 0-9)

  // Count samples first to pre-allocate
  while (std::getline(file, line)) {
    num_samples++;
  }

  std::cout << "Found " << num_samples << " samples." << std::endl;

  // Reset file to beginning
  file.clear();
  file.seekg(0);

  // Pre-allocate memory
  num_features = 28 * 28; // MNIST is 28x28 pixels
  feature_data.reserve(num_samples * num_features);
  label_data.resize(num_samples * num_classes, 0.0f);

  size_t line_count = 0;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;

    // First column is the label
    std::getline(ss, cell, ',');
    int label = std::stoi(cell);

    // One-hot encode the label
    label_data[line_count * num_classes + label] = 1.0f;

    // Rest are pixel values
    size_t pixel_idx = 0;
    while (std::getline(ss, cell, ',') && pixel_idx < num_features) {
      float pixel_value = std::stof(cell);
      if (normalize) {
        pixel_value /= 255.0f;
      }
      feature_data.push_back(pixel_value);
      pixel_idx++;
    }

    line_count++;

    if (line_count % 10000 == 0 || line_count == num_samples) {
      std::cout << "  Processed " << line_count << "/" << num_samples
                << " samples..." << std::endl;
    }
  }

  std::cout << "Loaded " << line_count << " samples with " << num_features
            << " features." << std::endl;

  // Create tensors
  auto X = std::make_shared<Tensor>(
      feature_data, std::vector<size_t>{num_samples, num_features}, false);
  auto Y = std::make_shared<Tensor>(
      label_data, std::vector<size_t>{num_samples, num_classes}, false);

  return {X, Y};
}

// Function to evaluate the model on the test set
void evaluate_model(Sequential &model, std::shared_ptr<Tensor> X_test,
                    std::shared_ptr<Tensor> Y_test, size_t batch_size = 100) {
  size_t num_correct = 0;
  size_t num_samples = X_test->shape[0];
  float total_loss = 0.0f;

  // Create dataloader for evaluation
  Dataloader eval_loader(X_test, Y_test, batch_size, false);
  eval_loader.reset();

  while (eval_loader.has_next()) {
    auto [X_batch, Y_batch] = eval_loader.next();

    // Forward pass
    auto Y_pred = model(X_batch);
    auto loss = mse_loss(Y_pred, Y_batch);
    total_loss += loss->data[0] * X_batch->shape[0];

    // Count correct predictions
    for (size_t j = 0; j < X_batch->shape[0]; ++j) {
      size_t true_label = 0;
      size_t pred_label = 0;
      float max_true = Y_batch->data[j * 10];
      float max_pred = Y_pred->data[j * 10];

      for (size_t k = 1; k < 10; ++k) {
        if (Y_batch->data[j * 10 + k] > max_true) {
          max_true = Y_batch->data[j * 10 + k];
          true_label = k;
        }

        if (Y_pred->data[j * 10 + k] > max_pred) {
          max_pred = Y_pred->data[j * 10 + k];
          pred_label = k;
        }
      }

      if (true_label == pred_label) {
        num_correct++;
      }
    }
  }

  float accuracy =
      static_cast<float>(num_correct) / static_cast<float>(num_samples);
  float avg_loss = total_loss / static_cast<float>(num_samples);

  std::cout << "Test set - Loss: " << std::fixed << std::setprecision(6)
            << avg_loss << ", Accuracy: " << std::fixed << std::setprecision(4)
            << (accuracy * 100.0f) << "%" << std::endl;
}

int main() {
  // Load MNIST dataset from CSV files
  std::cout << "Loading MNIST dataset from CSV files..." << std::endl;
  auto [X_train, Y_train] = load_mnist_csv(MNIST_DIR + TRAIN_CSV);
  auto [X_test, Y_test] = load_mnist_csv(MNIST_DIR + TEST_CSV);

  std::cout << "Training set: " << X_train->shape[0]
            << " images, Test set: " << X_test->shape[0] << " images"
            << std::endl;

  // Configure training parameters
  const size_t input_dim = 784; // 28x28 pixels
  const size_t hidden1_dim = 1024;
  const size_t hidden2_dim = 512;
  const size_t hidden3_dim = 256;
  const size_t hidden4_dim = 128;
  const size_t output_dim = 10; // 10 digit classes

  const size_t batch_size = 128;
  const size_t num_epochs = 5;
  const float learning_rate = 0.01f;

  std::cout << "Creating neural network for MNIST classification..."
            << std::endl;

  // Create the model
  Sequential model;
  model.add(Linear(input_dim, hidden1_dim));
  model.add(Relu());
  model.add(Linear(hidden1_dim, hidden2_dim));
  model.add(Relu());
  model.add(Linear(hidden2_dim, hidden3_dim));
  model.add(Relu());
  model.add(Linear(hidden3_dim, hidden4_dim));
  model.add(Relu());
  model.add(Linear(hidden4_dim, output_dim));

  // Calculate total parameters
  size_t total_params = 0;
  for (auto &param : model.parameters()) {
    total_params += param->numel();
  }
  std::cout << "Total parameters: " << total_params << std::endl;

  // Create dataloader
  Dataloader dataloader(X_train, Y_train, batch_size, true);

  // Create optimizer
  SGD optimizer(model.parameters(), learning_rate);

  std::cout << "Starting training for " << num_epochs << " epochs..."
            << std::endl;
  std::cout << "----------------------------------------------------"
            << std::endl;

  // Track overall statistics
  auto training_start = std::chrono::high_resolution_clock::now();
  float avg_batch_time = 0.0f;
  size_t batch_count = 0;

  // Training loop
  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    // Reset dataloader for new epoch
    dataloader.reset();

    // Track epoch statistics
    float epoch_loss = 0.0f;
    size_t num_batches = 0;
    auto epoch_start = std::chrono::high_resolution_clock::now();

    // Batch loop
    while (dataloader.has_next()) {
      auto batch_start = std::chrono::high_resolution_clock::now();

      // Get next batch
      auto [X_batch, Y_batch] = dataloader.next();

      // Forward pass
      optimizer.zero_grad();
      auto Y_pred = model(X_batch);
      auto loss = mse_loss(Y_pred, Y_batch);

      // Backward pass and optimize
      loss->backward();
      optimizer.step();

      // Update statistics
      float batch_loss = loss->data[0];
      epoch_loss += batch_loss;
      num_batches++;

      // Measure batch time
      auto batch_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> batch_duration = batch_end - batch_start;
      avg_batch_time = (avg_batch_time * batch_count + batch_duration.count()) /
                       (batch_count + 1);
      batch_count++;

      // Print batch metrics every 50 batches
      if (num_batches % 50 == 0) {
        std::cout << "  Batch " << std::setw(4) << num_batches
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << batch_loss << ", Time: " << std::fixed
                  << std::setprecision(3) << (batch_duration.count() * 1000)
                  << "ms" << std::endl;
      }
    }

    // Calculate epoch metrics
    auto epoch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
    float avg_loss = epoch_loss / num_batches;

    // Print epoch metrics
    std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
              << ", Avg Loss: " << std::fixed << std::setprecision(6)
              << avg_loss << ", Time: " << format_duration(epoch_duration)
              << ", Samples/sec: " << std::fixed << std::setprecision(2)
              << (X_train->shape[0] / epoch_duration.count()) << std::endl;

    // Evaluate on test data after each epoch
    std::cout << "Evaluating on test set..." << std::endl;
    evaluate_model(model, X_test, Y_test, batch_size);
  }

  // Calculate overall metrics
  auto training_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = training_end - training_start;

  // Print summary
  std::cout << "----------------------------------------------------"
            << std::endl;
  std::cout << "Training completed!" << std::endl;
  std::cout << "Total training time: " << format_duration(total_duration)
            << std::endl;
  std::cout << "Average time per epoch: "
            << format_duration(total_duration / num_epochs) << std::endl;
  std::cout << "Average time per batch: " << std::fixed << std::setprecision(3)
            << (avg_batch_time * 1000) << "ms" << std::endl;
  std::cout << "Average samples/sec: " << std::fixed << std::setprecision(2)
            << (X_train->shape[0] * num_epochs / total_duration.count())
            << std::endl;

  // Final evaluation on test set
  std::cout << "----------------------------------------------------"
            << std::endl;
  std::cout << "Final evaluation on test set..." << std::endl;
  evaluate_model(model, X_test, Y_test, batch_size);

  return 0;
}
