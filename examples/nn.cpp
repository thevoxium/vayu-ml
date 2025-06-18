#include "../include/vayu.h"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>

int main() {
  // Much larger dataset - 10x more samples
  auto x = make_const({10000, 1024}, 0.5f, false);
  auto y = make_const({10000, 512}, 1.0f, false); // Larger output too

  Sequential model;
  // Much larger model with more layers and parameters
  model.add(Linear(1024, 2048)); // ~2M params
  model.add(Relu());
  model.add(Linear(2048, 2048)); // ~4M params
  model.add(Relu());
  model.add(Linear(2048, 1536)); // ~3M params
  model.add(Relu());
  model.add(Linear(1536, 1024)); // ~1.5M params
  model.add(Relu());
  model.add(Linear(1024, 512)); // ~0.5M params
  model.add(Sigmoid());

  size_t epochs = 100; // More epochs = more SGD steps
  size_t batch_size = 16;
  Dataloader dataloader(x, y, batch_size,
                        true); // Smaller batches = more SGD steps per epoch

  int tot = 0;
  auto params = model.parameters();
  for (auto &param : params) {
    tot += param->numel();
  }

  size_t total_batches = dataloader.num_batch();
  size_t total_samples = x->shape[0];

  std::cout << "=== Training Configuration ===" << std::endl;
  std::cout << "Total parameters: " << tot << std::endl;
  std::cout << "Total samples: " << total_samples << std::endl;
  std::cout << "Batch size: " << batch_size << std::endl;
  std::cout << "Batches per epoch: " << total_batches << std::endl;
  std::cout << "Total epochs: " << epochs << std::endl;
  std::cout << "Total batches: " << epochs * total_batches << std::endl;
  std::cout << "==============================" << std::endl << std::endl;

  SGD optim(model.parameters(), 0.001f);

  auto training_start = std::chrono::high_resolution_clock::now();
  size_t global_batch_count = 0;

  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    auto epoch_start = std::chrono::high_resolution_clock::now();
    std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs << " ";

    dataloader.reset();
    size_t batch_in_epoch = 0;

    while (dataloader.has_next()) {
      auto batch_start = std::chrono::high_resolution_clock::now();

      auto batch = dataloader.next();
      auto y_pred = model(batch.first);
      auto loss = mse_loss(y_pred, batch.second);
      loss->backward();
      optim.step();
      optim.zero_grad();

      batch_in_epoch++;
      global_batch_count++;

      // Print batch progress every 50 batches or at the end of epoch
      if (batch_in_epoch % 50 == 0 || batch_in_epoch == total_batches) {
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(batch_end -
                                                                  batch_start);

        // Calculate samples per second for this batch
        double samples_per_sec = (batch_size * 1000.0) / batch_duration.count();

        std::cout << "[" << std::setw(3) << batch_in_epoch << "/"
                  << total_batches << "] " << std::fixed << std::setprecision(1)
                  << samples_per_sec << " samples/sec";

        if (batch_in_epoch == total_batches) {
          std::cout << " - Complete";
        }
        std::cout << std::endl;
      }
    }

    auto epoch_end = std::chrono::high_resolution_clock::now();
    auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        epoch_end - epoch_start);

    // Calculate epoch-level metrics
    double epoch_samples_per_sec =
        (total_samples * 1000.0) / epoch_duration.count();
    double avg_time_per_batch =
        static_cast<double>(epoch_duration.count()) / total_batches;

    // Calculate ETA
    auto training_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        epoch_end - training_start);
    double avg_epoch_time =
        static_cast<double>(training_elapsed.count()) / (epoch + 1);
    size_t remaining_epochs = epochs - epoch - 1;
    size_t eta_seconds = static_cast<size_t>(avg_epoch_time * remaining_epochs);

    std::cout << "    Time: " << epoch_duration.count() << "ms"
              << " | " << std::fixed << std::setprecision(1)
              << avg_time_per_batch << "ms/batch"
              << " | " << std::setprecision(0) << epoch_samples_per_sec
              << " samples/sec";

    if (remaining_epochs > 0) {
      std::cout << " | ETA: " << eta_seconds / 60 << "m " << eta_seconds % 60
                << "s";
    }
    std::cout << std::endl;

    // Print loss every 10 epochs
    if (epoch % 10 == 0 || epoch == epochs - 1) {
      auto sample_pred = model(x);
      auto sample_loss = mse_loss(sample_pred, y);
      std::cout << "    Loss: " << std::scientific << std::setprecision(6)
                << sample_loss->data[0] << std::endl;
    }
    std::cout << std::endl;
  }

  auto training_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      training_end - training_start);

  std::cout << "=== Training Complete ===" << std::endl;
  std::cout << "Total training time: " << total_duration.count() / 1000000.0
            << " seconds" << std::endl;

  // Calculate comprehensive performance metrics
  size_t total_sgd_ops = static_cast<size_t>(tot) * epochs * total_batches;
  double ops_per_second =
      static_cast<double>(total_sgd_ops) / (total_duration.count() / 1000000.0);
  double total_samples_processed = epochs * total_samples;
  double overall_samples_per_sec =
      total_samples_processed / (total_duration.count() / 1000000.0);

  std::cout << "SGD operations per second: " << std::fixed
            << std::setprecision(0) << ops_per_second << std::endl;
  std::cout << "Overall samples per second: " << overall_samples_per_sec
            << std::endl;
  std::cout << "Average time per epoch: "
            << (total_duration.count() / 1000000.0) / epochs << " seconds"
            << std::endl;
  std::cout << "Total batches processed: " << global_batch_count << std::endl;

  return 0;
}
