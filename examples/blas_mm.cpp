#include "../include/tensor.h"
#include <chrono>
#include <iomanip>
#include <iostream>

long long benchmark_mm(size_t size, bool use_fast) {
  auto a = tensor({size, size}, true);
  auto b = tensor({size, size}, true);

  // Initialize with some values
  for (size_t i = 0; i < a->numel(); i++) {
    a->data[i] = static_cast<float>(i % 10) / 10.0f;
    b->data[i] = static_cast<float>((i + 1) % 10) / 10.0f;
  }

  auto start = std::chrono::high_resolution_clock::now();
  auto c = a->mm(b, use_fast);
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

int main() {
  std::vector<size_t> sizes = {10, 100, 256, 512, 1024, 2048};

  std::cout << std::setw(6) << "Size" << std::setw(12) << "Non-BLAS"
            << std::setw(12) << "BLAS" << std::setw(10) << "Speedup"
            << std::endl;
  std::cout << std::string(40, '-') << std::endl;

  for (size_t size : sizes) {
    long long nonblas_time = benchmark_mm(size, false);
    long long blas_time = benchmark_mm(size, true);
    double speedup = static_cast<double>(nonblas_time) / blas_time;

    std::cout << std::setw(6) << size << std::setw(10) << nonblas_time << "μs"
              << std::setw(10) << blas_time << "μs" << std::setw(8)
              << std::fixed << std::setprecision(1) << speedup << "x"
              << std::endl;
  }

  return 0;
}
