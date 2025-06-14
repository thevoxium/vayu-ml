#include "../include/tensor.h"
#include <chrono>
#include <iostream>

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  auto a = tensor({1000, 1000}, true);
  auto b = tensor({1000, 1000}, true);
  auto c = a->mm(b);
  std::cout << *c << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Execution time: " << duration.count() << " microseconds"
            << std::endl;
  return 0;
}
