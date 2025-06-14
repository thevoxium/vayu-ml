#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({0.2, 0.1}, {2, 1}, true);
  auto b = tensor({0.3, 0.2}, {2, 1}, true);
  auto c = a + b;
  std::cout << *c << std::endl;
  return 0;
}
