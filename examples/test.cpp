#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({0.2, 0.1}, {2, 1}, true);
  auto b = tensor({0.3, 0.2}, {1, 2}, true);
  auto c = a->mm(b);
  std::cout << *c << std::endl;
  return 0;
}
