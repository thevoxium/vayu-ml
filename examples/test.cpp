#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({-0.1, 0.2}, {2, 1}, true);
  auto b = tensor({0.2, 0.3}, {2, 1}, true);
  auto c = a * b;
  auto d = c->sigmoid();
  std::cout << *d << std::endl;
  return 0;
}
