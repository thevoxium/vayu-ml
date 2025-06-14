#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({-0.1, 0.2}, {2, 1}, true);
  auto b = tensor({0.2, 0.3}, {2, 1}, true);
  auto c = a * b;
  auto d = c->sigmoid();
  auto e = a->transpose();
  std::cout << *e << std::endl;
  std::cout << (*a)[0];
  return 0;
}
