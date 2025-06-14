#include "../include/tensor.h"
#include <ios>
#include <iostream>
#include <ostream>

int main() {
  auto a = tensor({2, 2}, true);
  auto b = tensor({2, 1}, true);
  std::cout << std::boolalpha << Tensor::can_broadcast(a->shape, b->shape)
            << std::endl;

  auto c = a + b;
  c->backward();
  std::cout << *a << std::endl;
  std::cout << *b << std::endl;
  std::cout << *c << std::endl;
  for (auto x : a->grad) {
    std::cout << x << ", ";
  }
  return 0;
}
