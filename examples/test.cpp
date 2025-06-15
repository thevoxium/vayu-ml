#include "../include/tensor.h"
#include <ios>
#include <iostream>
#include <ostream>

int main() {
  auto a = make_ones({2, 2}, true);
  auto b = make_ones({2, 1}, true);
  std::cout << std::boolalpha << Tensor::can_broadcast(a->shape, b->shape)
            << std::endl;

  auto c = a - b;
  auto d = a * c;
  std::cout << *d << std::endl;
  return 0;
}
