#include "../include/tensor.h"
#include <ios>
#include <iostream>

int main() {
  auto a = tensor({-0.1, 0.2}, {1, 2}, true);
  auto b = tensor({2, 5}, true);
  std::cout << std::boolalpha
            << Tensor::is_broadcast_possible(a->shape, b->shape) << std::endl;
  /*auto c = a * b;*/
  /*auto d = c->sigmoid();*/
  /*auto e = a->transpose();*/
  /*std::cout << *e << std::endl;*/
  /*std::cout << (*a)[0];*/
  return 0;
}
