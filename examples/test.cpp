#include "../include/tensor.h"
#include <ios>
#include <iostream>
#include <ostream>

int main() {
  auto a = tensor({-0.1, 0.2}, {2, 1}, true);
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
  /*auto c = a * b;*/
  /*auto d = c->sigmoid();*/
  /*auto e = a->transpose();*/
  /*std::cout << *e << std::endl;*/
  /*std::cout << (*a)[0];*/
  return 0;
}
