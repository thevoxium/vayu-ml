#include "../include/value.h"
#include <iostream>

int main() {
  auto a = make_value(2.0);
  auto b = make_value(-3.0);

  auto c = pow(a, 3) + b;

  auto d = relu(c);

  d->backward();

  std::cout << "a: " << *a << std::endl;
  std::cout << "b: " << *b << std::endl;
  std::cout << "c: " << *c << std::endl;
  std::cout << "d: " << *d << std::endl;

  return 0;
}
