#include "../include/value.h"
#include <iostream>

int main() {
  auto a = make_value(2.0);
  auto b = make_value(-3.0);

  auto c = a * b * a;

  c->backward();

  std::cout << "a: " << *a << std::endl;
  std::cout << "b: " << *b << std::endl;
  std::cout << "c: " << *c << std::endl;

  return 0;
}
