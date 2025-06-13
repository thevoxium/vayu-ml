#include "../include/value.h"
#include <iostream>

int main() {
  auto a = make_value(2.0);
  auto b = make_value(-3.0);

  auto c = a + b;

  c->backward();

  std::cout << *c << std::endl;

  return 0;
}
