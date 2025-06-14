#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({0.2, 0.1}, {2, 1}, true);
  std::cout << *a << std::endl;
  return 0;
}
