#include "../include/tensor.h"
#include <iostream>

int main() {
  auto a = tensor({2, 1}, true);
  std::cout << *a << std::endl;
  return 0;
}
