
#include <chrono>
#include <iostream>

#define START_TIMER(name)                                                      \
  auto start_##name = std::chrono::high_resolution_clock::now();
#define END_TIMER(name)                                                        \
  auto end_##name = std::chrono::high_resolution_clock::now();
#define PRINT_TIMER(name)                                                      \
  std::cout << #name " took "                                                  \
            << std::chrono::duration_cast<std::chrono::microseconds>(          \
                   end_##name - start_##name)                                  \
                   .count()                                                    \
            << " microseconds.\n";
