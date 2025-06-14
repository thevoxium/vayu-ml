# vayu

**A lightweight deep learning framework built from scratch in C++ [WIP]**

[![GitHub stars](https://img.shields.io/github/stars/thevoxium/vayu-ml?style=social)](https://github.com/thevoxium/vayu-ml)


## Quick Start

### Scalar Automatic Differentiation

```cpp
#include "include/value.h"
#include <iostream>

int main() {
    auto a = make_value(2.0);
    auto b = make_value(-3.0);
    auto c = pow(a, 3) - b;  // c = a³ - b
    auto d = c / b;          // d = c / b  
    auto e = -d;             // e = -d

    e->backward();           // compute gradients

    std::cout << "a: " << *a << std::endl;  // Value(data=2, grad=4)
    std::cout << "b: " << *b << std::endl;  // Value(data=-3, grad=0.888889)
    
    return 0;
}
```

### Tensor Operations with Broadcasting

```cpp
#include "include/tensor.h"
#include <iostream>

int main() {
    auto a = random_tensor({2, 2}, true);     // 2x2 random tensor
    auto b = make_ones({2, 1}, true);         // 2x1 ones tensor
    
    auto c = a + b;                           // Broadcasting addition
    auto d = a * c;                           // Element-wise multiplication
    
    d->backward();                            // Compute gradients
    
    // Access gradients
    for (auto grad : a->grad) {
        std::cout << grad << ", ";
    }
    
    return 0;
}
```

## Building

### Basic Build
```bash
git clone https://github.com/thevoxium/vayu-ml.git
cd vayu-ml
g++ -std=c++17 -O2 -o grad examples/main.cpp src/*.cpp && ./grad
```

### With BLAS Optimization (macOS)
```bash
g++ -std=c++17 -O3 -march=native -DUSE_OPENBLAS \
    -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    examples/test.cpp src/tensor.cpp -lopenblas -o grad && ./grad
```

### With Accelerate Framework (macOS)
```bash
g++ -std=c++17 -O3 -march=native -framework Accelerate \
    examples/test.cpp src/tensor.cpp -o grad && ./grad
```

### BLAS Performance Benchmark
Compare optimized vs non-optimized matrix multiplication:

```bash
g++ -std=c++17 -O3 -DUSE_OPENBLAS examples/blas_mm.cpp src/tensor.cpp -lopenblas -o bench && ./bench
```

Expected output shows significant speedups for larger matrices:
```
  Size   Non-BLAS       BLAS   Speedup
----------------------------------------
    10       12μs        8μs      1.5x
   100      156μs       45μs      3.5x
   256     1250μs      185μs      6.8x
   512     9800μs      890μs     11.0x
  1024    78000μs     4200μs     18.6x
  2048   620000μs    28000μs     22.1x
```

## API Overview

### Value Class (Scalar Operations)
- `make_value(double)` - Create a scalar value
- `+`, `-`, `*`, `/` - Basic arithmetic with automatic differentiation
- `pow(value, exponent)` - Power operation
- `relu()` - ReLU activation function
- `backward()` - Compute gradients via backpropagation

### Tensor Class (Multi-dimensional Arrays)
- `tensor(data, shape, requires_grad)` - Create tensor from data
- `random_tensor(shape)` - Create random tensor
- `make_ones(shape)` - Create tensor filled with ones
- `+`, `*` - Element-wise operations with broadcasting
- `mm(other, fast=true)` - Matrix multiplication (BLAS-optimized)
- `relu()`, `sigmoid()` - Activation functions
- `sum()`, `transpose()` - Reduction and transformation operations
- `backward()` - Automatic differentiation

### Broadcasting Support
Tensors automatically broadcast compatible shapes:
- `(2, 3) + (1, 3)` → `(2, 3)`
- `(4, 1) * (4, 5)` → `(4, 5)`


## Contributing

Contributions are welcome. Open a PR with your improvements!

## License

MIT License - feel free to use this for learning and teaching automatic differentiation concepts.
