# vayu

**A lightweight deep learning framework built from scratch in C++**

[![GitHub stars](https://img.shields.io/github/stars/thevoxium/vayu-ml?style=social)](https://github.com/thevoxium/vayu-ml)


## Features

- **Automatic Differentiation**: Full backward pass support for scalars and tensors
- **Broadcasting**: NumPy-style tensor broadcasting for flexible operations
- **BLAS Optimization**: Accelerated matrix operations using OpenBLAS, MKL, or Apple Accelerate
- **Modern C++**: Built with C++17 features and smart pointers
- **Lightweight**: Minimal dependencies, easy to integrate and understand

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

### Linear Regression Example

```cpp
#include "include/tensor.h"
#include <iostream>

int main() {
    // Training data: y = 2x + 1
    auto X = tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5, 1}, false);
    auto y = tensor({3.0f, 5.0f, 7.0f, 9.0f, 11.0f}, {5, 1}, false);
    
    // Model parameters
    auto W = tensor({0.1f}, {1, 1}, true);
    auto b = tensor({0.1f}, {1, 1}, true);
    
    float learning_rate = 0.01f;
    int epochs = 200;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Zero gradients
        W->zero_grad();
        b->zero_grad();
        
        // Forward pass
        auto y_pred = X->mm(W) + b;
        auto diff = y_pred - y;
        auto squared_error = diff * diff;
        auto loss = squared_error->sum() * tensor({1.0f / y->numel()}, {1, 1}, false);
        
        // Backward pass
        loss->backward();
        
        // Update parameters
        W->data[0] -= learning_rate * W->grad[0];
        b->data[0] -= learning_rate * b->grad[0];
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss->data[0] 
                      << " | W: " << W->data[0] << " | b: " << b->data[0] << std::endl;
        }
    }
    
    return 0;
}
```

## Building

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Optional: OpenBLAS, Intel MKL, or Apple Accelerate for optimized matrix operations

### Basic Build
```bash
git clone https://github.com/thevoxium/vayu-ml.git
cd vayu-ml
g++ -std=c++17 -O2 -o grad examples/main.cpp src/*.cpp && ./grad
```

### With OpenBLAS Optimization (Recommended)

**macOS (Homebrew):**
```bash
# Install OpenBLAS
brew install openblas

# Compile with OpenBLAS
g++ -std=c++17 -O3 -DUSE_OPENBLAS -march=native \
    -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    examples/test.cpp src/tensor.cpp -lopenblas -o grad && ./grad
```

**Ubuntu/Debian:**
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev

# Compile with OpenBLAS
g++ -std=c++17 -O3 -DUSE_OPENBLAS -march=native \
    -I/usr/include/openblas \
    examples/test.cpp src/tensor.cpp -lopenblas -o grad && ./grad
```

### With Apple Accelerate Framework (macOS)
```bash
g++ -std=c++17 -O3 -march=native -framework Accelerate \
    examples/test.cpp src/tensor.cpp -o grad && ./grad
```

### With Intel MKL
```bash
g++ -std=c++17 -O3 -DUSE_MKL -march=native \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    examples/test.cpp src/tensor.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -o grad && ./grad
```


## API Reference

### Tensor Class

#### Constructor
```cpp
Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = true)
Tensor(const std::vector<size_t>& shape, bool requires_grad = true)
```

#### Factory Functions
```cpp
// Create tensor from data
auto t = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);

// Create tensor with specific shape (initialized with zeros)
auto t = tensor({3, 3}, true);

// Create random tensor
auto t = random_tensor({2, 3}, true, 0.0f, 1.0f);  // min=0.0, max=1.0

// Create tensor filled with ones
auto t = make_ones({2, 3}, true);
```

#### Basic Operations
```cpp
// Element-wise operations (with broadcasting)
auto c = a + b;    // Addition
auto c = a - b;    // Subtraction  
auto c = a * b;    // Element-wise multiplication

// Matrix multiplication
auto c = a->mm(b);              // Standard matrix multiplication
auto c = a->mm(b, true);        // BLAS-optimized (default)
auto c = a->mm(b, false);       // Non-optimized implementation
```

#### Activation Functions
```cpp
auto activated = tensor->relu();     // ReLU activation
auto activated = tensor->sigmoid();  // Sigmoid activation
```

#### Tensor Manipulation
```cpp
auto summed = tensor->sum();                    // Sum all elements
auto transposed = tensor->transpose();          // Matrix transpose
auto reshaped = tensor->reshape({4, 2});       // Reshape tensor
```

#### Gradient Operations
```cpp
tensor->backward();     // Compute gradients via backpropagation
tensor->zero_grad();    // Zero out gradients
tensor->clear_graph();  // Clear computation graph
```

#### Properties
```cpp
size_t count = tensor->numel();           // Number of elements
std::vector<size_t> dims = tensor->shape; // Tensor dimensions
std::vector<float> values = tensor->data; // Raw data access
std::vector<float> grads = tensor->grad;  // Gradient values
bool trainable = tensor->requires_grad;   // Gradient tracking flag
```

### Broadcasting Rules

Vayu follows NumPy-style broadcasting rules:

1. **Trailing dimensions alignment**: Dimensions are aligned from the rightmost
2. **Size compatibility**: Dimensions are compatible if they are equal, or one of them is 1
3. **Missing dimensions**: Missing dimensions are treated as size 1

Examples:
```cpp
// Compatible shapes
(2, 3) + (1, 3) → (2, 3)    // Broadcast first dimension
(4, 1) * (4, 5) → (4, 5)    // Broadcast second dimension  
(3, 1, 2) + (1, 4, 1) → (3, 4, 2)  // Broadcast multiple dimensions

// Check compatibility
bool compatible = Tensor::can_broadcast({2, 3}, {1, 3});  // true
auto result_shape = Tensor::broadcast_shape({2, 3}, {1, 3});  // {2, 3}
```



## Performance Tips

1. **Use BLAS optimization**: Always compile with `-DUSE_OPENBLAS` or equivalent for matrix operations
2. **Enable compiler optimizations**: Use `-O3 -march=native` for best performance
3. **Batch operations**: Perform operations on larger tensors when possible
4. **Memory management**: Use `clear_graph()` to free computation graphs when not needed
5. **Gradient accumulation**: Call `zero_grad()` before each backward pass in training loops

## Examples

See the `examples/` directory for more complete examples:

- `examples/test.cpp` - Basic tensor operations and broadcasting
- `examples/linear_regression_test.cpp` - Linear regression implementation
- `examples/blas_mm.cpp` - Matrix multiplication benchmarks




## License

MIT License 
