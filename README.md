# vayu

**A lightweight deep learning framework built from scratch in C++**

[![GitHub stars](https://img.shields.io/github/stars/thevoxium/vayu-ml?style=social)](https://github.com/thevoxium/vayu-ml)

## Quick Start

### Neural Network Training Example

```cpp
#include "include/vayu.h"
#include <iostream>

int main() {
  // Create random dataset
  auto x = random_tensor({1000, 784}, false);
  auto y = random_tensor({1000, 1}, false);
  
  // Create model
  Sequential model;
  model.add(Linear(784, 512));
  model.add(Relu());
  model.add(Linear(512, 256));
  model.add(Relu());
  model.add(Linear(256, 1));
  model.add(Sigmoid());
  
  // Training parameters
  size_t epochs = 100;
  float learning_rate = 0.01f;
  
  // Create dataloader with batch size 32
  Dataloader dataloader(x, y, 32, true);
  
  // Training loop
  for (size_t epoch = 0; epoch < epochs; ++epoch) {
    dataloader.reset();
    
    while (dataloader.has_next()) {
      // Get batch
      auto batch = dataloader.next();
      auto batch_x = batch.first;
      auto batch_y = batch.second;
      
      // Forward pass
      auto y_pred = model(batch_x);
      
      // Compute MSE loss
      auto loss = mse_loss(y_pred, batch_y);
      
      // Backward pass
      model.zero_grad();
      loss->backward();
      
      // Simple parameter update (gradient descent)
      auto params = model.parameters();
      for (auto& param : params) {
        for (size_t i = 0; i < param->numel(); ++i) {
          param->data[i] -= learning_rate * param->grad[i];
        }
      }
    }
    
    // Print progress
    if (epoch % 10 == 0) {
      auto sample_pred = model(x);
      auto sample_loss = mse_loss(sample_pred, y);
      std::cout << "Epoch " << epoch << ", Loss: " << sample_loss->data[0] << std::endl;
    }
  }
  
  return 0;
}
```

### Basic Tensor Operations

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

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Optional: OpenBLAS, Intel MKL, or Apple Accelerate for optimized matrix operations

### Basic Build
```bash
git clone https://github.com/thevoxium/vayu-ml.git
cd vayu-ml
g++ -std=c++17 -O2 -o main examples/main.cpp src/*.cpp && ./main
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
    examples/main.cpp src/*.cpp -lopenblas -o main && ./main
```

**Ubuntu/Debian:**
```bash
# Install OpenBLAS
sudo apt-get install libopenblas-dev

# Compile with OpenBLAS
g++ -std=c++17 -O3 -DUSE_OPENBLAS -march=native \
    -I/usr/include/openblas \
    examples/main.cpp src/*.cpp -lopenblas -o main && ./main
```

### With Apple Accelerate Framework (macOS)
```bash
g++ -std=c++17 -O3 -march=native -framework Accelerate \
    examples/main.cpp src/*.cpp -o main && ./main
```

### With Intel MKL
```bash
g++ -std=c++17 -O3 -DUSE_MKL -march=native \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    examples/main.cpp src/*.cpp -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -o main && ./main
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

// Create tensor filled with constant value
auto t = make_const({2, 3}, 5.0f, true);
```

#### Basic Operations
```cpp
// Element-wise operations (with broadcasting)
auto c = a + b;    // Addition
auto c = a - b;    // Subtraction  
auto c = a * b;    // Element-wise multiplication

// Matrix multiplication
auto c = a->mm(b);              // BLAS-optimized matrix multiplication
auto c = a->mm(b, false);       // Non-optimized implementation
```

#### Activation Functions
```cpp
auto activated = tensor->relu();     // ReLU activation
auto activated = tensor->sigmoid();  // Sigmoid activation
```

#### Mathematical Functions
```cpp
auto powered = tensor->pow(2.0f);    // Element-wise power
auto exponential = tensor->exp();    // Element-wise exponential
auto logarithm = tensor->log();      // Element-wise natural logarithm
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

### Neural Network Layers

#### Linear Layer
```cpp
Linear layer(input_dim, output_dim);

// Forward pass
auto output = layer.forward(input);
auto output = layer(input);  // Shorthand

// Access parameters
auto weights = layer.get_weights();
auto bias = layer.get_bias();
auto params = layer.parameters();  // Returns vector of {weights, bias}

// Gradient operations
layer.zero_grad();
```

#### Activation Layers
```cpp
Relu relu_layer;
Sigmoid sigmoid_layer;

auto activated = relu_layer(input);
auto activated = sigmoid_layer(input);
```

### Sequential Model

```cpp
Sequential model;

// Add layers
model.add(Linear(784, 128));
model.add(Relu());
model.add(Linear(128, 10));
model.add(Sigmoid());

// Forward pass
auto output = model(input);
auto output = model.forward(input);  // Equivalent

// Get all parameters
auto all_params = model.parameters();

// Gradient operations
model.zero_grad();

// Model info
size_t num_layers = model.size();
```

### DataLoader

```cpp
// Create dataloader
Dataloader loader(X, y, batch_size, shuffle);

// Training loop
loader.reset();  // Reset to beginning, reshuffle if enabled
while (loader.has_next()) {
    auto batch = loader.next();
    auto batch_X = batch.first;
    auto batch_y = batch.second;
    
    // Train on batch...
}

// Get specific batch
auto batch = loader.get_batch(batch_id);

// Utility
size_t total_batches = loader.num_batch();
```

### Loss Functions

```cpp
// Mean Squared Error
auto loss = mse_loss(predictions, targets);
auto loss = predictions->mse_loss(targets);  // Method version
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

## License

MIT License
