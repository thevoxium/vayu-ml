<img src="https://raw.githubusercontent.com/thevoxium/vayu/refs/heads/main/logo2.png" alt="Project Logo" width="" style="border-radius: 50%; float: left; margin-right: 15px; margin-bottom: -10px;" />



**A lightweight deep learning framework built from scratch in C++**

[![GitHub stars](https://img.shields.io/github/stars/thevoxium/vayu-ml?style=social)](https://github.com/thevoxium/vayu-ml)



## Quick Start

### Complete Neural Network Example

```cpp
#include "include/vayu.h"
#include <iostream>

int main() {
    // Create synthetic dataset (1000 samples, 784 features, 10 classes)
    auto X_train = random_tensor({1000, 784}, false, -1.0f, 1.0f);
    auto Y_train = random_tensor({1000, 10}, false, 0.0f, 1.0f);
    
    // Build neural network
    Sequential model;
    model.add(Linear(784, 512));
    model.add(Relu());
    model.add(Linear(512, 256));
    model.add(Tanh());
    model.add(Linear(256, 128));
    model.add(Relu());
    model.add(Linear(128, 10));
    model.add(Softmax());
    
    // Setup optimizer and training parameters
    auto params = model.parameters();
    SGD optimizer(params, 0.001f);  // Learning rate: 0.001
    
    size_t epochs = 100;
    size_t batch_size = 32;
    
    // Create dataloader with shuffling
    Dataloader dataloader(X_train, Y_train, batch_size, true);
    
    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        dataloader.reset();
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        
        while (dataloader.has_next()) {
            auto [batch_x, batch_y] = dataloader.next();
            
            // Forward pass
            auto predictions = model(batch_x);
            
            // Compute cross-entropy loss
            auto loss = cross_entropy_loss(predictions, batch_y);
            
            // Backward pass
            optimizer.zero_grad();
            loss->backward();
            optimizer.step();
            
            epoch_loss += loss->data[0];
            num_batches++;
        }
        
        // Print progress
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch 
                      << ", Average Loss: " << epoch_loss / num_batches 
                      << std::endl;
        }
    }
    
    return 0;
}

```

### Basic Tensor Operations

```cpp
#include "include/vayu.h"
#include <iostream>

int main() {
    // Create tensors
    auto a = random_tensor({3, 4}, true, -2.0f, 2.0f);
    auto b = make_ones({4, 2}, true);
    auto c = make_const({3, 1}, 0.5f, true);
    
    // Matrix operations
    auto mm_result = a->mm(b);           // Matrix multiplication (3x4) × (4x2) = (3x2)
    auto add_result = mm_result + c;     // Broadcasting addition (3x2) + (3x1) = (3x2)
    
    // Element-wise operations
    auto squared = add_result->pow(2.0f);
    auto activated = squared->relu();
    auto final = activated->sum();       // Scalar result
    
    // Backpropagation
    final->backward();
    
    // Access gradients
    std::cout << "Gradient of 'a':" << std::endl;
    for (size_t i = 0; i < a->shape[0]; ++i) {
        for (size_t j = 0; j < a->shape[1]; ++j) {
            std::cout << a->grad[i * a->shape[1] + j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}

```

## Installation & Building

### Requirements

-   **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
-   **Optional BLAS Backend** (for optimized matrix operations):
    -   OpenBLAS (recommended)
    -   Intel MKL
    -   Apple Accelerate Framework (macOS)

### Basic Build

```bash
git clone https://github.com/thevoxium/vayu-ml.git
cd vayu-ml

# Basic compilation
g++ -std=c++17 -O3 -march=native -o main examples/main.cpp src/*.cpp && ./main

```

### Optimized Builds

#### With OpenBLAS (Recommended)

**macOS (Homebrew):**

```bash
brew install openblas

g++ -std=c++17 -O3 -DUSE_OPENBLAS -march=native \
    -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    examples/main.cpp src/*.cpp -lopenblas -o main && ./main

```

**Ubuntu/Debian:**

```bash
sudo apt-get install libopenblas-dev

g++ -std=c++17 -O3 -DUSE_OPENBLAS -march=native \
    -I/usr/include/openblas \
    examples/main.cpp src/*.cpp -lopenblas -o main && ./main

```

#### With Apple Accelerate Framework (macOS)

```bash
g++ -std=c++17 -O3 -march=native -framework Accelerate \
    examples/main.cpp src/*.cpp -o main && ./main

```

#### With Intel MKL

```bash
g++ -std=c++17 -O3 -DUSE_MKL -march=native \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    examples/main.cpp src/*.cpp \
    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -o main && ./main

```

## Documentation

### Tensor Operations

#### Tensor Creation

```cpp
// From data
auto tensor = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);

// Zeros tensor
auto zeros = tensor({3, 3}, true);

// Random tensors
auto uniform = random_tensor({2, 3}, true, -1.0f, 1.0f);  // Uniform distribution

// Special tensors
auto ones = make_ones({2, 3}, true);
auto constant = make_const({2, 3}, 5.0f, true);

// From 2D vector
std::vector<std::vector<float>> data = {{1, 2}, {3, 4}};
auto from_vector = asvector(data, true);

// Range tensor
auto range_tensor = range(0, 10, 2, true);  // [0, 2, 4, 6, 8]

```

#### Arithmetic Operations

```cpp
auto a = random_tensor({3, 4}, true);
auto b = random_tensor({3, 4}, true);
auto c = random_tensor({1, 4}, true);  // For broadcasting

// Element-wise operations
auto sum = a + b;           // Addition
auto diff = a - b;          // Subtraction
auto product = a * b;       // Element-wise multiplication
auto broadcast_sum = a + c; // Broadcasting addition

// Matrix operations
auto mm_result = a->mm(b->transpose());  // Matrix multiplication
auto transposed = a->transpose();        // Transpose
auto reshaped = a->reshape({4, 3});      // Reshape

```

#### Mathematical Functions

```cpp
auto x = random_tensor({2, 3}, true, 0.1f, 5.0f);

// Power, exponential, logarithm
auto squared = x->pow(2.0f);
auto exponential = x->exp();
auto logarithm = x->log();

// Can also use free functions
auto result1 = pow(x, 3.0f);
auto result2 = exp(x);
auto result3 = log(x);

```

#### Activation Functions

```cpp
auto x = random_tensor({2, 5}, true, -3.0f, 3.0f);

// Available activations
auto relu_out = x->relu();        // ReLU: max(0, x)
auto sigmoid_out = x->sigmoid();  // Sigmoid: 1/(1+e^(-x))
auto tanh_out = x->tanh();        // Hyperbolic tangent
auto softmax_out = x->softmax();  // Softmax (for 2D tensors)

// Can also use free functions
auto result1 = relu(x);
auto result2 = sigmoid(x);
auto result3 = tanh(x);
auto result4 = softmax(x);

```

#### Reduction Operations

```cpp
auto x = random_tensor({3, 4}, true);

auto total_sum = x->sum();  // Sum all elements -> shape: {1}

```

#### Gradient Operations

```cpp
auto x = random_tensor({2, 2}, true);
auto y = x->pow(2.0f)->sum();

// Compute gradients
y->backward();

// Access gradients
for (size_t i = 0; i < x->numel(); ++i) {
    std::cout << "grad[" << i << "] = " << x->grad[i] << std::endl;
}

// Zero gradients
x->zero_grad();

// Clear computation graph (free memory)
y->clear_graph();

```

#### Broadcasting

Vayu supports NumPy-style broadcasting:

```cpp
auto a = random_tensor({3, 4}, true);  // Shape: (3, 4)
auto b = random_tensor({1, 4}, true);  // Shape: (1, 4)
auto c = random_tensor({3, 1}, true);  // Shape: (3, 1)

auto result1 = a + b;  // Broadcasting: (3,4) + (1,4) -> (3,4)
auto result2 = a + c;  // Broadcasting: (3,4) + (3,1) -> (3,4)

// Check broadcasting compatibility
bool can_broadcast = Tensor::can_broadcast({3, 4}, {1, 4});  // true
auto result_shape = Tensor::broadcast_shape({3, 4}, {1, 4}); // {3, 4}

```

### Neural Net Layers

#### Linear (Fully Connected) Layer

```cpp
// Create layer
Linear layer(input_dim, output_dim);

// Xavier/Glorot initialization is applied automatically
auto weights = layer.get_weights();  // Shape: {input_dim, output_dim}
auto bias = layer.get_bias();        // Shape: {output_dim}

// Forward pass
auto input = random_tensor({batch_size, input_dim}, false);
auto output = layer(input);          // Shape: {batch_size, output_dim}
auto output = layer.forward(input);  // Equivalent

// Get all parameters for optimizer
auto params = layer.parameters();    // Returns {weights, bias}

// Gradient operations
layer.zero_grad();

```

#### Activation Layers

```cpp
// Create activation layers
Relu relu_layer;
Sigmoid sigmoid_layer;
Tanh tanh_layer;
Softmax softmax_layer;

auto input = random_tensor({32, 128}, false);

// Apply activations
auto relu_out = relu_layer(input);
auto sigmoid_out = sigmoid_layer(input);
auto tanh_out = tanh_layer(input);
auto softmax_out = softmax_layer(input);  // Note: Expects 2D input

// Activation layers have no parameters
auto params = relu_layer.parameters();  // Returns empty vector

```

#### Sequential Model

```cpp
Sequential model;

// Add layers using template-based approach
model.add(Linear(784, 512));
model.add(Relu());
model.add(Linear(512, 256));
model.add(Tanh());
model.add(Linear(256, 10));
model.add(Softmax());

// Forward pass
auto input = random_tensor({32, 784}, false);
auto output = model(input);           // Shape: {32, 10}
auto output = model.forward(input);   // Equivalent

// Get all model parameters
auto all_params = model.parameters();

// Model information
size_t num_layers = model.size();     // Number of layers

// Gradient operations
model.zero_grad();

```

### Loss Functions

#### Mean Squared Error (MSE)

```cpp
auto predictions = random_tensor({32, 1}, true);
auto targets = random_tensor({32, 1}, false);

// Compute MSE loss
auto loss = mse_loss(predictions, targets);
auto loss = predictions->mse_loss(targets);  // Method version

// Loss is a scalar tensor with shape {1, 1}
float loss_value = loss->data[0];

```

#### Cross-Entropy Loss

```cpp
auto predictions = random_tensor({32, 10}, true);  // Logits or probabilities
auto targets = make_ones({32, 10}, false);         // One-hot encoded

// Compute cross-entropy loss
auto loss = cross_entropy_loss(predictions, targets);
auto loss = predictions->cross_entropy_loss(targets);  // Method version

float loss_value = loss->data[0];

```

### Optimizers

#### Stochastic Gradient Descent (SGD)

```cpp
// Get model parameters
Sequential model;
// ... add layers ...
auto params = model.parameters();

// Create SGD optimizer
float learning_rate = 0.01f;
SGD optimizer(params, learning_rate);

// Training step
auto loss = compute_loss();  // Your loss computation

optimizer.zero_grad();       // Zero gradients
loss->backward();           // Compute gradients
optimizer.step();           // Update parameters

// Manual zero_grad alternative
for (auto& param : params) {
    param->zero_grad();
}

```

### Data Loading

#### DataLoader

```cpp
auto X = random_tensor({1000, 784}, false);  // 1000 samples, 784 features
auto Y = random_tensor({1000, 10}, false);   // 1000 samples, 10 classes

// Create dataloader
size_t batch_size = 32;
bool shuffle = true;
Dataloader loader(X, Y, batch_size, shuffle);

// Training loop
for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    loader.reset();  // Reset for new epoch, reshuffle if enabled
    
    while (loader.has_next()) {
        auto [batch_x, batch_y] = loader.next();
        
        // batch_x shape: {32, 784} or {remaining_samples, 784} for last batch
        // batch_y shape: {32, 10} or {remaining_samples, 10} for last batch
        
        // Train on batch...
    }
}

// Utility functions
size_t total_batches = loader.num_batch();
auto specific_batch = loader.get_batch(batch_id);
bool has_more = loader.has_next();

```

### Performance Monitoring

#### Timer Utilities

```cpp
#include "utils/timer.h"

// Time a code block
START_TIMER(forward_pass);
auto output = model(input);
END_TIMER(forward_pass);
PRINT_TIMER(forward_pass);  // Prints execution time in microseconds

// Time multiple operations
START_TIMER(training_step);
// ... training code ...
END_TIMER(training_step);

START_TIMER(backward_pass);
loss->backward();
END_TIMER(backward_pass);

PRINT_TIMER(training_step);
PRINT_TIMER(backward_pass);

```


## Complete Examples

### Image Classification Network

```cpp
#include "include/vayu.h"

int main() {
    // MNIST-like dataset (28x28 grayscale images, 10 classes)
    auto X_train = random_tensor({60000, 784}, false, 0.0f, 1.0f);
    auto Y_train = random_tensor({60000, 10}, false, 0.0f, 1.0f);
    
    // Build CNN-like network (flattened)
    Sequential model;
    model.add(Linear(784, 1024));
    model.add(Relu());
    model.add(Linear(1024, 512));
    model.add(Relu());
    model.add(Linear(512, 256));
    model.add(Relu());
    model.add(Linear(256, 10));
    model.add(Softmax());
    
    // Training setup
    auto params = model.parameters();
    SGD optimizer(params, 0.001f);
    
    size_t epochs = 50;
    size_t batch_size = 64;
    Dataloader dataloader(X_train, Y_train, batch_size, true);
    
    // Training loop with progress tracking
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        dataloader.reset();
        
        float epoch_loss = 0.0f;
        size_t batch_count = 0;
        
        START_TIMER(epoch_time);
        
        while (dataloader.has_next()) {
            auto [batch_x, batch_y] = dataloader.next();
            
            // Forward pass
            auto predictions = model(batch_x);
            auto loss = cross_entropy_loss(predictions, batch_y);
            
            // Backward pass
            optimizer.zero_grad();
            loss->backward();
            optimizer.step();
            
            epoch_loss += loss->data[0];
            batch_count++;
        }
        
        END_TIMER(epoch_time);
        
        if (epoch % 5 == 0) {
            std::cout << "Epoch " << epoch 
                      << " | Loss: " << epoch_loss / batch_count;
            PRINT_TIMER(epoch_time);
        }
    }
    
    return 0;
}

```

### Regression Network

```cpp
#include "include/vayu.h"

int main() {
    // Generate synthetic regression data
    auto X = random_tensor({5000, 20}, false, -2.0f, 2.0f);
    
    // Create target: y = sum(X^2) + noise
    auto Y = tensor({5000, 1}, false);
    for (size_t i = 0; i < 5000; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < 20; ++j) {
            float val = X->data[i * 20 + j];
            sum += val * val;
        }
        Y->data[i] = sum + (rand() / float(RAND_MAX) - 0.5f) * 0.1f;  // Add noise
    }
    
    // Build regression network
    Sequential model;
    model.add(Linear(20, 64));
    model.add(Tanh());
    model.add(Linear(64, 32));
    model.add(Relu());
    model.add(Linear(32, 16));
    model.add(Tanh());
    model.add(Linear(16, 1));
    // No activation on output for regression
    
    // Training
    auto params = model.parameters();
    SGD optimizer(params, 0.01f);
    
    Dataloader dataloader(X, Y, 32, true);
    
    for (size_t epoch = 0; epoch < 100; ++epoch) {
        dataloader.reset();
        
        while (dataloader.has_next()) {
            auto [batch_x, batch_y] = dataloader.next();
            
            auto predictions = model(batch_x);
            auto loss = mse_loss(predictions, batch_y);  // MSE for regression
            
            optimizer.zero_grad();
            loss->backward();
            optimizer.step();
        }
        
        if (epoch % 20 == 0) {
            auto test_pred = model(X);
            auto test_loss = mse_loss(test_pred, Y);
            std::cout << "Epoch " << epoch 
                      << " | Test MSE: " << test_loss->data[0] << std::endl;
        }
    }
    
    return 0;
}

```



----------

**Vayu** - _Sanskrit for "wind"_ - designed to be fast, lightweight, and powerful for deep learning applications.
