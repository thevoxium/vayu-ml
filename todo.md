
### **1. Core Tensor Operations**

```cpp
// Division
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> other);

// Comparison operations
std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> other); /* DONE */
std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> other);

// Logical operations
std::shared_ptr<Tensor> logical_and(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> logical_or(std::shared_ptr<Tensor> other);
std::shared_ptr<Tensor> logical_not();

// Selection operations
std::shared_ptr<Tensor> where(std::shared_ptr<Tensor> condition, 
                              std::shared_ptr<Tensor> other);

```

### **2. Advanced Reductions**

```cpp
// Reduction operations with axis support
std::shared_ptr<Tensor> sum(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> mean(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> max(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> min(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> argmax(int axis = -1);
std::shared_ptr<Tensor> argmin(int axis = -1);
std::shared_ptr<Tensor> var(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> std(const std::vector<int>& axes = {}, bool keepdim = false);
std::shared_ptr<Tensor> prod(const std::vector<int>& axes = {}, bool keepdim = false);

```

### **3. Shape Manipulation**

```cpp
// Advanced shape operations
std::shared_ptr<Tensor> squeeze(const std::vector<int>& axes = {});
std::shared_ptr<Tensor> unsqueeze(int axis);
std::shared_ptr<Tensor> expand(const std::vector<size_t>& shape);
std::shared_ptr<Tensor> repeat(const std::vector<size_t>& repeats);
std::shared_ptr<Tensor> permute(const std::vector<int>& axes);
std::shared_ptr<Tensor> flatten(int start_dim = 0, int end_dim = -1);

// Concatenation and splitting
std::shared_ptr<Tensor> cat(const std::vector<std::shared_ptr<Tensor>>& tensors, int axis);
std::vector<std::shared_ptr<Tensor>> split(int split_size, int axis);
std::shared_ptr<Tensor> stack(const std::vector<std::shared_ptr<Tensor>>& tensors, int axis);

```

### **4. Indexing and Slicing**

```cpp
// Advanced indexing
std::shared_ptr<Tensor> slice(const std::vector<std::pair<int, int>>& ranges);
std::shared_ptr<Tensor> gather(std::shared_ptr<Tensor> indices, int axis);
std::shared_ptr<Tensor> scatter(std::shared_ptr<Tensor> indices, 
                                std::shared_ptr<Tensor> updates, int axis);
std::shared_ptr<Tensor> index_select(std::shared_ptr<Tensor> indices, int axis);

```

### **5. Convolution Operations**

```cpp
class Conv2d : public Layer {
private:
    size_t in_channels, out_channels, kernel_size, stride, padding;
    std::shared_ptr<Tensor> weight, bias;
public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, 
           size_t stride = 1, size_t padding = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

// Pooling operations
std::shared_ptr<Tensor> max_pool2d(const std::vector<int>& kernel_size, 
                                   const std::vector<int>& stride = {},
                                   const std::vector<int>& padding = {});
std::shared_ptr<Tensor> avg_pool2d(const std::vector<int>& kernel_size, 
                                   const std::vector<int>& stride = {},
                                   const std::vector<int>& padding = {});

```

### **6. Normalization Layers**

```cpp
class BatchNorm2d : public Layer {
private:
    size_t num_features;
    std::shared_ptr<Tensor> weight, bias, running_mean, running_var;
    float eps, momentum;
    bool training;
public:
    BatchNorm2d(size_t num_features, float eps = 1e-5, float momentum = 0.1);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

class LayerNorm : public Layer {
private:
    std::vector<size_t> normalized_shape;
    std::shared_ptr<Tensor> weight, bias;
    float eps;
public:
    LayerNorm(const std::vector<size_t>& normalized_shape, float eps = 1e-5);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};

```

### **7. Attention Mechanisms**

```cpp
class MultiHeadAttention : public Layer {
private:
    size_t embed_dim, num_heads, head_dim;
    std::shared_ptr<Linear> q_proj, k_proj, v_proj, out_proj;
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> query,
                                   std::shared_ptr<Tensor> key,
                                   std::shared_ptr<Tensor> value,
                                   std::shared_ptr<Tensor> mask = nullptr) override;
};

// Attention functions
std::shared_ptr<Tensor> scaled_dot_product_attention(
    std::shared_ptr<Tensor> query,
    std::shared_ptr<Tensor> key, 
    std::shared_ptr<Tensor> value,
    std::shared_ptr<Tensor> mask = nullptr);

```


### **8. Loss Functions**

```cpp
// Additional loss functions
std::shared_ptr<Tensor> l1_loss(std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> smooth_l1_loss(std::shared_ptr<Tensor> target, float beta = 1.0);
std::shared_ptr<Tensor> binary_cross_entropy(std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> nll_loss(std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> kl_div_loss(std::shared_ptr<Tensor> target);

```


### **9. Initialization Functions**

```cpp
// Weight initialization
std::shared_ptr<Tensor> xavier_uniform(const std::vector<size_t>& shape);
std::shared_ptr<Tensor> xavier_normal(const std::vector<size_t>& shape);
std::shared_ptr<Tensor> kaiming_uniform(const std::vector<size_t>& shape);
std::shared_ptr<Tensor> kaiming_normal(const std::vector<size_t>& shape);

```
