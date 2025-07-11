#ifndef TENSOR_H
#define TENSOR_H
#include <cassert>
#include <cstddef>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef USE_OPENBLAS
#include <cblas.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#elif defined(USE_MKL)
#include <mkl.h>
#endif

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
  size_t cached_size;
  std::vector<size_t> strides;

  void compute_strides() {
    strides.resize(shape.size());
    if (shape.empty() == false) {
      strides.back() = 1;
      for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
      }
    }
  }

  size_t
  get_broadcast_source_index(const std::vector<size_t> &output_indices,
                             const std::vector<size_t> &source_shape) const;

public:
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<size_t> shape;
  std::function<void()> _backward;
  std::set<std::shared_ptr<Tensor>> _prev;
  std::string _op;
  int ndim;
  bool requires_grad;

  Tensor(const std::vector<size_t> &shape, bool requires_grad = true);
  Tensor(const std::vector<float> &data, const std::vector<size_t> &shape,
         bool requires_grad = true);

  size_t get_flatten_index(const std::vector<int> &indices) const {
    assert(indices.size() == shape.size());
    size_t idx = 0;
    for (int i = 0; i < indices.size(); i++) {
      idx += indices[i] * strides[i];
    }
    return idx;
  }

  std::vector<size_t> unflatten_index(size_t &idx) const {
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      indices[i] = (idx / strides[i]) % shape[i];
    }
    return indices;
  }

  std::vector<size_t> get_strides() { return strides; }
  size_t numel() const { return cached_size; }

  std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> other);

  std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> other);

  std::shared_ptr<Tensor> operator[](size_t idx);
  std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> mul(float scalar);
  std::shared_ptr<Tensor> div(float scalar);
  std::shared_ptr<Tensor> clamp(float min_val, float max_val);

  std::shared_ptr<Tensor> neg();
  std::shared_ptr<Tensor> log2();
  std::shared_ptr<Tensor> exp2();
  std::shared_ptr<Tensor> sqrt();
  std::shared_ptr<Tensor> sin();
  std::shared_ptr<Tensor> cos();
  std::shared_ptr<Tensor> tan();
  std::shared_ptr<Tensor> trunc();
  std::shared_ptr<Tensor> ceil();
  std::shared_ptr<Tensor> floor();
  std::shared_ptr<Tensor> round();
  std::shared_ptr<Tensor> square();
  std::shared_ptr<Tensor> sign();
  std::shared_ptr<Tensor> abs();
  std::shared_ptr<Tensor> reciprocal();
  std::shared_ptr<Tensor> pow(float exponent);
  std::shared_ptr<Tensor> exp();
  std::shared_ptr<Tensor> log();

  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> tanh();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> softmax();
  std::shared_ptr<Tensor> elu(float alpha = 1.0f);
  std::shared_ptr<Tensor> swish();
  std::shared_ptr<Tensor> gelu();
  std::shared_ptr<Tensor> leaky_relu(float alpha = 0.01f);

  std::shared_ptr<Tensor> mm(std::shared_ptr<Tensor> other, bool fast = true);
  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> transpose();
  std::shared_ptr<Tensor> reshape(const std::vector<size_t> &shape);

  std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> target);
  std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> target);

  void init_grad();
  void backward();
  void zero_grad();
  void clear_graph();

  std::shared_ptr<Tensor> broadcast(const std::vector<size_t> &target_shape);
  static bool can_broadcast(const std::vector<size_t> shape1,
                            const std::vector<size_t> shape2);
  static std::vector<size_t> broadcast_shape(const std::vector<size_t> shape1,
                                             const std::vector<size_t> shape2);

  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
  void print_shape() {
    std::cout << "(" + std::to_string(shape[0]) + ", " +
                     std::to_string(shape[1]) + ")"
              << std::endl;
  }
};

std::shared_ptr<Tensor> tensor(const std::vector<float> &data,
                               const std::vector<size_t> &shape,
                               bool requires_grad);
std::shared_ptr<Tensor> tensor(const std::vector<size_t> &shape,
                               bool requires_grad);
std::shared_ptr<Tensor> empty(const std::vector<size_t> &shape,
                              bool requires_grad = true);
std::shared_ptr<Tensor> ones(const std::vector<size_t> &shape,
                             bool requires_grad = true);
std::shared_ptr<Tensor> full(const std::vector<size_t> &shape, float val,
                             bool requires_grad = true);
std::shared_ptr<Tensor> asvector(const std::vector<std::vector<float>> &input,
                                 bool requires_grad = true);
std::shared_ptr<Tensor> arange(int start, int end, size_t step = 1,
                               bool requires_grad = true);
std::shared_ptr<Tensor> linspace(double start = 0.0, double stop = 2.0,
                                 size_t num_steps = 1,
                                 bool requires_grad = true);
std::shared_ptr<Tensor> random_tensor(const std::vector<size_t> &shape,
                                      bool requires_grad = true,
                                      float min_val = 0.0f,
                                      float max_value = 1.0f);

std::shared_ptr<Tensor> gt(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> eq(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> ne(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> le(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> lt(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> ge(std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> a,
                                  std::shared_ptr<Tensor> b);

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> base, float exponent);
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> base);
std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> num);
std::shared_ptr<Tensor> exp2(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tan(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> trunc(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> ceil(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> floor(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> round(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> square(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, float scalar);
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, float scalar);
std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val,
                              float max_val);
std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> a);

std::shared_ptr<Tensor> elu(std::shared_ptr<Tensor> a, float alpha = 1.0f);
std::shared_ptr<Tensor> swish(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a,
                                   float alpha = 0.01f);
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> predicted,
                                 std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> predicted,
                                           std::shared_ptr<Tensor> target);

template <typename T> bool is_tensor(const T &obj) {
  if constexpr (std::is_same_v<T, std::shared_ptr<Tensor>>) {
    return obj != nullptr;
  } else {
    return false;
  }
}

#endif // !TENSOR_H
