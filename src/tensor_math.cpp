#include "../include/tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

std::shared_ptr<Tensor> Tensor::log2() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    out->data[i] = std::log2(this->data[i]);
  }
  out->_prev = {shared_from_this()};
  out->_op = "log2";
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        self_ptr->grad[i] +=
            (out->grad[i] / (self_ptr->data[i] * std::log(2.0) + 1e-8));
      }
    }
  };
  return out;
}

std::shared_ptr<Tensor> Tensor::neg() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < out->numel(); i++) {
    out->data[i] = -1.0f * this->data[i];
  }
  out->_prev = {shared_from_this()};
  out->_op = "neg";
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < out->numel(); i++) {
        self_ptr->grad[i] += (-out->grad[i]);
      }
    }
  };
  return out;
}

std::shared_ptr<Tensor> Tensor::pow(float exponent) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::pow(this->data[i], exponent);
  }

  out->_prev = {shared_from_this()};
  out->_op = "pow";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, exponent]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (exponent == 0.0f)
          self_ptr->grad[i] = 0.0f;
        else {
          self_ptr->grad[i] += out->grad[i] * exponent *
                               std::pow(self_ptr->data[i], exponent - 1.0f);
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::exp() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::exp(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "exp";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += out->grad[i] * out->data[i];
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::log() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  const float epsilon = 1e-8f;
  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::log(std::max(this->data[i], epsilon));
  }

  out->_prev = {shared_from_this()};
  out->_op = "log";

  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] > 0.0f)
          self_ptr->grad[i] += out->grad[i] * (1.0f / self_ptr->data[i]);
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::exp2() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  const float ln2 = std::log(2.0f);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::exp2(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "exp2";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, ln2]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] +=
            out->grad[i] * out->data[i] * ln2; // d/dx(2^x) = 2^x * ln(2)
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::sqrt() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::sqrt(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "sqrt";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] > 0.0f) {
          self_ptr->grad[i] +=
              out->grad[i] / (2.0f * out->data[i]); // d/dx(√x) = 1/(2√x)
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::sin() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::sin(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "sin";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] +=
            out->grad[i] * std::cos(self_ptr->data[i]); // d/dx(sin(x)) = cos(x)
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::cos() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::cos(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "cos";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] +=
            out->grad[i] *
            (-std::sin(self_ptr->data[i])); // d/dx(cos(x)) = -sin(x)
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::tan() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::tan(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "tan";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        float cos_val = std::cos(self_ptr->data[i]);
        self_ptr->grad[i] +=
            out->grad[i] /
            (cos_val * cos_val); // d/dx(tan(x)) = sec²(x) = 1/cos²(x)
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::square() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = this->data[i] * this->data[i];
  }

  out->_prev = {shared_from_this()};
  out->_op = "square";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] +=
            out->grad[i] * 2.0f * self_ptr->data[i]; // d/dx(x²) = 2x
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::abs() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::abs(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "abs";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] > 0.0f) {
          self_ptr->grad[i] += out->grad[i]; // d/dx(|x|) = 1 for x > 0
        } else if (self_ptr->data[i] < 0.0f) {
          self_ptr->grad[i] += -out->grad[i]; // d/dx(|x|) = -1 for x < 0
        }
        // At x = 0, gradient is undefined, so we don't update
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::reciprocal() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = 1.0f / this->data[i];
  }

  out->_prev = {shared_from_this()};
  out->_op = "reciprocal";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (std::abs(self_ptr->data[i]) > 1e-8f) {
          self_ptr->grad[i] +=
              out->grad[i] * (-1.0f / (self_ptr->data[i] *
                                       self_ptr->data[i])); // d/dx(1/x) = -1/x²
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::trunc() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::trunc(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "trunc";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    // Non-differentiable - no gradient propagation
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::ceil() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::ceil(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "ceil";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    // Non-differentiable - no gradient propagation
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::floor() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::floor(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "floor";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    // Non-differentiable - no gradient propagation
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::round() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::round(this->data[i]);
  }

  out->_prev = {shared_from_this()};
  out->_op = "round";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    // Non-differentiable - no gradient propagation
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::sign() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] > 0.0f) {
      out->data[i] = 1.0f;
    } else if (this->data[i] < 0.0f) {
      out->data[i] = -1.0f;
    } else {
      out->data[i] = 0.0f;
    }
  }

  out->_prev = {shared_from_this()};
  out->_op = "sign";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    // Non-differentiable - no gradient propagation
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::elu(float alpha) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] >= 0.0f) {
      out->data[i] = this->data[i];
    } else {
      out->data[i] = alpha * (std::exp(this->data[i]) - 1.0f);
    }
  }

  out->_prev = {shared_from_this()};
  out->_op = "elu";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, alpha]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] >= 0.0f) {
          self_ptr->grad[i] += out->grad[i]; // d/dx(x) = 1 for x >= 0
        } else {
          self_ptr->grad[i] +=
              out->grad[i] * alpha *
              std::exp(self_ptr->data[i]); // d/dx(α(e^x - 1)) = α*e^x for x < 0
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::swish() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    float sigmoid_val = 1.0f / (1.0f + std::exp(-this->data[i]));
    out->data[i] = this->data[i] * sigmoid_val;
  }

  out->_prev = {shared_from_this()};
  out->_op = "swish";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        float x = self_ptr->data[i];
        float sigmoid_val = 1.0f / (1.0f + std::exp(-x));
        // d/dx(x * σ(x)) = σ(x) + x * σ(x) * (1 - σ(x))
        float grad_val = sigmoid_val + x * sigmoid_val * (1.0f - sigmoid_val);
        self_ptr->grad[i] += out->grad[i] * grad_val;
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::gelu() {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);
  const float sqrt_2_pi = std::sqrt(2.0f / M_PI);

  for (size_t i = 0; i < this->numel(); i++) {
    float x = this->data[i];
    // GELU approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_pi * (x + 0.044715f * x_cubed);
    float tanh_val = std::tanh(tanh_arg);
    out->data[i] = 0.5f * x * (1.0f + tanh_val);
  }

  out->_prev = {shared_from_this()};
  out->_op = "gelu";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, sqrt_2_pi]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        float x = self_ptr->data[i];
        float x_squared = x * x;
        float x_cubed = x_squared * x;

        float tanh_arg = sqrt_2_pi * (x + 0.044715f * x_cubed);
        float tanh_val = std::tanh(tanh_arg);
        float sech_squared = 1.0f - tanh_val * tanh_val;

        float grad_tanh_arg = sqrt_2_pi * (1.0f + 3.0f * 0.044715f * x_squared);

        // Complex derivative for GELU
        float grad_val =
            0.5f * (1.0f + tanh_val) + 0.5f * x * sech_squared * grad_tanh_arg;
        self_ptr->grad[i] += out->grad[i] * grad_val;
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::leaky_relu(float alpha) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    if (this->data[i] > 0.0f) {
      out->data[i] = this->data[i];
    } else {
      out->data[i] = alpha * this->data[i];
    }
  }

  out->_prev = {shared_from_this()};
  out->_op = "leaky_relu";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, alpha]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        if (self_ptr->data[i] > 0.0f) {
          self_ptr->grad[i] += out->grad[i]; // gradient = 1 for x > 0
        } else {
          self_ptr->grad[i] += out->grad[i] * alpha; // gradient = α for x <= 0
        }
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::mul(float scalar) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = this->data[i] * scalar;
  }

  out->_prev = {shared_from_this()};
  out->_op = "mul_scalar";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, scalar]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += out->grad[i] * scalar; // d/dx(c*x) = c
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::div(float scalar) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = this->data[i] / scalar;
  }

  out->_prev = {shared_from_this()};
  out->_op = "div_scalar";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, scalar]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        self_ptr->grad[i] += out->grad[i] / scalar; // d/dx(x/c) = 1/c
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> Tensor::clamp(float min_val, float max_val) {
  auto out = std::make_shared<Tensor>(this->shape, this->requires_grad);

  for (size_t i = 0; i < this->numel(); i++) {
    out->data[i] = std::max(min_val, std::min(max_val, this->data[i]));
  }

  out->_prev = {shared_from_this()};
  out->_op = "clamp";
  auto self_ptr = shared_from_this();

  out->_backward = [self_ptr, out, min_val, max_val]() {
    if (self_ptr->requires_grad) {
      for (size_t i = 0; i < self_ptr->numel(); i++) {
        // Gradient flows only if input is within [min_val, max_val]
        if (self_ptr->data[i] >= min_val && self_ptr->data[i] <= max_val) {
          self_ptr->grad[i] += out->grad[i];
        }
        // Otherwise gradient is 0
      }
    }
  };

  return out;
}

std::shared_ptr<Tensor> neg(std::shared_ptr<Tensor> a) { return a->neg(); }
std::shared_ptr<Tensor> log2(std::shared_ptr<Tensor> a) { return a->log2(); }
std::shared_ptr<Tensor> pow(std::shared_ptr<Tensor> base, float exponent) {
  return base->pow(exponent);
}
std::shared_ptr<Tensor> exp2(std::shared_ptr<Tensor> a) { return a->exp2(); }
std::shared_ptr<Tensor> sqrt(std::shared_ptr<Tensor> a) { return a->sqrt(); }
std::shared_ptr<Tensor> sin(std::shared_ptr<Tensor> a) { return a->sin(); }
std::shared_ptr<Tensor> cos(std::shared_ptr<Tensor> a) { return a->cos(); }
std::shared_ptr<Tensor> tan(std::shared_ptr<Tensor> a) { return a->tan(); }
std::shared_ptr<Tensor> trunc(std::shared_ptr<Tensor> a) { return a->trunc(); }
std::shared_ptr<Tensor> ceil(std::shared_ptr<Tensor> a) { return a->ceil(); }
std::shared_ptr<Tensor> floor(std::shared_ptr<Tensor> a) { return a->floor(); }
std::shared_ptr<Tensor> round(std::shared_ptr<Tensor> a) { return a->round(); }
std::shared_ptr<Tensor> square(std::shared_ptr<Tensor> a) {
  return a->square();
}
std::shared_ptr<Tensor> sign(std::shared_ptr<Tensor> a) { return a->sign(); }
std::shared_ptr<Tensor> abs(std::shared_ptr<Tensor> a) { return a->abs(); }
std::shared_ptr<Tensor> reciprocal(std::shared_ptr<Tensor> a) {
  return a->reciprocal();
}
std::shared_ptr<Tensor> elu(std::shared_ptr<Tensor> a, float alpha) {
  return a->elu(alpha);
}
std::shared_ptr<Tensor> swish(std::shared_ptr<Tensor> a) { return a->swish(); }
std::shared_ptr<Tensor> gelu(std::shared_ptr<Tensor> a) { return a->gelu(); }
std::shared_ptr<Tensor> leaky_relu(std::shared_ptr<Tensor> a, float alpha) {
  return a->leaky_relu(alpha);
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, float scalar) {
  return a->mul(scalar);
}
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, float scalar) {
  return a->div(scalar);
}
std::shared_ptr<Tensor> clamp(std::shared_ptr<Tensor> a, float min_val,
                              float max_val) {
  return a->clamp(min_val, max_val);
}

std::shared_ptr<Tensor> log(std::shared_ptr<Tensor> num) { return num->log(); }
std::shared_ptr<Tensor> exp(std::shared_ptr<Tensor> base) {
  return base->exp();
}
