#include "../include/value.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <strstream>
#include <vector>

Value::Value(double data, std::set<std::shared_ptr<Value>> children,
             std::string op)
    : data(data), grad(0.0), _op(op), _prev(children) {
  _backward = []() {};
}

std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> other) {
  auto out = std::make_shared<Value>(
      this->data + other->data,
      std::set<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, other, out]() {
    self_ptr->grad += out->grad;
    other->grad += out->grad;
  };

  return out;
}
std::shared_ptr<Value> operator+(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b) {
  return a->operator+(b);
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) {
  auto out = std::make_shared<Value>(
      this->data * other->data,
      std::set<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
  auto self_ptr = shared_from_this();
  out->_backward = [self_ptr, other, out]() {
    self_ptr->grad += (out->grad * other->data);
    other->grad += (out->grad * self_ptr->data);
  };
  return out;
}
std::shared_ptr<Value> operator*(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b) {
  return a->operator*(b);
}
void Value::backward() {
  std::vector<std::shared_ptr<Value>> topo;
  std::set<std::shared_ptr<Value>> visited;
  std::function<void(std::shared_ptr<Value>)> build_topo =
      [&](std::shared_ptr<Value> v) {
        if (visited.find(v) == visited.end()) {
          visited.insert(v);
          for (auto child : v->_prev) {
            build_topo(child);
          }
          topo.push_back(v);
        }
      };

  build_topo(shared_from_this());
  this->grad = 1.0;
  std::reverse(topo.begin(), topo.end());
  for (auto x : topo) {
    x->_backward();
  }
}

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
  return os;
}

std::shared_ptr<Value> make_value(double data) {
  return std::make_shared<Value>(data);
}
