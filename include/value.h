#ifndef VALUE_H
#define VALUE_H

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

class Value : public std::enable_shared_from_this<Value> {
public:
  double data;
  double grad;
  std::function<void()> _backward;
  std::set<std::shared_ptr<Value>> _prev;
  std::string _op, label;

  Value(double data, std::set<std::shared_ptr<Value>> children = {},
        std::string op = "");

  std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
  std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
  std::shared_ptr<Value> pow(double other);
  std::shared_ptr<Value> relu();
  std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
  std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);

  void backward();

  friend std::ostream &operator<<(std::ostream &os, const Value &v);
};

std::shared_ptr<Value> make_value(double data);
std::shared_ptr<Value> operator+(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b);

std::shared_ptr<Value> operator*(std::shared_ptr<Value> a,
                                 std::shared_ptr<Value> b);
#endif // VALUE_H
