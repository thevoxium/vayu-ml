#include "../include/value.h"
#include <iostream>

int main() {
  auto w = make_value(0.0);
  auto b = make_value(0.0);
  double learning_rate = 0.0001;

  double x_data = 3.0, y_data = 5.0;

  for (int i = 0; i < 10000; i++) {
    auto x = make_value(x_data);
    auto y_true = make_value(y_data);
    auto y_pred = w * x + b;
    auto loss = (y_pred - y_true);
    auto loss_sq = pow(loss, 2);

    loss_sq->backward();

    w = make_value(w->data - learning_rate * w->grad);
    b = make_value(b->data - learning_rate * b->grad);

    w->grad = 0.0;
    b->grad = 0.0;
  }

  std::cout << "Trained w: " << *w << std::endl;
  std::cout << "Trained b: " << *b << std::endl;

  return 0;
}
