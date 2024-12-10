// Copyright © 2023 Apple Inc.

#include <iostream>

#include "mlx/mlx.h"
#include "time_utils.h"

namespace mx = mlx::core;

void time_value_and_grad() {
  auto x = mx::ones({200, 1000});
  mx::eval(x);
  auto fn = [](mx::array x) {
    for (int i = 0; i < 20; ++i) {
      x = mx::log(mx::exp(x));
    }
    return mx::sum(x);
  };

  auto grad_fn = mx::grad(fn);
  auto independent_value_and_grad = [&]() {
    auto value = fn(x);
    auto dfdx = grad_fn(x);
    return std::vector<mx::array>{value, dfdx};
  };
  TIME(independent_value_and_grad);

  auto value_and_grad_fn = mx::value_and_grad(fn);
  auto combined_value_and_grad = [&]() {
    auto [value, dfdx] = value_and_grad_fn(x);
    return std::vector<mx::array>{value, dfdx};
  };
  TIME(combined_value_and_grad);
}

int main() {
  std::cout << "Benchmarks for " << mx::default_device() << std::endl;
  time_value_and_grad();
}
