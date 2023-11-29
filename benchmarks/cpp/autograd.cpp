#include <iostream>

#include "mlx/mlx.h"
#include "time_utils.h"

using namespace mlx::core;

void time_value_and_grad() {
  auto x = ones({200, 1000});
  eval(x);
  auto fn = [](array x) {
    for (int i = 0; i < 20; ++i) {
      x = log(exp(x));
    }
    return sum(x);
  };

  auto grad_fn = grad(fn);
  auto independent_value_and_grad = [&]() {
    auto value = fn(x);
    auto dfdx = grad_fn(x);
    return std::vector<array>{value, dfdx};
  };
  TIME(independent_value_and_grad);

  auto value_and_grad_fn = value_and_grad(fn);
  auto combined_value_and_grad = [&]() {
    auto [value, dfdx] = value_and_grad_fn(x);
    return std::vector<array>{value, dfdx};
  };
  TIME(combined_value_and_grad);
}

int main() {
  std::cout << "Benchmarks for " << default_device() << std::endl;
  time_value_and_grad();
}
