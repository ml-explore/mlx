// Copyright © 2024 Apple Inc.

#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  auto x = mx::array({1, 2, 3});
  auto y = mx::array({1, 2, 3});
  std::cout << x + y << std::endl;
  return 0;
}
