// Copyright © 2024 Apple Inc.

#include <mlx/mlx.h>
#include <iostream>

namespace mx = mlx::core;

int main() {
  int batch_size = 8;
  int input_dim = 32;

  // Make the input
  mx::random::seed(42);
  auto example_x = mx::random::uniform({batch_size, input_dim});

  // Import the function
  auto forward = mx::import_function("eval_mlp.mlxfn");

  // Call the imported function
  auto out = forward({example_x})[0];

  std::cout << out << std::endl;

  return 0;
}
