// Copyright © 2024 Apple Inc.

#include <iostream>

#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  if (!distributed::is_available()) {
    std::cout << "No communication backend found" << std::endl;
    return 1;
  }

  auto global_group = distributed::init();
  std::cout << global_group.rank() << " / " << global_group.size() << std::endl;

  array x = ones({10});
  array out = distributed::all_reduce_sum(x, global_group);

  std::cout << out << std::endl;
}
