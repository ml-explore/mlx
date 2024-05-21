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
  array x = ones({10});
  array out = zeros({10});
  eval(x, out);

  std::cout << global_group->rank() << " / " << global_group->size()
            << std::endl;
  distributed::all_reduce_sum(global_group, x, out);

  std::cout << out << std::endl;
}
