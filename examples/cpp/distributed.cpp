// Copyright © 2024 Apple Inc.

#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  if (!mx::distributed::is_available()) {
    std::cout << "No communication backend found" << std::endl;
    return 1;
  }

  auto global_group = mx::distributed::init();
  std::cout << global_group.rank() << " / " << global_group.size() << std::endl;

  mx::array x = mx::ones({10});
  mx::array out = mx::distributed::all_sum(x, global_group);

  std::cout << out << std::endl;
}
