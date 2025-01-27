// Copyright © 2023 Apple Inc.

#include <iostream>
#include "mlx/mlx.h"
#include "time_utils.h"

namespace mx = mlx::core;

void time_cholesky() {
  std::vector<int> sizes{10, 100, 1000, 2000, 3000};
  set_default_device(mx::Device::cpu);
  for (auto size : sizes) {
    // Method to generate symmetric positive definite matrices:
    // https://math.stackexchange.com/a/358092
    auto A = mx::random::uniform({size, size});
    A = 0.5 * (A + mx::transpose(A));
    A = A + mx::diag(A);
    mx::eval(A);
    std::cout << "Size " << size << std::endl;
    TIMEM("cpu", mx::linalg::cholesky, A, false, mx::Device::cpu);
  }
}

int main() {
  time_cholesky();
}
