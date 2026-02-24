// Copyright © 2026 Apple Inc.
//
// Regression test for https://github.com/ml-explore/mlx/issues/3126
// Verifies that the process exits cleanly when a background thread is
// performing GPU work and the main thread exits.

#include <iostream>
#include <thread>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  std::thread t([] {
    auto a = mx::random::normal({2048, 2048});
    std::cout << "START" << std::endl;
    for (int i = 0; i < 1000; ++i) {
      a = mx::matmul(a, a);
      // Eval periodically to avoid building a huge graph
      if (i % 10 == 0) {
        mx::eval(a);
        std::cout << "Step " << i << std::endl;
      }
    }
    mx::eval(a);
    std::cout << "Done: " << a.shape(0) << "x" << a.shape(1) << std::endl;
  });

  sleep(1);
  t.detach();
  std::cout << "Main thread exiting." << std::endl;
  mx::metal::set_enabled(false);
  return 0;
}
