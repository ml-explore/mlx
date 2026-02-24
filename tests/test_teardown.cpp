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
  if (!mx::metal::is_available()) {
    std::cout << "Metal not available, skipping." << std::endl;
    return 0;
  }

  std::thread t([] {
    auto a = mx::random::normal({2048, 2048});
    for (int i = 0; i < 1000; ++i) {
      a = mx::matmul(a, a);
      if (i % 10 == 0) {
        mx::eval(a);
      }
    }
    mx::eval(a);
  });

  // Give the background thread time to start GPU work,
  // then exit while it is still running.
  sleep(1);
  mx::metal::set_enabled(false);
  t.detach();
  return 0;
}
