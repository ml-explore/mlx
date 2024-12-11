// Copyright © 2023 Apple Inc.

#include <chrono>
#include <cmath>
#include <iostream>

#include "mlx/mlx.h"
#include "timer.h"

/**
 * An example of linear regression with MLX.
 */
namespace mx = mlx::core;

int main() {
  int num_features = 100;
  int num_examples = 1'000;
  int num_iters = 10'000;
  float learning_rate = 0.01;

  // True parameters
  auto w_star = mx::random::normal({num_features});

  // The input examples (design matrix)
  auto X = mx::random::normal({num_examples, num_features});

  // Noisy labels
  auto eps = 1e-2 * mx::random::normal({num_examples});
  auto y = mx::matmul(X, w_star) + eps;

  // Initialize random parameters
  mx::array w = 1e-2 * mx::random::normal({num_features});

  auto loss_fn = [&](mx::array w) {
    auto yhat = mx::matmul(X, w);
    return (0.5f / num_examples) * mx::sum(mx::square(yhat - y));
  };

  auto grad_fn = mx::grad(loss_fn);

  auto tic = timer::time();
  for (int it = 0; it < num_iters; ++it) {
    auto grads = grad_fn(w);
    w = w - learning_rate * grads;
    mx::eval(w);
  }
  auto toc = timer::time();

  auto loss = loss_fn(w);
  auto error_norm = std::sqrt(mx::sum(mx::square(w - w_star)).item<float>());
  auto throughput = num_iters / timer::seconds(toc - tic);
  std::cout << "Loss " << loss << ", |w - w*| = " << error_norm
            << ", Throughput " << throughput << " (it/s)." << std::endl;
}
