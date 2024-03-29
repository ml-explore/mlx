// Copyright © 2023 Apple Inc.

#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>

#include "mlx/mlx.h"

#define milliseconds(x) \
  (std::chrono::duration_cast<std::chrono::nanoseconds>(x).count() / 1e6)
#define time_now() std::chrono::high_resolution_clock::now()

#define TIME(FUNC, ...)                                                        \
  std::cout << "Timing " << #FUNC << " ... " << std::flush                     \
            << std::setprecision(5) << time_fn(FUNC, ##__VA_ARGS__) << " msec" \
            << std::endl;

#define TIMEM(MSG, FUNC, ...)                                                  \
  std::cout << "Timing "                                                       \
            << "(" << MSG << ") " << #FUNC << " ... " << std::flush            \
            << std::setprecision(5) << time_fn(FUNC, ##__VA_ARGS__) << " msec" \
            << std::endl;

template <typename F, typename... Args>
double time_fn(F fn, Args&&... args) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    eval(fn(std::forward<Args>(args)...));
  }

  int num_iters = 100;
  auto start = time_now();
  for (int i = 0; i < num_iters; i++) {
    eval(fn(std::forward<Args>(args)...));
  }
  auto end = time_now();
  return milliseconds(end - start) / static_cast<double>(num_iters);
}
