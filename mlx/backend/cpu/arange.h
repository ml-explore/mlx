// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/scheduler.h"

namespace mlx::core {

namespace {

template <typename T>
void arange(T start, T next, array& out, size_t size, Stream stream) {
  auto ptr = out.data<T>();
  auto step_size = next - start;
  scheduler::enqueue(stream, [ptr, start, step_size, size]() mutable {
    for (int i = 0; i < size; ++i) {
      ptr[i] = start;
      start += step_size;
    }
  });
}

} // namespace

} // namespace mlx::core
