// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/array.h"

namespace mlx::core {

namespace {

template <typename T>
void arange(T start, T next, array& out, size_t size) {
  auto ptr = out.data<T>();
  auto step_size = next - start;
  for (int i = 0; i < size; ++i) {
    ptr[i] = start;
    start += step_size;
  }
}

} // namespace

void arange(
    const std::vector<array>& inputs,
    array& out,
    double start,
    double step) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  switch (out.dtype()) {
    case bool_:
      throw std::runtime_error("Bool type unsupported for arange.");
      break;
    case uint8:
      arange<uint8_t>(start, start + step, out, out.size());
      break;
    case uint16:
      arange<uint16_t>(start, start + step, out, out.size());
      break;
    case uint32:
      arange<uint32_t>(start, start + step, out, out.size());
      break;
    case uint64:
      arange<uint64_t>(start, start + step, out, out.size());
      break;
    case int8:
      arange<int8_t>(start, start + step, out, out.size());
      break;
    case int16:
      arange<int16_t>(start, start + step, out, out.size());
      break;
    case int32:
      arange<int32_t>(start, start + step, out, out.size());
      break;
    case int64:
      arange<int64_t>(start, start + step, out, out.size());
      break;
    case float16:
      arange<float16_t>(start, start + step, out, out.size());
      break;
    case float32:
      arange<float>(start, start + step, out, out.size());
      break;
    case bfloat16:
      arange<bfloat16_t>(start, start + step, out, out.size());
      break;
    case complex64:
      arange<complex64_t>(start, start + step, out, out.size());
      break;
  }
}

} // namespace mlx::core
