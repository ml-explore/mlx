// Copyright © 2023 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
void softmax(const array& in, array& out) {
  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();
  int N = in.shape().back();
  int M = in.data_size() / N;
  const T* current_in_ptr;
  T* current_out_ptr;

  for (int i = 0; i < M; i++, in_ptr += N, out_ptr += N) {
    // Find the maximum
    current_in_ptr = in_ptr;
    T maximum = *current_in_ptr;
    for (int j = 0; j < N; j++, current_in_ptr++) {
      maximum = (maximum < *current_in_ptr) ? *current_in_ptr : maximum;
    }

    // Compute the normalizer and the exponentials
    T normalizer = 0;
    current_out_ptr = out_ptr;
    current_in_ptr = in_ptr;
    for (int j = 0; j < N; j++, current_out_ptr++, current_in_ptr++) {
      T expv = std::exp(*current_in_ptr - maximum);
      normalizer += expv;
      *current_out_ptr = expv;
    }
    normalizer = 1 / normalizer;

    // Normalize
    current_out_ptr = out_ptr;
    for (int j = 0; j < N; j++, current_out_ptr++) {
      *current_out_ptr *= normalizer;
    }
  }
}

} // namespace

void Softmax::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // Make sure that the last dimension is contiguous
  auto check_input = [](array x) {
    bool no_copy = x.strides()[x.ndim() - 1] == 1;
    if (x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      return x;
    } else {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      copy(x, x_copy, CopyType::General);
      return x_copy;
    }
  };
  array in = check_input(std::move(inputs[0]));
  if (in.is_donatable()) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(
        allocator::malloc_or_wait(in.data_size() * in.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  }

  switch (in.dtype()) {
    case bool_:
    case uint8:
    case uint16:
    case uint32:
    case uint64:
    case int8:
    case int16:
    case int32:
    case int64:
      throw std::invalid_argument(
          "Softmax is defined only for floating point types");
      break;
    case float32:
      softmax<float>(in, out);
      break;
    case float16:
      softmax<float16_t>(in, out);
      break;
    case bfloat16:
      softmax<bfloat16_t>(in, out);
      break;
    case complex64:
      throw std::invalid_argument(
          "[Softmax] Not yet implemented for complex64");
      break;
  }
}

} // namespace mlx::core
