// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

using namespace mlx::core::simd;

template <typename T, typename AccT>
void softmax(const array& in, array& out) {
  constexpr bool same_t = std::is_same_v<T, AccT>;
  constexpr int N = std::min(max_size<AccT>, max_size<T>);

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();
  int M = in.shape().back();
  int L = in.data_size() / M;
  const T* current_in_ptr;
  T* current_out_ptr;

  for (int i = 0; i < L; i++, in_ptr += M, out_ptr += M) {
    // Find the maximum
    current_in_ptr = in_ptr;
    Simd<AccT, N> vmaximum(-std::numeric_limits<float>::infinity());
    size_t s = M;
    while (s >= N) {
      Simd<AccT, N> vals = load<T, N>(current_in_ptr);
      vmaximum = maximum(vals, vmaximum);
      current_in_ptr += N;
      s -= N;
    }

    AccT maximum = max(vmaximum);
    while (s-- > 0) {
      maximum = std::max(maximum, static_cast<AccT>(*current_in_ptr));
      current_in_ptr++;
    }

    // Compute the normalizer and the exponentials
    Simd<AccT, N> vnormalizer(0.0);
    current_out_ptr = out_ptr;
    current_in_ptr = in_ptr;
    s = M;
    while (s >= N) {
      Simd<AccT, N> vexp = load<T, N>(current_in_ptr);
      vexp = exp(vexp - maximum);
      if constexpr (same_t) {
        store(current_out_ptr, vexp);
      }
      vnormalizer = vnormalizer + vexp;
      current_in_ptr += N;
      current_out_ptr += N;
      s -= N;
    }
    AccT normalizer = sum(vnormalizer);
    while (s-- > 0) {
      AccT _exp = std::exp(*current_in_ptr - maximum);
      if constexpr (same_t) {
        *current_out_ptr = _exp;
      }
      normalizer += _exp;
      current_in_ptr++;
      current_out_ptr++;
    }
    normalizer = 1 / normalizer;

    // Normalize
    current_out_ptr = out_ptr;
    current_in_ptr = in_ptr;
    s = M;
    while (s >= N) {
      if constexpr (same_t) {
        store(
            current_out_ptr,
            Simd<T, N>(load<T, N>(current_out_ptr) * normalizer));
      } else {
        Simd<AccT, N> vexp = load<T, N>(current_in_ptr);
        vexp = exp(vexp - maximum) * normalizer;
        store(current_out_ptr, Simd<T, N>(vexp));
        current_in_ptr += N;
      }
      current_out_ptr += N;
      s -= N;
    }
    while (s-- > 0) {
      if constexpr (same_t) {
        *current_out_ptr *= normalizer;
      } else {
        AccT _exp = std::exp(*current_in_ptr - maximum);
        *current_out_ptr = static_cast<T>(_exp * normalizer);
        current_in_ptr++;
      }
      current_out_ptr++;
    }
  }
}

} // namespace

void Softmax::eval_cpu(const std::vector<array>& inputs, array& out) {
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
      throw std::runtime_error(
          "Softmax is defined only for floating point types");
      break;
    case float32:
      softmax<float, float>(in, out);
      break;
    case float16:
      if (precise_) {
        softmax<float16_t, float>(in, out);
      } else {
        softmax<float16_t, float16_t>(in, out);
      }
      break;
    case bfloat16:
      if (precise_) {
        softmax<bfloat16_t, float>(in, out);
      } else {
        softmax<bfloat16_t, bfloat16_t>(in, out);
      }
      break;
    case complex64:
      throw std::invalid_argument(
          "[Softmax] Not yet implemented for complex64");
      break;
  }
}

} // namespace mlx::core
