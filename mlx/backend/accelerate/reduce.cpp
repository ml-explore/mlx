// Copyright © 2023 Apple Inc.

#include <cassert>

#include <simd/vector.h>
#include <vecLib/vDSP.h>

#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T, typename VT, int N>
void _vectorized_strided_sum(const T* x, T* accum, int size, size_t stride) {
  for (int i = 0; i < size; i++) {
    size_t s = stride;
    T* a = accum;
    while (s >= N) {
      VT val = (*(VT*)x);
      *(VT*)a += val;
      x += N;
      a += N;
      s -= N;
    }
    while (s-- > 0) {
      *a++ += *x++;
    }
  }
}

// TODO: Add proper templates for the strided reduce algorithm so we don't have
// to write max/min/sum etc.
template <typename T, typename VT, int N>
void _vectorized_strided_max(const T* x, T* accum, int size, size_t stride) {
  for (int i = 0; i < size; i++) {
    size_t s = stride;
    T* a = accum;
    while (s >= N) {
      *(VT*)a = simd_max((*(VT*)x), (*(VT*)a));
      x += N;
      a += N;
      s -= N;
    }
    while (s-- > 0) {
      *a = std::max(*a, *x);
      a++;
      x++;
    }
  }
}

template <typename T, typename VT, int N>
void _vectorized_strided_min(const T* x, T* accum, int size, size_t stride) {
  for (int i = 0; i < size; i++) {
    size_t s = stride;
    T* a = accum;
    while (s >= N) {
      *(VT*)a = simd_min((*(VT*)x), (*(VT*)a));
      x += N;
      a += N;
      s -= N;
    }
    while (s-- > 0) {
      *a = std::min(*a, *x);
      a++;
      x++;
    }
  }
}

template <typename T, typename VT, int N>
void _vectorized_sum(const T* x, T* accum, int size) {
  VT _sum = {0};
  while (size >= N) {
    _sum += (*(VT*)x);
    x += N;
    size -= N;
  }
  T sum = _sum[0];
  for (int i = 1; i < N; i++) {
    sum += _sum[i];
  }
  *accum += sum;
}

void Reduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  if (in.dtype() == float32) {
    if (reduce_type_ == Reduce::Sum) {
      reduction_op<float, float>(
          in,
          out,
          axes_,
          0,
          [](const auto* x, auto* accum, int size, size_t stride) {
            _vectorized_strided_sum<float, simd_float16, 16>(
                (const float*)x, (float*)accum, size, stride);
          },
          [](const auto* x, auto* accum, int size) {
            float acc;
            vDSP_sve((const float*)x, 1, &acc, size);
            (*accum) += acc;
          },
          [](auto* accum, auto x) { *accum += x; });
      return;
    } else if (reduce_type_ == Reduce::Max) {
      reduction_op<float, float>(
          in,
          out,
          axes_,
          -std::numeric_limits<float>::infinity(),
          [](const auto* x, auto* accum, int size, size_t stride) {
            _vectorized_strided_max<float, simd_float16, 16>(
                (const float*)x, (float*)accum, size, stride);
          },
          [](const auto* x, auto* accum, int size) {
            float max;
            vDSP_maxv((const float*)x, 1, &max, size);
            (*accum) = (*accum < max) ? max : *accum;
          },
          [](auto* accum, auto x) { (*accum) = (*accum < x) ? x : *accum; });
      return;
    } else if (reduce_type_ == Reduce::Min) {
      reduction_op<float, float>(
          in,
          out,
          axes_,
          std::numeric_limits<float>::infinity(),
          [](const auto* x, auto* accum, int size, size_t stride) {
            _vectorized_strided_min<float, simd_float16, 16>(
                (const float*)x, (float*)accum, size, stride);
          },
          [](const auto* x, auto* accum, int size) {
            float min;
            vDSP_minv((const float*)x, 1, &min, size);
            (*accum) = (*accum > min) ? min : *accum;
          },
          [](auto* accum, auto x) { (*accum) = (*accum > x) ? x : *accum; });
      return;
    }
  }
  // TODO: Add integer addition and min/max using the templates above and
  //       simd_int16 and friends.
  eval(inputs, out);
}

} // namespace mlx::core
