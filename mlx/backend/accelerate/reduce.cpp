// Copyright © 2023 Apple Inc.

#include <cassert>

#include <simd/vector.h>
#include <vecLib/vDSP.h>

#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, typename VT>
struct MinReduction {
  T operator()(const T& a, const T& b) {
    return std::min(a, b);
  }

  VT operator()(VT a, VT b) {
    return simd_min(a, b);
  }
};

template <typename T, typename VT>
struct MaxReduction {
  T operator()(const T& a, const T& b) {
    return std::max(a, b);
  }

  VT operator()(VT a, VT b) {
    return simd_max(a, b);
  }
};

template <typename T, typename VT>
struct SumReduction {
  T operator()(const T& a, const T& b) {
    return a + b;
  }

  VT operator()(VT a, VT b) {
    return a + b;
  }
};

template <typename T, typename VT, int N, typename Reduction>
struct StridedReduce {
  void operator()(const T* x, T* accum, int size, size_t stride) {
    Reduction op;

    for (int i = 0; i < size; i++) {
      size_t s = stride;
      T* a = accum;
      while (s >= N) {
        *(VT*)a = op((*(VT*)x), (*(VT*)a));
        x += N;
        a += N;
        s -= N;
      }
      while (s-- > 0) {
        *a = op(*a, *x);
        a++;
        x++;
      }
    }
  }
};

} // namespace

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
          StridedReduce<
              float,
              simd_float16,
              16,
              SumReduction<float, simd_float16>>(),
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
          StridedReduce<
              float,
              simd_float16,
              16,
              MaxReduction<float, simd_float16>>(),
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
          StridedReduce<
              float,
              simd_float16,
              16,
              MinReduction<float, simd_float16>>(),
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
