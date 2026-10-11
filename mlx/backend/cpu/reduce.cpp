// Copyright © 2023 Apple Inc.

#include <bit>
#include <cassert>
#include <functional>
#include <limits>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename U>
struct Limits {
  static const U max;
  static const U min;
};

#define instantiate_default_limit(type)                           \
  template <>                                                     \
  struct Limits<type> {                                           \
    static constexpr type max = std::numeric_limits<type>::max(); \
    static constexpr type min = std::numeric_limits<type>::min(); \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type) \
  template <>                         \
  struct Limits<type> {               \
    static const type max;            \
    static const type min;            \
  };

instantiate_float_limit(float16_t);
instantiate_float_limit(bfloat16_t);
instantiate_float_limit(float);
instantiate_float_limit(double);
instantiate_float_limit(complex64_t);

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

const float Limits<float>::max = std::numeric_limits<float>::infinity();
const float Limits<float>::min = -std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::max =
    std::numeric_limits<float>::infinity();
const bfloat16_t Limits<bfloat16_t>::min =
    -std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::max = std::numeric_limits<float>::infinity();
const float16_t Limits<float16_t>::min =
    -std::numeric_limits<float>::infinity();
const double Limits<double>::max = std::numeric_limits<double>::infinity();
const double Limits<double>::min = -std::numeric_limits<double>::infinity();
const complex64_t Limits<complex64_t>::max =
    std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::min =
    -std::numeric_limits<float>::infinity();

struct SumReduce;

template <typename T, typename U, typename Op>
constexpr bool use_pairwise_sum =
    std::is_same_v<T, float> && std::is_same_v<U, float> &&
    std::is_same_v<Op, SumReduce> && simd::max_size<float> == 1;

template <bool Strided = false>
float pairwise_sum(const float* x, int64_t size, int64_t stride = 1) {
  const auto step = Strided ? stride : 1;
  if constexpr (Strided) {
    if (stride == 0 && size != 0) {
      return 0.0f + x[0] * static_cast<float>(size);
    }
  }
  if (size > 128) {
    auto mid = size / 2;
    return pairwise_sum<Strided>(x, mid, step) +
        pairwise_sum<Strided>(x + mid * step, size - mid, step);
  }
  float v[8] = {};
  int64_t i = 0;
  for (; i + 8 <= size; i += 8) {
    for (int j = 0; j < 8; j++) {
      v[j] += x[(i + j) * step];
    }
  }
  float out = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
  for (; i < size; i++) {
    out += x[i * step];
  }
  return out;
}

void pairwise_strided_sum(
    const float* x,
    float* out,
    int size,
    size_t stride,
    float* scratch,
    bool accumulate = false) {
  if (size <= 16) {
    if (!accumulate) {
      std::fill_n(out, stride, 0.0f);
    }
    for (int i = 0; i < size; i++) {
      for (size_t j = 0; j < stride; j++) {
        out[j] += x[j];
      }
      x += stride;
    }
    return;
  }
  int mid = size / 2;
  pairwise_strided_sum(x, out, mid, stride, scratch + stride, accumulate);
  pairwise_strided_sum(
      x + mid * stride, scratch, size - mid, stride, scratch + stride);
  for (size_t j = 0; j < stride; j++) {
    out[j] += scratch[j];
  }
}

float pairwise_general_sum(
    const float* x,
    const Shape& shape,
    const Strides& strides,
    size_t dim,
    int64_t start,
    int64_t size) {
  const auto stride = static_cast<int64_t>(strides[dim]);
  if (dim + 1 == shape.size()) {
    return pairwise_sum<true>(x + start * stride, size, stride);
  }
  if (size > 8) {
    auto mid = size / 2;
    return pairwise_general_sum(x, shape, strides, dim, start, mid) +
        pairwise_general_sum(x, shape, strides, dim, start + mid, size - mid);
  }
  float v[8] = {};
  for (int64_t i = 0; i < size; i++) {
    v[i] = pairwise_general_sum(
        x + (start + i) * stride, shape, strides, dim + 1, 0, shape[dim + 1]);
  }
  return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}

class PairwiseBufferSum {
 public:
  PairwiseBufferSum(size_t width, int64_t count)
      : width_(width),
        levels_(std::bit_width(static_cast<uint64_t>(count))),
        values_((levels_ + 1) * width) {}

  float* block() {
    return values_.data() + levels_ * width_;
  }

  void add() {
    float* v = block();
    auto n = count_++;
    size_t level = 0;
    for (; n & 1; n >>= 1, level++) {
      auto p = values_.data() + level * width_;
      for (size_t j = 0; j < width_; j++) {
        v[j] += p[j];
      }
    }
    std::copy_n(v, width_, values_.data() + level * width_);
  }

  void finish(float* out) {
    std::fill_n(out, width_, 0.0f);
    size_t level = 0;
    for (auto n = count_; n; n >>= 1, level++) {
      if (n & 1) {
        auto p = values_.data() + level * width_;
        for (size_t j = 0; j < width_; j++) {
          out[j] += p[j];
        }
      }
    }
  }

 private:
  size_t width_;
  size_t levels_;
  uint64_t count_ = 0;
  std::vector<float> values_;
};

int64_t reduction_count(const Shape& shape) {
  int64_t n = 1;
  for (auto size : shape) {
    n *= size;
  }
  return n;
}

template <typename T, typename U, typename Op>
struct ReductionAccumulator {
  static constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  // Widen to float32 only if the input is float16 (with N=1) or bfloat16 as it
  // improves performance.
  static constexpr bool widen_to_float =
      (std::is_same_v<T, bfloat16_t> ||
       (N == 1 && std::is_same_v<T, float16_t>));

  using type = std::conditional_t<widen_to_float, float, U>;
};

template <typename T, typename U, typename Op>
void strided_reduce(
    const T* x,
    U* accumulator,
    int size,
    size_t stride,
    Op op) {
  if constexpr (use_pairwise_sum<T, U, Op>) {
    if (size <= 16) {
      pairwise_strided_sum(x, accumulator, size, stride, nullptr, true);
      return;
    }
    size_t levels = 0;
    for (auto n = size; n > 16; n = (n + 1) / 2) {
      levels++;
    }
    std::vector<float> values(levels * stride);
    pairwise_strided_sum(x, accumulator, size, stride, values.data(), true);
    return;
  }
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  for (int i = 0; i < size; i++) {
    U* moving_accumulator = accumulator;
    auto s = stride;
    while (s >= N) {
      auto acc = simd::load<U, N>(moving_accumulator);
      auto v = simd::Simd<U, N>(simd::load<T, N>(x));
      simd::store<U, N>(moving_accumulator, op(acc, v));
      moving_accumulator += N;
      x += N;
      s -= N;
    }
    while (s-- > 0) {
      *moving_accumulator = op(*moving_accumulator, *x);
      moving_accumulator++;
      x++;
    }
  }
};

template <typename T, typename U, typename Op>
void contiguous_reduce(
    const T* x,
    U* accumulator,
    int64_t size,
    Op op,
    U init) {
  if constexpr (use_pairwise_sum<T, U, Op>) {
    *accumulator += pairwise_sum(x, size);
    return;
  }
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  simd::Simd<U, N> accumulator_v(init);
  while (size >= N) {
    accumulator_v = op(accumulator_v, simd::Simd<U, N>(simd::load<T, N>(x)));
    x += N;
    size -= N;
  }
  *accumulator = op(*accumulator, op(accumulator_v));
  while (size-- > 0) {
    *accumulator = op(*accumulator, *x);
    x++;
  }
}

// Helper for the ndimensional strided loop
void nd_loop(
    std::function<void(int64_t)> callback,
    const Shape& shape,
    const Strides& strides) {
  std::function<void(int, int64_t)> loop_inner;
  loop_inner = [&](int dim, int64_t offset) {
    if (dim < shape.size() - 1) {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        loop_inner(dim + 1, offset + i * stride);
      }
    } else {
      auto size = shape[dim];
      auto stride = strides[dim];
      for (int i = 0; i < size; i++) {
        callback(offset + i * stride);
      }
    }
  };
  loop_inner(0, 0);
}

template <typename T, typename U, typename Op>
void reduction_op(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init) {
  ReductionPlan plan = get_reduction_plan(x, axes);

  auto in_ptr = x.data<T>();
  auto out_ptr = out.data<U>();
  if (plan.type == ContiguousAllReduce) {
    *out_ptr = init;
    contiguous_reduce(in_ptr, out_ptr, x.size(), Op{}, init);
    return;
  }

  if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape[0];
    for (int i = 0; i < out.size(); i++, out_ptr++, in_ptr += reduction_size) {
      *out_ptr = init;
      contiguous_reduce(in_ptr, out_ptr, reduction_size, Op{}, init);
    }
    return;
  }

  if (plan.type == GeneralContiguousReduce || plan.type == ContiguousReduce) {
    int reduction_size = plan.shape.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    // Unrolling the following loop (and implementing it in order for
    // ContiguousReduce) should hold extra performance boost.
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);
    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int64_t offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        contiguous_reduce(in_ptr + offset, out_ptr, reduction_size, Op{}, init);
      }
    } else {
      for (int i = 0; i < out.size(); i++, out_ptr++) {
        int64_t offset = elem_to_loc(i, shape, strides);
        *out_ptr = init;
        if constexpr (use_pairwise_sum<T, U, Op>) {
          PairwiseBufferSum total(1, reduction_count(plan.shape));
          nd_loop(
              [&](int64_t extra_offset) {
                *total.block() = pairwise_sum(
                    in_ptr + offset + extra_offset, reduction_size);
                total.add();
              },
              plan.shape,
              plan.strides);
          total.finish(out_ptr);
          continue;
        }
        nd_loop(
            [&](int64_t extra_offset) {
              contiguous_reduce(
                  in_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  Op{},
                  init);
            },
            plan.shape,
            plan.strides);
      }
    }
    return;
  }

  if (plan.type == ContiguousStridedReduce && plan.shape.size() == 1) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    for (int i = 0; i < out.size(); i += reduction_stride) {
      std::fill_n(out_ptr, reduction_stride, init);
      strided_reduce(in_ptr, out_ptr, reduction_size, reduction_stride, Op{});
      in_ptr += reduction_stride * reduction_size;
      out_ptr += reduction_stride;
    }
    return;
  }

  if (plan.type == GeneralStridedReduce ||
      plan.type == ContiguousStridedReduce) {
    int reduction_size = plan.shape.back();
    size_t reduction_stride = plan.strides.back();
    plan.shape.pop_back();
    plan.strides.pop_back();
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);

    if (plan.shape.size() == 0) {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int64_t offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        strided_reduce(
            in_ptr + offset, out_ptr, reduction_size, reduction_stride, Op{});
        out_ptr += reduction_stride;
      }
    } else {
      for (int i = 0; i < out.size(); i += reduction_stride) {
        int64_t offset = elem_to_loc(i, shape, strides);
        std::fill_n(out_ptr, reduction_stride, init);
        if constexpr (use_pairwise_sum<T, U, Op>) {
          PairwiseBufferSum total(
              reduction_stride, reduction_count(plan.shape));
          nd_loop(
              [&](int64_t extra_offset) {
                std::fill_n(total.block(), reduction_stride, 0.0f);
                strided_reduce(
                    in_ptr + offset + extra_offset,
                    total.block(),
                    reduction_size,
                    reduction_stride,
                    Op{});
                total.add();
              },
              plan.shape,
              plan.strides);
          total.finish(out_ptr);
          out_ptr += reduction_stride;
          continue;
        }
        nd_loop(
            [&](int64_t extra_offset) {
              strided_reduce(
                  in_ptr + offset + extra_offset,
                  out_ptr,
                  reduction_size,
                  reduction_stride,
                  Op{});
            },
            plan.shape,
            plan.strides);
        out_ptr += reduction_stride;
      }
    }
    return;
  }

  if (plan.type == GeneralReduce) {
    auto [shape, strides] = shapes_without_reduction_axes(x, axes);

    for (int i = 0; i < out.size(); i++, out_ptr++) {
      int64_t offset = elem_to_loc(i, shape, strides);
      U val = init;
      if constexpr (use_pairwise_sum<T, U, Op>) {
        *out_ptr = pairwise_general_sum(
            in_ptr + offset, plan.shape, plan.strides, 0, 0, plan.shape[0]);
        continue;
      }
      nd_loop(
          [&](int64_t extra_offset) {
            val = Op{}(val, *(in_ptr + offset + extra_offset));
          },
          plan.shape,
          plan.strides);
      *out_ptr = val;
    }
  }
}

template <typename T, typename U, typename Op>
void float_reduction(
    const array& x,
    array& out,
    const std::vector<int>& axes,
    U init) {
  using AccT = ReductionAccumulator<T, U, Op>::type;
  if constexpr (std::is_same_v<AccT, U>) {
    reduction_op<T, U, Op>(x, out, axes, init);
  } else {
    array temp(out.shape(), TypeToDtype<AccT>(), nullptr, {});
    temp.set_data(allocator::malloc(temp.nbytes()));
    reduction_op<T, AccT, Op>(x, temp, axes, static_cast<AccT>(init));
    std::copy_n(temp.data<AccT>(), out.size(), out.data<U>());
  }
}

struct AndReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x & static_cast<bool>(y);
  }

  bool operator()(bool x, bool y) {
    return x & y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x & (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x & y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::all(x);
  };
};

struct OrReduce {
  template <typename T>
  bool operator()(bool x, T y) {
    return x | static_cast<bool>(y);
  }

  bool operator()(bool x, bool y) {
    return x | y;
  }

  template <int N, typename T>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<T, N> x) {
    return x | (y != 0);
  };

  template <int N>
  simd::Simd<bool, N> operator()(simd::Simd<bool, N> y, simd::Simd<bool, N> x) {
    return x | y;
  };

  template <int N, typename T>
  bool operator()(simd::Simd<T, N> x) {
    return simd::any(x);
  };
};

struct MaxReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::maximum(x, y);
  };

  template <int N, typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    return simd::max(x);
  };

  template <int N, typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    if (simd::any(x != x)) {
      return static_cast<T>(NAN);
    }
    return simd::max(x);
  };
};

struct MinReduce {
  template <typename T>
  T operator()(T y, T x) {
    return (*this)(simd::Simd<T, 1>(x), simd::Simd<T, 1>(y)).value;
  };

  template <int N, typename T>
  simd::Simd<T, N> operator()(simd::Simd<T, N> y, simd::Simd<T, N> x) {
    return simd::minimum(x, y);
  };

  template <int N, typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    return simd::min(x);
  };

  template <int N, typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(simd::Simd<T, N> x) {
    if (simd::any(x != x)) {
      return static_cast<T>(NAN);
    }
    return simd::min(x);
  };
};

struct SumReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return x + y;
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return y + x;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::sum(x);
  };
};

struct ProdReduce {
  template <typename T, typename U>
  U operator()(U y, T x) {
    return x * y;
  };

  template <int N, typename T, typename U>
  simd::Simd<U, N> operator()(simd::Simd<U, N> y, simd::Simd<T, N> x) {
    return x * y;
  };

  template <int N, typename T>
  T operator()(simd::Simd<T, N> x) {
    return simd::prod(x);
  };
};

template <typename InT>
void reduce_dispatch_and_or(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::And) {
    reduction_op<InT, bool, AndReduce>(in, out, axes, true);
  } else {
    reduction_op<InT, bool, OrReduce>(in, out, axes, false);
  }
}

template <typename InT>
void reduce_dispatch_sum_prod(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Sum) {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t, SumReduce>(in, out, axes, 0);
    } else {
      float_reduction<InT, InT, SumReduce>(in, out, axes, 0);
    }
  } else {
    if constexpr (std::is_integral_v<InT> && sizeof(InT) <= 4) {
      reduction_op<InT, int32_t, ProdReduce>(in, out, axes, 1);
    } else {
      float_reduction<InT, InT, ProdReduce>(in, out, axes, 1);
    }
  }
}

template <typename InT>
void reduce_dispatch_min_max(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Max) {
    auto init = Limits<InT>::min;
    reduction_op<InT, InT, MaxReduce>(in, out, axes, init);
  } else {
    auto init = Limits<InT>::max;
    reduction_op<InT, InT, MinReduce>(in, out, axes, init);
  }
}

namespace {

template <bool Squared, bool Strided = false>
float variance_sum(
    const float* x,
    int64_t size,
    int64_t stride,
    float origin,
    float mu) {
  const auto step = Strided ? stride : 1;
  if constexpr (Strided) {
    if (stride == 0) {
      auto d = (0.5f * x[0] - origin) - mu;
      return (Squared ? d * d : d) * static_cast<float>(size);
    }
  }
  if (size > 128) {
    auto mid = size / 2;
    return variance_sum<Squared, Strided>(x, mid, step, origin, mu) +
        variance_sum<Squared, Strided>(
               x + mid * step, size - mid, step, origin, mu);
  }
  float v[8] = {};
  int64_t i = 0;
  for (; i + 7 < size; i += 8) {
    for (int j = 0; j < 8; ++j) {
      auto d = (0.5f * x[(i + j) * step] - origin) - mu;
      v[j] += Squared ? d * d : d;
    }
  }
  auto total =
      ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
  for (; i < size; ++i) {
    auto d = (0.5f * x[i * step] - origin) - mu;
    total += Squared ? d * d : d;
  }
  return total;
}

template <bool Squared>
float variance_nd(
    const float* x,
    const Shape& shape,
    const Strides& strides,
    int dim,
    int64_t start,
    int64_t size,
    float origin,
    float mu) {
  auto stride = static_cast<int64_t>(strides[dim]);
  if (dim + 1 == shape.size()) {
    x += start * stride;
    return stride == 1
        ? variance_sum<Squared>(x, size, stride, origin, mu)
        : variance_sum<Squared, true>(x, size, stride, origin, mu);
  }
  if (stride == 0) {
    return variance_nd<Squared>(
               x, shape, strides, dim + 1, 0, shape[dim + 1], origin, mu) *
        static_cast<float>(size);
  }
  if (size > 8) {
    auto mid = size / 2;
    return variance_nd<Squared>(
               x, shape, strides, dim, start, mid, origin, mu) +
        variance_nd<Squared>(
               x, shape, strides, dim, start + mid, size - mid, origin, mu);
  }
  float v[8] = {};
  for (int64_t i = 0; i < size; ++i) {
    v[i] = variance_nd<Squared>(
        x + (start + i) * stride,
        shape,
        strides,
        dim + 1,
        0,
        shape[dim + 1],
        origin,
        mu);
  }
  return ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
}

template <bool Squared>
void variance_columns(
    const float* x,
    float* out,
    int64_t size,
    int64_t stride,
    int width,
    const float* origin,
    const float* mu,
    float* scratch) {
  if (size <= 16) {
    std::fill_n(out, width, 0.0f);
    for (int64_t i = 0; i < size; ++i) {
      for (int j = 0; j < width; ++j) {
        auto d = 0.5f * x[i * stride + j] - origin[j];
        if constexpr (Squared) {
          d -= mu[j];
        }
        out[j] += Squared ? d * d : d;
      }
    }
    return;
  }
  auto mid = size / 2;
  variance_columns<Squared>(
      x, out, mid, stride, width, origin, mu, scratch + width);
  variance_columns<Squared>(
      x + mid * stride,
      scratch,
      size - mid,
      stride,
      width,
      origin,
      mu,
      scratch + width);
  for (int j = 0; j < width; ++j) {
    out[j] += scratch[j];
  }
}

void variance(
    const array& in,
    array& out,
    const std::vector<int>& axes,
    int ddof) {
  int64_t count = 1;
  for (auto ax : axes) {
    count *= in.shape(ax);
  }
  auto factor = 4.0f /
      std::max(static_cast<float>(count) - static_cast<float>(ddof), 0.0f);
  auto x = in.data<float>();
  auto dst = out.data<float>();
  if (out.size() == 0) {
    return;
  }
  if (count == 0) {
    std::fill_n(dst, out.size(), 0.0f * factor);
    return;
  }
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  if (count == 1) {
    for (int64_t i = 0; i < out.size(); ++i) {
      auto v = x[elem_to_loc(i, shape, strides)];
      auto d = 0.5f * v - 0.5f * v;
      dst[i] = d * d * factor;
    }
    return;
  }
  auto plan = get_reduction_plan(in, axes);
  if (plan.type == ContiguousAllReduce) {
    auto origin = 0.5f * x[0];
    auto mu = variance_sum<false>(x, count, 1, origin, 0.0f) /
        static_cast<float>(count);
    dst[0] = variance_sum<true>(x, count, 1, origin, mu) * factor;
    return;
  }
  if ((plan.type == ContiguousStridedReduce ||
       plan.type == GeneralStridedReduce) &&
      plan.shape.size() == 1) {
    auto stride = static_cast<int64_t>(plan.strides[0]);
    int levels = 0;
    for (auto n = count; n > 16; n = (n + 1) / 2) {
      ++levels;
    }
    std::vector<float> scratch(levels * 128);
    float origin[128];
    float mu[128];
    for (int64_t i = 0; i < out.size(); i += stride) {
      auto offset = elem_to_loc(i, shape, strides);
      for (int64_t j = 0; j < stride; j += 128) {
        int width = std::min<int64_t>(128, stride - j);
        auto ptr = x + offset + j;
        for (int k = 0; k < width; ++k) {
          origin[k] = 0.5f * ptr[k];
        }
        variance_columns<false>(
            ptr, mu, count, stride, width, origin, nullptr, scratch.data());
        for (int k = 0; k < width; ++k) {
          mu[k] /= static_cast<float>(count);
        }
        variance_columns<true>(
            ptr, dst + i + j, count, stride, width, origin, mu, scratch.data());
        for (int k = 0; k < width; ++k) {
          dst[i + j + k] *= factor;
        }
      }
    }
    return;
  }
  for (int64_t i = 0; i < out.size(); ++i) {
    auto ptr = x + elem_to_loc(i, shape, strides);
    auto origin = 0.5f * ptr[0];
    auto mu =
        variance_nd<false>(
            ptr, plan.shape, plan.strides, 0, 0, plan.shape[0], origin, 0.0f) /
        static_cast<float>(count);
    dst[i] =
        variance_nd<true>(
            ptr, plan.shape, plan.strides, 0, 0, plan.shape[0], origin, mu) *
        factor;
  }
}

} // namespace

void Variance::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  assert(inputs[0].dtype() == float32);
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(inputs[0]);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(inputs[0]),
                    out = array::unsafe_weak_copy(out),
                    axes = axes_,
                    ddof = ddof_]() mutable { variance(in, out, axes, ddof); });
}

void Reduce::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    reduce_type_ = reduce_type_,
                    axes_ = axes_]() mutable {
    switch (reduce_type_) {
      case Reduce::And:
      case Reduce::Or: {
        // Integers can be reduced as whatever type has the same width, since
        // only their bits matter. Floats cannot: -0.0 compares equal to zero
        // but has a bit set, so it has to be tested as a float.
        switch (in.dtype()) {
          case bool_:
          case uint8:
          case int8:
            reduce_dispatch_and_or<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
          case uint16:
            reduce_dispatch_and_or<int16_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_and_or<float16_t>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_and_or<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
          case int32:
            reduce_dispatch_and_or<int32_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_and_or<float>(in, out, reduce_type_, axes_);
            break;
          case uint64:
          case int64:
            reduce_dispatch_and_or<int64_t>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_and_or<complex64_t>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_and_or<double>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
      case Reduce::Sum:
      case Reduce::Prod: {
        switch (in.dtype()) {
          case bool_:
          case uint8:
            reduce_dispatch_sum_prod<uint8_t>(in, out, reduce_type_, axes_);
            break;
          case uint16:
            reduce_dispatch_sum_prod<uint16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
            reduce_dispatch_sum_prod<uint32_t>(in, out, reduce_type_, axes_);
            break;
          case uint64:
            reduce_dispatch_sum_prod<uint64_t>(in, out, reduce_type_, axes_);
            break;
          case int8:
            reduce_dispatch_sum_prod<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
            reduce_dispatch_sum_prod<int16_t>(in, out, reduce_type_, axes_);
            break;
          case int32:
            reduce_dispatch_sum_prod<int32_t>(in, out, reduce_type_, axes_);
            break;
          case int64:
            reduce_dispatch_sum_prod<int64_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_sum_prod<float16_t>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_sum_prod<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_sum_prod<float>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_sum_prod<double>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_sum_prod<complex64_t>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
      case Reduce::Max:
      case Reduce::Min: {
        switch (in.dtype()) {
          case bool_:
            reduce_dispatch_min_max<bool>(in, out, reduce_type_, axes_);
            break;
          case uint8:
            reduce_dispatch_min_max<uint8_t>(in, out, reduce_type_, axes_);
            break;
          case uint16:
            reduce_dispatch_min_max<uint16_t>(in, out, reduce_type_, axes_);
            break;
          case uint32:
            reduce_dispatch_min_max<uint32_t>(in, out, reduce_type_, axes_);
            break;
          case uint64:
            reduce_dispatch_min_max<uint64_t>(in, out, reduce_type_, axes_);
            break;
          case int8:
            reduce_dispatch_min_max<int8_t>(in, out, reduce_type_, axes_);
            break;
          case int16:
            reduce_dispatch_min_max<int16_t>(in, out, reduce_type_, axes_);
            break;
          case int32:
            reduce_dispatch_min_max<int32_t>(in, out, reduce_type_, axes_);
            break;
          case int64:
            reduce_dispatch_min_max<int64_t>(in, out, reduce_type_, axes_);
            break;
          case float16:
            reduce_dispatch_min_max<float16_t>(in, out, reduce_type_, axes_);
            break;
          case float32:
            reduce_dispatch_min_max<float>(in, out, reduce_type_, axes_);
            break;
          case float64:
            reduce_dispatch_min_max<double>(in, out, reduce_type_, axes_);
            break;
          case bfloat16:
            reduce_dispatch_min_max<bfloat16_t>(in, out, reduce_type_, axes_);
            break;
          case complex64:
            reduce_dispatch_min_max<complex64_t>(in, out, reduce_type_, axes_);
            break;
        }
        break;
      }
    }
  });
}

} // namespace mlx::core
