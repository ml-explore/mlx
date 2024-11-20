// Copyright © 2023 Apple Inc.

#include <cassert>
#include <functional>
#include <limits>

#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

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
const complex64_t Limits<complex64_t>::max =
    std::numeric_limits<float>::infinity();
const complex64_t Limits<complex64_t>::min =
    -std::numeric_limits<float>::infinity();

struct AndReduce {
  template <typename T>
  void operator()(bool* a, T b) {
    (*a) &= (b != 0);
  }

  void operator()(bool* y, bool x) {
    (*y) &= x;
  }
};

struct OrReduce {
  template <typename T>
  void operator()(bool* a, T b) {
    (*a) |= (b != 0);
  }

  void operator()(bool* y, bool x) {
    (*y) |= x;
  }
};

struct MaxReduce {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>> operator()(T* y, T x) {
    (*y) = (*y > x) ? *y : x;
  };

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>> operator()(T* y, T x) {
    if (std::isnan(x)) {
      *y = x;
    } else {
      (*y) = (*y > x) ? *y : x;
    }
  };
};

struct MinReduce {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>> operator()(T* y, T x) {
    (*y) = (*y < x) ? *y : x;
  };

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>> operator()(T* y, T x) {
    if (std::isnan(x)) {
      *y = x;
    } else {
      (*y) = (*y < x) ? *y : x;
    }
  };
};

template <typename InT>
void reduce_dispatch_and_or(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::And) {
    reduction_op<InT, bool>(in, out, axes, true, AndReduce());
  } else {
    reduction_op<InT, bool>(in, out, axes, false, OrReduce());
  }
}

template <typename InT>
void reduce_dispatch_sum_prod(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  if (rtype == Reduce::Sum) {
    auto op = [](auto y, auto x) { (*y) = (*y) + x; };
    if (out.dtype() == int32) {
      // special case since the input type can be bool
      reduction_op<InT, int32_t>(in, out, axes, 0, op);
    } else {
      reduction_op<InT, InT>(in, out, axes, 0, op);
    }
  } else {
    auto op = [](auto y, auto x) { (*y) *= x; };
    reduction_op<InT, InT>(in, out, axes, 1, op);
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
    reduction_op<InT, InT>(in, out, axes, init, MaxReduce());
  } else {
    auto init = Limits<InT>::max;
    reduction_op<InT, InT>(in, out, axes, init, MinReduce());
  }
}

} // namespace

void nd_loop(
    std::function<void(int)> callback,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
  std::function<void(int, int)> loop_inner;
  loop_inner = [&](int dim, int offset) {
    if (dim < shape.size() - 1) {
      int size = shape[dim];
      size_t stride = strides[dim];
      for (int i = 0; i < size; i++) {
        loop_inner(dim + 1, offset + i * stride);
      }
    } else {
      int size = shape[dim];
      size_t stride = strides[dim];
      for (int i = 0; i < size; i++) {
        callback(offset + i * stride);
      }
    }
  };
  loop_inner(0, 0);
}

void Reduce::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  switch (reduce_type_) {
    case Reduce::And:
    case Reduce::Or: {
      switch (in.dtype()) {
        case bool_:
        case uint8:
        case int8:
          reduce_dispatch_and_or<int8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
        case uint16:
        case float16:
        case bfloat16:
          reduce_dispatch_and_or<int16_t>(in, out, reduce_type_, axes_);
          break;
        case uint32:
        case int32:
        case float32:
          reduce_dispatch_and_or<int32_t>(in, out, reduce_type_, axes_);
          break;
        case uint64:
        case int64:
        case complex64:
          reduce_dispatch_and_or<int64_t>(in, out, reduce_type_, axes_);
          break;
      }
      break;
    }
    case Reduce::Sum:
    case Reduce::Prod: {
      switch (in.dtype()) {
        case bool_:
        case uint8:
        case int8:
          reduce_dispatch_sum_prod<int8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
        case uint16:
          reduce_dispatch_sum_prod<int16_t>(in, out, reduce_type_, axes_);
          break;
        case int32:
        case uint32:
          reduce_dispatch_sum_prod<int32_t>(in, out, reduce_type_, axes_);
          break;
        case int64:
        case uint64:
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
          reduce_dispatch_min_max<uint8_t>(in, out, reduce_type_, axes_);
          break;
        case int16:
          reduce_dispatch_min_max<uint16_t>(in, out, reduce_type_, axes_);
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
}

} // namespace mlx::core
