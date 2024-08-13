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
void reduce_dispatch_out(
    const array& in,
    array& out,
    Reduce::ReduceType rtype,
    const std::vector<int>& axes) {
  switch (rtype) {
    case Reduce::And: {
      reduction_op<InT, bool>(in, out, axes, true, AndReduce());
      break;
    }
    case Reduce::Or: {
      reduction_op<InT, bool>(in, out, axes, false, OrReduce());
      break;
    }
    case Reduce::Sum: {
      auto op = [](auto y, auto x) { (*y) = (*y) + x; };
      if (out.dtype() == int32) {
        // special case since the input type can be bool
        reduction_op<InT, int32_t>(in, out, axes, 0, op);
      } else {
        reduction_op<InT, InT>(in, out, axes, 0, op);
      }
      break;
    }
    case Reduce::Prod: {
      auto op = [](auto y, auto x) { (*y) *= x; };
      reduction_op<InT, InT>(in, out, axes, 1, op);
      break;
    }
    case Reduce::Max: {
      auto init = Limits<InT>::min;
      reduction_op<InT, InT>(in, out, axes, init, MaxReduce());
      break;
    }
    case Reduce::Min: {
      auto init = Limits<InT>::max;
      reduction_op<InT, InT>(in, out, axes, init, MinReduce());
      break;
    }
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
  switch (in.dtype()) {
    case bool_:
      reduce_dispatch_out<bool>(in, out, reduce_type_, axes_);
      break;
    case uint8:
      reduce_dispatch_out<uint8_t>(in, out, reduce_type_, axes_);
      break;
    case uint16:
      reduce_dispatch_out<uint16_t>(in, out, reduce_type_, axes_);
      break;
    case uint32:
      reduce_dispatch_out<uint32_t>(in, out, reduce_type_, axes_);
      break;
    case uint64:
      reduce_dispatch_out<uint64_t>(in, out, reduce_type_, axes_);
      break;
    case int8:
      reduce_dispatch_out<uint8_t>(in, out, reduce_type_, axes_);
      break;
    case int16:
      reduce_dispatch_out<uint16_t>(in, out, reduce_type_, axes_);
      break;
    case int32:
      reduce_dispatch_out<int32_t>(in, out, reduce_type_, axes_);
      break;
    case int64:
      reduce_dispatch_out<int64_t>(in, out, reduce_type_, axes_);
      break;
    case float16:
      reduce_dispatch_out<float16_t>(in, out, reduce_type_, axes_);
      break;
    case float32:
      reduce_dispatch_out<float>(in, out, reduce_type_, axes_);
      break;
    case bfloat16:
      reduce_dispatch_out<bfloat16_t>(in, out, reduce_type_, axes_);
      break;
    case complex64:
      reduce_dispatch_out<complex64_t>(in, out, reduce_type_, axes_);
      break;
  }
}

} // namespace mlx::core
