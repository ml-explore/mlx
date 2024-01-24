// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

struct AbsOp {
  template <typename T>
  T operator()(T x) {
    return std::abs(x);
  }
  uint8_t operator()(uint8_t x) {
    return x;
  }
  uint16_t operator()(uint16_t x) {
    return x;
  }
  uint32_t operator()(uint32_t x) {
    return x;
  }
  uint64_t operator()(uint64_t x) {
    return x;
  }
  bool operator()(bool x) {
    return x;
  }
};

struct SignOp {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  }

  uint8_t operator()(uint8_t x) {
    return x != 0;
  }
  uint16_t operator()(uint16_t x) {
    return x != 0;
  }
  uint32_t operator()(uint32_t x) {
    return x != 0;
  }
  uint64_t operator()(uint64_t x) {
    return x != 0;
  }
};

struct RoundOp {
  template <typename T>
  T operator()(T x) {
    return std::rint(x);
  }

  complex64_t operator()(complex64_t x) {
    return {std::rint(x.real()), std::rint(x.imag())};
  }
};

void set_unary_output_data(const array& in, array& out) {
  if (in.is_donatable() && in.itemsize() == out.itemsize()) {
    out.copy_shared_buffer(in);
  } else {
    auto size = in.data_size();
    out.set_data(
        allocator::malloc_or_wait(size * out.itemsize()),
        size,
        in.strides(),
        in.flags());
  }
}

template <typename T, typename Op>
void unary_op(const array& a, array& out, Op op) {
  const T* a_ptr = a.data<T>();
  if (a.flags().contiguous) {
    set_unary_output_data(a, out);
    T* dst = out.data<T>();
    for (size_t i = 0; i < a.data_size(); ++i) {
      dst[i] = op(a_ptr[i]);
    }
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    T* dst = out.data<T>();
    for (size_t i = 0; i < out.size(); ++i) {
      // TODO this is super inefficient, need to fix.
      int a_idx = elem_to_loc(i, a.shape(), a.strides());
      dst[i] = op(a_ptr[a_idx]);
    }
  }
}

template <typename Op>
void unary(const array& a, array& out, Op op) {
  switch (out.dtype()) {
    case bool_:
      unary_op<bool>(a, out, op);
      break;
    case uint8:
      unary_op<uint8_t>(a, out, op);
      break;
    case uint16:
      unary_op<uint16_t>(a, out, op);
      break;
    case uint32:
      unary_op<uint32_t>(a, out, op);
      break;
    case uint64:
      unary_op<uint64_t>(a, out, op);
      break;
    case int8:
      unary_op<int8_t>(a, out, op);
      break;
    case int16:
      unary_op<int16_t>(a, out, op);
      break;
    case int32:
      unary_op<int32_t>(a, out, op);
      break;
    case int64:
      unary_op<int64_t>(a, out, op);
      break;
    case float16:
      unary_op<float16_t>(a, out, op);
      break;
    case float32:
      unary_op<float>(a, out, op);
      break;
    case bfloat16:
      unary_op<bfloat16_t>(a, out, op);
      break;
    case complex64:
      unary_op<complex64_t>(a, out, op);
      break;
  }
}

template <typename Op>
void unary_fp(const array& a, array& out, Op op) {
  switch (out.dtype()) {
    case bfloat16:
      unary_op<bfloat16_t>(a, out, op);
      break;
    case float16:
      unary_op<float16_t>(a, out, op);
      break;
    case float32:
      unary_op<float>(a, out, op);
      break;
    case complex64:
      unary_op<complex64_t>(a, out, op);
      break;
    default:
      std::ostringstream err;
      err << "[unary_fp] Does not support " << out.dtype();
      throw std::runtime_error(err.str());
  }
}

} // namespace

} // namespace mlx::core
