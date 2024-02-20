// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

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
