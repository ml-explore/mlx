// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/utils.h"

namespace mlx::core {

void set_unary_output_data(const array& in, array& out) {
  if (is_donatable(in, out)) {
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

template <typename T, typename U = T, typename Op>
void unary_op(const T* a, U* out, Op op, size_t shape, size_t stride) {
  for (size_t i = 0; i < shape; i += 1) {
    out[i] = op(*a);
    a += stride;
  }
}

template <typename T, typename U = T, typename Op>
void unary_op(const array& a, array& out, Op op) {
  const T* a_ptr = a.data<T>();
  if (a.flags().contiguous) {
    set_unary_output_data(a, out);
    U* dst = out.data<U>();
    constexpr int N = simd::max_size<T>;
    size_t size = a.data_size();
    while (size >= N) {
      simd::store(dst, op(simd::load<T, N>(a_ptr)));
      size -= N;
      a_ptr += N;
      dst += N;
    }
    while (size > 0) {
      *dst = op(*a_ptr);
      size--;
      dst++;
      a_ptr++;
    }
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    U* dst = out.data<U>();
    size_t shape = a.ndim() > 0 ? a.shape(-1) : 1;
    size_t stride = a.ndim() > 0 ? a.strides(-1) : 1;
    if (a.ndim() <= 1) {
      unary_op(a_ptr, dst, op, shape, stride);
      return;
    }
    ContiguousIterator it(a.shape(), a.strides(), a.ndim() - 1);
    for (size_t elem = 0; elem < a.size(); elem += shape) {
      unary_op(a_ptr + it.loc, dst + elem, op, shape, stride);
      it.step();
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
    case float64:
      unary_op<double>(a, out, op);
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
    case float64:
      unary_op<double>(a, out, op);
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

template <typename Op>
void unary_int(const array& a, array& out, Op op) {
  switch (out.dtype()) {
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
    default:
      std::ostringstream err;
      err << "[unary_int] Does not support " << out.dtype();
      throw std::runtime_error(err.str());
  }
}

} // namespace mlx::core
