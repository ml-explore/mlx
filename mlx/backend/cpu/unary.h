// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/unary.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/utils.h"

namespace mlx::core {

template <typename T, typename U = T, typename Op>
void unary_op(const T* a, U* out, size_t shape, size_t stride) {
  for (size_t i = 0; i < shape; i += 1) {
    out[i] = Op{}(*a);
    a += stride;
  }
}

template <typename T, typename U = T, typename Op>
void unary_op(const array& a, array& out, Op) {
  const T* src = a.data<T>();
  U* dst = out.data<U>();
  auto ndim = a.ndim();
  if (a.flags().contiguous) {
    auto size = a.data_size();
    constexpr int N = simd::max_size<T>;
    while (size >= N) {
      simd::store(dst, Op{}(simd::load<T, N>(src)));
      size -= N;
      src += N;
      dst += N;
    }
    while (size > 0) {
      *dst = Op{}(*src);
      size--;
      dst++;
      src++;
    }
  } else {
    size_t shape = ndim > 0 ? a.shape().back() : 1;
    size_t stride = ndim > 0 ? a.strides().back() : 1;
    if (ndim <= 1) {
      unary_op<T, U, Op>(src, dst, shape, stride);
      return;
    }
    auto it = ContiguousIterator(a.shape(), a.strides(), ndim - 1);
    for (size_t elem = 0; elem < a.size(); elem += shape) {
      unary_op<T, U, Op>(src + it.loc, dst + elem, shape, stride);
      it.step();
    }
  }
}

template <typename Op>
void unary(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable {
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
  });
}

template <typename Op>
void unary_real_fp(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable {
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
      default:
        std::ostringstream err;
        err << "[unary_real] Does not support " << out.dtype();
        throw std::runtime_error(err.str());
    }
  });
}
template <typename Op>
void unary_fp(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable {
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
  });
}

template <typename Op>
void unary_signed(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable {
    switch (out.dtype()) {
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
      default:
        throw std::runtime_error("[Abs] Called on unsigned type");
    }
  });
}

template <typename Op>
void unary_complex(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable { unary_op<complex64_t>(a, out, op); });
}

template <typename Op>
void unary_complex_to_float(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch(
      [a = array::unsafe_weak_copy(a),
       out = array::unsafe_weak_copy(out),
       op = op]() mutable { unary_op<complex64_t, float>(a, out, op); });
}

template <typename Op>
void unary_int(const array& a, array& out, Op op, Stream stream) {
  set_unary_output_data(a, out);
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    out = array::unsafe_weak_copy(out),
                    op = op]() mutable {
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
  });
}

} // namespace mlx::core
