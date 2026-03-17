// Copyright © 2023-2026 Apple Inc.

#pragma once

#include "mlx/backend/common/unary.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/utils.h"

namespace mlx::core {

template <typename T, typename U = T, typename Op>
void unary_op(const T* a, U* out, size_t shape, size_t stride) {
  for (size_t i = 0; i < shape; i += 1) {
    out[i] = Op{}(*a);
    a += stride;
  }
}

// Helper to process a contiguous chunk of unary op with SIMD
template <typename T, typename U, typename Op>
void unary_op_contiguous_chunk(const T* src, U* dst, size_t size) {
  constexpr int N = std::min(simd::max_size<T>, simd::max_size<U>);
  while (size >= N) {
    simd::store(dst, simd::Simd<U, N>(Op{}(simd::load<T, N>(src))));
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
}

template <typename T, typename U = T, typename Op>
void unary_op(const array& a, array& out, Op) {
  const T* src = a.data<T>();
  U* dst = out.data<U>();
  auto ndim = a.ndim();
  if (a.flags().contiguous) {
    auto size = a.data_size();

    // Check if parallelization is beneficial
    auto& pool = cpu::ThreadPool::instance();
    int n_threads = cpu::effective_threads(size, pool.max_threads());

    if (n_threads > 1) {
      // Parallel path for large contiguous arrays
      pool.parallel_for(n_threads, [&](int tid, int nth) {
        size_t chunk = (size + nth - 1) / nth;
        size_t start = chunk * tid;
        size_t end = std::min(start + chunk, size);
        if (start < end) {
          unary_op_contiguous_chunk<T, U, Op>(
              src + start, dst + start, end - start);
        }
      });
    } else {
      // Single-threaded path
      unary_op_contiguous_chunk<T, U, Op>(src, dst, size);
    }
  } else {
    size_t inner_shape = ndim > 0 ? a.shape().back() : 1;
    size_t inner_stride = ndim > 0 ? a.strides().back() : 1;
    if (ndim <= 1) {
      unary_op<T, U, Op>(src, dst, inner_shape, inner_stride);
      return;
    }

    size_t num_iterations = a.size() / inner_shape;

    // Check if parallelization is beneficial
    auto& pool = cpu::ThreadPool::instance();
    int n_threads = cpu::effective_threads(a.size(), pool.max_threads());

    if (n_threads > 1 && num_iterations >= static_cast<size_t>(n_threads)) {
      // Parallel path for strided arrays
      pool.parallel_for(n_threads, [&](int tid, int nth) {
        size_t chunk = (num_iterations + nth - 1) / nth;
        size_t start_iter = chunk * tid;
        size_t end_iter = std::min(start_iter + chunk, num_iterations);

        if (start_iter >= end_iter) {
          return;
        }

        ContiguousIterator it(a.shape(), a.strides(), ndim - 1);
        it.seek(static_cast<int64_t>(start_iter));

        for (size_t iter = start_iter; iter < end_iter; ++iter) {
          unary_op<T, U, Op>(
              src + it.loc,
              dst + iter * inner_shape,
              inner_shape,
              inner_stride);
          it.step();
        }
      });
    } else {
      // Sequential path
      auto it = ContiguousIterator(a.shape(), a.strides(), ndim - 1);
      for (size_t elem = 0; elem < a.size(); elem += inner_shape) {
        unary_op<T, U, Op>(src + it.loc, dst + elem, inner_shape, inner_stride);
        it.step();
      }
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
