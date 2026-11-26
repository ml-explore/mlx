// Copyright © 2026 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

// Number of independent accumulators for the row reductions. Several
// accumulators break the dependency chain of a single running sum and let
// the compiler vectorize the loop without reassociating floating point math.
constexpr int kAcc = 8;

// Sum of x[i] (or of (x[i] - shift)^2 when Square is true) over a row.
template <bool Square, typename T, typename AccT>
AccT row_sum(const T* x, int n, AccT shift) {
  AccT acc[kAcc] = {};
  int i = 0;
  for (; i + kAcc <= n; i += kAcc) {
    for (int k = 0; k < kAcc; k++) {
      AccT v = static_cast<AccT>(x[i + k]);
      if constexpr (Square) {
        v -= shift;
        acc[k] += v * v;
      } else {
        acc[k] += v;
      }
    }
  }
  for (int k = 1; k < kAcc; k++) {
    acc[0] += acc[k];
  }
  for (; i < n; i++) {
    AccT v = static_cast<AccT>(x[i]);
    if constexpr (Square) {
      v -= shift;
      acc[0] += v * v;
    } else {
      acc[0] += v;
    }
  }
  return acc[0];
}

// Make the last axis of x contiguous and set up the output so that it has
// the same layout as the (possibly copied) input.
array prepare_rows(const array& x, array& out, Stream s) {
  if (x.flags().row_contiguous) {
    if (x.is_donatable() && x.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(x);
    } else {
      out.set_data(
          allocator::malloc(x.data_size() * x.itemsize()),
          x.data_size(),
          x.strides(),
          x.flags());
    }
    return x;
  }
  array x_copy = contiguous_copy_cpu(x, s);
  out.copy_shared_buffer(x_copy);
  return x_copy;
}

template <typename T, typename AccT>
void rms_norm(
    const array& x,
    const array& w,
    array& out,
    float eps,
    Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_output_array(out);

  int M = x.shape().back();
  int64_t L = M == 0 ? 0 : x.data_size() / M;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  encoder.dispatch([x_ptr = x.data<T>(),
                    w_ptr = w.data<T>(),
                    out_ptr = out.data<T>(),
                    M,
                    L,
                    w_stride,
                    eps]() {
    for (int64_t r = 0; r < L; r++) {
      const T* xr = x_ptr + r * M;
      T* outr = out_ptr + r * M;
      AccT ss = row_sum<true, T, AccT>(xr, M, AccT(0));
      AccT n = AccT(1) /
          std::sqrt(ss / static_cast<AccT>(M) + static_cast<AccT>(eps));
      // Same rounding as the composed fallback: normalize in AccT,
      // round to T, then scale by the weight in T.
      if (w_stride == 1) {
        for (int i = 0; i < M; i++) {
          outr[i] = static_cast<T>(static_cast<AccT>(xr[i]) * n) * w_ptr[i];
        }
      } else {
        for (int i = 0; i < M; i++) {
          outr[i] = static_cast<T>(static_cast<AccT>(xr[i]) * n) *
              w_ptr[i * w_stride];
        }
      }
    }
  });
}

template <typename T, typename AccT>
void layer_norm(
    const array& x,
    const array& w,
    const array& b,
    array& out,
    float eps,
    Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  int M = x.shape().back();
  int64_t L = M == 0 ? 0 : x.data_size() / M;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  int64_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  encoder.dispatch([x_ptr = x.data<T>(),
                    w_ptr = w.data<T>(),
                    b_ptr = b.data<T>(),
                    out_ptr = out.data<T>(),
                    M,
                    L,
                    w_stride,
                    b_stride,
                    eps]() {
    for (int64_t r = 0; r < L; r++) {
      const T* xr = x_ptr + r * M;
      T* outr = out_ptr + r * M;
      // Two passes over the row (mean, then centered variance), like
      // the composed fallback. The row is in cache for the second.
      AccT mu = row_sum<false, T, AccT>(xr, M, AccT(0)) / static_cast<AccT>(M);
      AccT var = row_sum<true, T, AccT>(xr, M, mu) / static_cast<AccT>(M);
      AccT n = AccT(1) / std::sqrt(var + static_cast<AccT>(eps));
      if (w_stride == 1 && b_stride == 1) {
        for (int i = 0; i < M; i++) {
          T y = static_cast<T>((static_cast<AccT>(xr[i]) - mu) * n);
          outr[i] = static_cast<T>(y * w_ptr[i]) + b_ptr[i];
        }
      } else {
        for (int i = 0; i < M; i++) {
          T y = static_cast<T>((static_cast<AccT>(xr[i]) - mu) * n);
          outr[i] =
              static_cast<T>(y * w_ptr[i * w_stride]) + b_ptr[i * b_stride];
        }
      }
    }
  });
}

template <typename F>
void dispatch_norm_type(Dtype dtype, const char* name, F&& f) {
  switch (dtype) {
    case float32:
      f(float{}, float{});
      break;
    case float16:
      f(float16_t{}, float{});
      break;
    case bfloat16:
      f(bfloat16_t{}, float{});
      break;
    case float64:
      f(double{}, double{});
      break;
    default: {
      std::ostringstream msg;
      msg << "[" << name << "] Unsupported type " << dtype << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

} // namespace

void RMSNorm::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  auto& out = outputs[0];
  auto x = prepare_rows(inputs[0], out, stream());
  dispatch_norm_type(out.dtype(), "rms_norm", [&](auto t, auto acc) {
    using T = decltype(t);
    using AccT = decltype(acc);
    rms_norm<T, AccT>(x, inputs[1], out, eps_, stream());
  });
}

void LayerNorm::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& out = outputs[0];
  auto x = prepare_rows(inputs[0], out, stream());
  dispatch_norm_type(out.dtype(), "layer_norm", [&](auto t, auto acc) {
    using T = decltype(t);
    using AccT = decltype(acc);
    layer_norm<T, AccT>(x, inputs[1], inputs[2], out, eps_, stream());
  });
}

} // namespace mlx::core::fast
