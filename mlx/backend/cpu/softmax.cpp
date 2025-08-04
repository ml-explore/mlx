// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"
#include "mlx/types/limits.h"

namespace mlx::core {

namespace {

using namespace mlx::core::simd;

template <typename T, typename AccT>
void softmax(const array& in, array& out, Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();

  int M = in.shape().back();
  int L = in.data_size() / M;

  encoder.dispatch([in_ptr, out_ptr, M, L]() mutable {
    constexpr bool same_t = std::is_same_v<T, AccT>;
    constexpr int N = std::min(max_size<AccT>, max_size<T>);

    const T* current_in_ptr;
    T* current_out_ptr;

    for (int i = 0; i < L; i++, in_ptr += M, out_ptr += M) {
      // Find the maximum
      current_in_ptr = in_ptr;
      Simd<AccT, N> vmaximum(-numeric_limits<AccT>::infinity());
      size_t s = M;
      while (s >= N) {
        Simd<AccT, N> vals = load<T, N>(current_in_ptr);
        vmaximum = maximum(vals, vmaximum);
        current_in_ptr += N;
        s -= N;
      }

      AccT maximum = max(vmaximum);
      while (s-- > 0) {
        maximum = std::max(maximum, static_cast<AccT>(*current_in_ptr));
        current_in_ptr++;
      }

      // Compute the normalizer and the exponentials
      Simd<AccT, N> vnormalizer(0.0);
      current_out_ptr = out_ptr;
      current_in_ptr = in_ptr;
      s = M;
      while (s >= N) {
        Simd<AccT, N> vexp = load<T, N>(current_in_ptr);
        vexp = exp(vexp - maximum);
        if constexpr (same_t) {
          store(current_out_ptr, vexp);
        }
        vnormalizer = vnormalizer + vexp;
        current_in_ptr += N;
        current_out_ptr += N;
        s -= N;
      }
      AccT normalizer = sum(vnormalizer);
      while (s-- > 0) {
        AccT _exp = std::exp(*current_in_ptr - maximum);
        if constexpr (same_t) {
          *current_out_ptr = _exp;
        }
        normalizer += _exp;
        current_in_ptr++;
        current_out_ptr++;
      }
      normalizer = 1 / normalizer;

      // Normalize
      current_out_ptr = out_ptr;
      current_in_ptr = in_ptr;
      s = M;
      while (s >= N) {
        if constexpr (same_t) {
          store(
              current_out_ptr,
              Simd<T, N>(load<T, N>(current_out_ptr) * normalizer));
        } else {
          Simd<AccT, N> vexp = load<T, N>(current_in_ptr);
          vexp = exp(vexp - maximum) * normalizer;
          store(current_out_ptr, Simd<T, N>(vexp));
          current_in_ptr += N;
        }
        current_out_ptr += N;
        s -= N;
      }
      while (s-- > 0) {
        if constexpr (same_t) {
          *current_out_ptr *= normalizer;
        } else {
          AccT _exp = std::exp(*current_in_ptr - maximum);
          *current_out_ptr = static_cast<T>(_exp * normalizer);
          current_in_ptr++;
        }
        current_out_ptr++;
      }
    }
  });
}

} // namespace

void Softmax::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // Make sure that the last dimension is contiguous
  auto set_output = [s = stream(), &out](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_cpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  auto in = set_output(inputs[0]);

  switch (in.dtype()) {
    case float32:
      softmax<float, float>(in, out, stream());
      break;
    case float16:
      if (precise_) {
        softmax<float16_t, float>(in, out, stream());
      } else {
        softmax<float16_t, float16_t>(in, out, stream());
      }
      break;
    case bfloat16:
      if (precise_) {
        softmax<bfloat16_t, float>(in, out, stream());
      } else {
        softmax<bfloat16_t, bfloat16_t>(in, out, stream());
      }
      break;
    case float64:
      softmax<double, double>(in, out, stream());
      break;
    default:
      throw std::runtime_error(
          "[softmax] Only defined for floating point types.");
      break;
  }
}

} // namespace mlx::core
