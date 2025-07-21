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
void logsumexp(const array& in, array& out, Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  const T* in_ptr = in.data<T>();
  T* out_ptr = out.data<T>();

  int M = in.shape().back();
  int L = in.data_size() / M;

  encoder.dispatch([in_ptr, out_ptr, M, L]() mutable {
    constexpr int N = std::min(max_size<AccT>, max_size<T>);

    const T* current_in_ptr;

    for (int i = 0; i < L; i++, in_ptr += M, out_ptr += 1) {
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
      current_in_ptr = in_ptr;
      s = M;
      while (s >= N) {
        Simd<AccT, N> vexp = load<T, N>(current_in_ptr);
        vexp = exp(vexp - maximum);
        vnormalizer = vnormalizer + vexp;
        current_in_ptr += N;
        s -= N;
      }
      AccT normalizer = sum(vnormalizer);
      while (s-- > 0) {
        AccT _exp = std::exp(*current_in_ptr - maximum);
        normalizer += _exp;
        current_in_ptr++;
      }
      // Normalize
      *out_ptr = std::isinf(maximum)
          ? static_cast<T>(maximum)
          : static_cast<T>(std::log(normalizer) + maximum);
    }
  });
}

} // namespace

void LogSumExp::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // Make sure that the last dimension is contiguous
  auto s = stream();
  auto& encoder = cpu::get_command_encoder(s);
  auto ensure_contiguous = [&s, &encoder](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      array x_copy = contiguous_copy_cpu(x, s);
      encoder.add_temporary(x_copy);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);
  if (in.flags().row_contiguous) {
    out.set_data(allocator::malloc(out.nbytes()));
  } else {
    auto n = in.shape(-1);
    auto flags = in.flags();
    auto strides = in.strides();
    for (auto& s : strides) {
      s /= n;
    }
    bool col_contig = strides[0] == 1;
    for (int i = 1; col_contig && i < strides.size(); ++i) {
      col_contig &=
          (out.shape(i) == 1 || strides[i - 1] == out.shape(i) * strides[i]);
    }
    flags.col_contiguous = col_contig;
    out.set_data(
        allocator::malloc(in.nbytes() / n),
        in.data_size() / n,
        std::move(strides),
        flags);
  }

  switch (in.dtype()) {
    case float32:
      logsumexp<float, float>(in, out, stream());
      break;
    case float16:
      logsumexp<float16_t, float>(in, out, stream());
      break;
    case bfloat16:
      logsumexp<bfloat16_t, float>(in, out, stream());
      break;
    case float64:
      logsumexp<double, double>(in, out, stream());
      break;
    default:
      throw std::runtime_error(
          "[logsumexp] only supports floating point types");
      break;
  }
}

} // namespace mlx::core
