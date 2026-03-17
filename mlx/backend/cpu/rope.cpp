// Copyright © 2026 Apple Inc.

#include <cassert>
#include <cmath>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

// ISA-specific SIMD implementations for traditional RoPE.
// To add a new ISA: create a new header (e.g. rope_avx512.h) and add
// an #elif before the #else stubs below.
#if defined(__AVX2__)
#include "mlx/backend/cpu/rope_avx2.h"
#else
// No ISA-specific SIMD available. Stub returns 0 so the caller falls
// through to the generic scalar loop.
constexpr bool has_simd_rope = false;
template <typename T, bool forward>
inline int
rope_traditional_simd(const T*, T*, const float*, const float*, int) {
  return 0;
}
#endif // ISA dispatch

namespace {

// Compute sin/cos tables for RoPE positions.
// cos_out, sin_out: [T * half_dims]
void compute_rope_sincos(
    float* cos_out,
    float* sin_out,
    const int* offsets,
    int batch_idx,
    bool per_batch_offset,
    int T,
    int half_dims,
    float base,
    float scale,
    const float* freqs) {
  int offset_val = per_batch_offset ? offsets[batch_idx] : offsets[0];
  float log_base = std::log(base);

  for (int t = 0; t < T; t++) {
    float position = static_cast<float>(t + offset_val) * scale;
    float* c = cos_out + t * half_dims;
    float* s = sin_out + t * half_dims;

    for (int j = 0; j < half_dims; j++) {
      float inv_freq;
      if (freqs) {
        inv_freq = 1.0f / freqs[j];
      } else {
        inv_freq = std::exp(static_cast<float>(-j) * log_base / half_dims);
      }
      float theta = position * inv_freq;
      c[j] = std::cos(theta);
      s[j] = std::sin(theta);
    }
  }
}

// Apply RoPE to a single head for one time step.
// Non-traditional (split halves): x1 = x[0:half], x2 = x[half:dims]
// x_in and x_out point to D contiguous elements.
// Uses SIMD (F16C for float16) -- both halves are contiguous so this
// vectorizes cleanly.
template <typename T, bool forward>
inline void rope_apply_non_traditional(
    const T* x_in,
    T* x_out,
    const float* cos_t,
    const float* sin_t,
    int D,
    int dims,
    int half_dims) {
  using namespace simd;
  constexpr int N = max_size<T>;

  int j = 0;
  for (; j + N <= half_dims; j += N) {
    Simd<float, N> x1(load<T, N>(x_in + j));
    Simd<float, N> x2(load<T, N>(x_in + j + half_dims));
    Simd<float, N> c = load<float, N>(cos_t + j);
    Simd<float, N> s = load<float, N>(sin_t + j);

    if constexpr (forward) {
      store(x_out + j, Simd<T, N>(x1 * c - x2 * s));
      store(x_out + j + half_dims, Simd<T, N>(x1 * s + x2 * c));
    } else {
      store(x_out + j, Simd<T, N>(x1 * c + x2 * s));
      store(x_out + j + half_dims, Simd<T, N>(x2 * c - x1 * s));
    }
  }
  // Scalar tail
  for (; j < half_dims; j++) {
    float x1 = static_cast<float>(x_in[j]);
    float x2 = static_cast<float>(x_in[j + half_dims]);
    float c = cos_t[j];
    float s = sin_t[j];

    if constexpr (forward) {
      x_out[j] = static_cast<T>(x1 * c - x2 * s);
      x_out[j + half_dims] = static_cast<T>(x1 * s + x2 * c);
    } else {
      x_out[j] = static_cast<T>(x1 * c + x2 * s);
      x_out[j + half_dims] = static_cast<T>(x2 * c - x1 * s);
    }
  }
  // Passthrough remaining elements
  for (int k = dims; k < D; k++) {
    x_out[k] = x_in[k];
  }
}

// Traditional (interleaved pairs): x1 = x[0::2], x2 = x[1::2]
// Data layout: [x0, y0, x1, y1, ...] where (xi, yi) form rotation pairs.
// SIMD approach: load 8 interleaved values (4 pairs), swap within pairs,
// duplicate cos/sin to match pair layout, compute rotation in one FMA.
template <typename T, bool forward>
inline void rope_apply_traditional(
    const T* x_in,
    T* x_out,
    const float* cos_t,
    const float* sin_t,
    int D,
    int dims,
    int half_dims) {
  using namespace simd;
  constexpr int N = max_size<T>;

  int j = 0;

  if constexpr (has_simd_rope && N >= 8) {
    j = rope_traditional_simd<T, forward>(x_in, x_out, cos_t, sin_t, half_dims);
  }

  // Scalar tail
  for (; j < half_dims; j++) {
    float x1 = static_cast<float>(x_in[2 * j]);
    float x2 = static_cast<float>(x_in[2 * j + 1]);
    float c = cos_t[j];
    float s = sin_t[j];

    if constexpr (forward) {
      x_out[2 * j] = static_cast<T>(x1 * c - x2 * s);
      x_out[2 * j + 1] = static_cast<T>(x1 * s + x2 * c);
    } else {
      x_out[2 * j] = static_cast<T>(x1 * c + x2 * s);
      x_out[2 * j + 1] = static_cast<T>(x2 * c - x1 * s);
    }
  }
  // Passthrough remaining elements
  for (int k = dims; k < D; k++) {
    x_out[k] = x_in[k];
  }
}

// Process assigned heads for contiguous input [B, N, T, D].
template <typename T, bool traditional, bool forward>
void rope_impl_contiguous(
    const T* in,
    T* out,
    const int* offsets,
    bool per_batch_offset,
    int B,
    int N,
    int T_len,
    int D,
    int dims,
    float base,
    float scale,
    const float* freqs,
    int head_start,
    int head_end) {
  int half_dims = dims / 2;

  int cur_batch = -1;
  std::vector<float> cos_table(T_len * half_dims);
  std::vector<float> sin_table(T_len * half_dims);

  for (int h = head_start; h < head_end; h++) {
    int b = h / N;
    int n = h % N;

    if (b != cur_batch) {
      cur_batch = b;
      compute_rope_sincos(
          cos_table.data(),
          sin_table.data(),
          offsets,
          b,
          per_batch_offset,
          T_len,
          half_dims,
          base,
          scale,
          freqs);
    }

    for (int t = 0; t < T_len; t++) {
      size_t idx = (static_cast<size_t>(b) * N * T_len + n * T_len + t) * D;
      const float* c = cos_table.data() + t * half_dims;
      const float* s = sin_table.data() + t * half_dims;

      if constexpr (traditional) {
        rope_apply_traditional<T, forward>(
            in + idx, out + idx, c, s, D, dims, half_dims);
      } else {
        rope_apply_non_traditional<T, forward>(
            in + idx, out + idx, c, s, D, dims, half_dims);
      }
    }
  }
}

// Process assigned heads for strided input.
// Input is accessed via strides; output is written contiguously.
// in_b_stride: stride between batches in the input
// in_n_stride: stride between heads in the input
// in_t_stride: stride between time steps in the input
// The D-dimension must have stride 1 in the input.
template <typename T, bool traditional, bool forward>
void rope_impl_strided(
    const T* in,
    T* out,
    const int* offsets,
    bool per_batch_offset,
    int B,
    int N,
    int T_len,
    int D,
    int dims,
    float base,
    float scale,
    const float* freqs,
    int64_t in_b_stride,
    int64_t in_n_stride,
    int64_t in_t_stride,
    int head_start,
    int head_end) {
  int half_dims = dims / 2;

  int cur_batch = -1;
  std::vector<float> cos_table(T_len * half_dims);
  std::vector<float> sin_table(T_len * half_dims);

  for (int h = head_start; h < head_end; h++) {
    int b = h / N;
    int n = h % N;

    if (b != cur_batch) {
      cur_batch = b;
      compute_rope_sincos(
          cos_table.data(),
          sin_table.data(),
          offsets,
          b,
          per_batch_offset,
          T_len,
          half_dims,
          base,
          scale,
          freqs);
    }

    for (int t = 0; t < T_len; t++) {
      // Input uses strides
      int64_t in_idx = b * in_b_stride + n * in_n_stride + t * in_t_stride;
      // Output is always row-contiguous
      size_t out_idx = (static_cast<size_t>(b) * N * T_len + n * T_len + t) * D;
      const float* c = cos_table.data() + t * half_dims;
      const float* s = sin_table.data() + t * half_dims;

      if constexpr (traditional) {
        rope_apply_traditional<T, forward>(
            in + in_idx, out + out_idx, c, s, D, dims, half_dims);
      } else {
        rope_apply_non_traditional<T, forward>(
            in + in_idx, out + out_idx, c, s, D, dims, half_dims);
      }
    }
  }
}

template <typename T>
void rope_dispatch(
    const T* in,
    T* out,
    const int* offsets,
    bool per_batch_offset,
    int B,
    int N,
    int T_len,
    int D,
    int dims,
    float base,
    float scale,
    const float* freqs,
    bool traditional,
    bool forward,
    bool use_strides,
    int64_t in_b_stride,
    int64_t in_n_stride,
    int64_t in_t_stride,
    cpu::ThreadPool& pool,
    int nth) {
  int total_heads = B * N;

  auto run = [&](auto trad_tag, auto fwd_tag) {
    auto do_work = [&](int start, int end) {
      if (use_strides) {
        rope_impl_strided<
            T,
            decltype(trad_tag)::value,
            decltype(fwd_tag)::value>(
            in,
            out,
            offsets,
            per_batch_offset,
            B,
            N,
            T_len,
            D,
            dims,
            base,
            scale,
            freqs,
            in_b_stride,
            in_n_stride,
            in_t_stride,
            start,
            end);
      } else {
        rope_impl_contiguous<
            T,
            decltype(trad_tag)::value,
            decltype(fwd_tag)::value>(
            in,
            out,
            offsets,
            per_batch_offset,
            B,
            N,
            T_len,
            D,
            dims,
            base,
            scale,
            freqs,
            start,
            end);
      }
    };

    if (nth > 1) {
      pool.parallel_for(nth, [&](int tid, int num_threads) {
        int per = (total_heads + num_threads - 1) / num_threads;
        int start = per * tid;
        int end = std::min(start + per, total_heads);
        do_work(start, end);
      });
    } else {
      do_work(0, total_heads);
    }
  };

  if (traditional) {
    if (forward) {
      run(std::true_type{}, std::true_type{});
    } else {
      run(std::true_type{}, std::false_type{});
    }
  } else {
    if (forward) {
      run(std::false_type{}, std::true_type{});
    } else {
      run(std::false_type{}, std::false_type{});
    }
  }
}

} // namespace

void RoPE::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  int ndim = in.ndim();
  int D = in.shape(ndim - 1);

  // Compute 4D dimensions: [B, N, T, D]
  int B = in.shape(0);
  int N = 1;
  for (int i = 1; i < ndim - 2; i++) {
    N *= in.shape(i);
  }
  int T = in.shape(ndim - 2);

  // Determine input layout
  bool row_contiguous = in.flags().row_contiguous;

  // For non-contiguous input, check if D-dimension has stride 1
  // (common case: transpose of (B,T,N,D) -> (B,N,T,D))
  bool d_stride_one = row_contiguous || (in.strides()[ndim - 1] == 1);

  // Compute strides for the B, N, T dimensions
  // For contiguous: b_stride = N*T*D, n_stride computed from middle dims,
  //                 t_stride = D
  // For non-contiguous: use actual strides
  int64_t in_b_stride = 0;
  int64_t in_n_stride = 0;
  int64_t in_t_stride = 0;
  bool use_strides = false;

  if (row_contiguous) {
    // Standard contiguous layout
    use_strides = false;
  } else if (d_stride_one && ndim >= 3) {
    // Non-contiguous but D is contiguous -- use strided access
    use_strides = true;
    in_b_stride = in.strides()[0];
    in_t_stride = in.strides()[ndim - 2];
    // For N (product of middle dims), use the stride of the first middle dim
    // This works for 4D (B, N, T, D) but for higher dims we need the stride
    // of the flattened N dimension. If the middle dims aren't contiguous with
    // each other, fall back to copy.
    if (ndim == 3) {
      in_n_stride = 0; // N=1, stride doesn't matter
    } else if (ndim == 4) {
      in_n_stride = in.strides()[1];
    } else {
      // For ndim > 4, check if middle dimensions are regularly strided
      // For simplicity, fall back to copy for >4D non-contiguous
      use_strides = false;
    }
  }

  // If we can't use strides, make a contiguous copy
  array x = in;
  bool made_copy = false;
  if (!row_contiguous && !use_strides) {
    x = contiguous_copy_cpu(in, stream());
    made_copy = true;
  }

  // Allocate output as row-contiguous
  out.set_data(allocator::malloc(out.nbytes()));

  // Get offset
  auto& offset_arr = inputs[1];
  bool per_batch_offset = offset_arr.size() > 1;
  bool has_freqs = inputs.size() == 3;

  // Set up command encoder
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(x);
  encoder.set_input_array(offset_arr);
  if (has_freqs) {
    encoder.set_input_array(inputs[2]);
  }
  encoder.set_output_array(out);
  if (made_copy) {
    encoder.add_temporary(x);
  }

  // Determine threading
  auto& pool = cpu::ThreadPool::instance();
  int total_heads = B * N;
  int nth = std::min(pool.max_threads(), total_heads);
  size_t total_elements = static_cast<size_t>(B) * N * T * D;
  if (total_elements < cpu::MIN_TOTAL_ELEMENTS) {
    nth = 1;
  }

  int dims = dims_;
  float base = base_;
  float scale = scale_;
  bool traditional = traditional_;
  bool forward = forward_;

  auto dispatch_typed = [&](auto dummy) {
    using DType = decltype(dummy);
    const DType* x_ptr = x.data<DType>();
    const int* off_ptr = offset_arr.data<int>();
    const float* freq_ptr = has_freqs ? inputs[2].data<float>() : nullptr;
    DType* out_ptr = out.data<DType>();

    encoder.dispatch([=, &pool]() {
      rope_dispatch<DType>(
          x_ptr,
          out_ptr,
          off_ptr,
          per_batch_offset,
          B,
          N,
          T,
          D,
          dims,
          base,
          scale,
          freq_ptr,
          traditional,
          forward,
          use_strides,
          in_b_stride,
          in_n_stride,
          in_t_stride,
          pool,
          nth);
    });
  };

  switch (x.dtype()) {
    case float32:
      dispatch_typed(float{});
      break;
    case float16:
      dispatch_typed(float16_t{});
      break;
    case bfloat16:
      dispatch_typed(bfloat16_t{});
      break;
    default:
      throw std::runtime_error("[RoPE::eval_cpu] Unsupported type");
  }
}

} // namespace mlx::core::fast
