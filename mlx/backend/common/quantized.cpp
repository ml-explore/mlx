// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/ops.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, int bits, int group_size>
void _qmm(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const int Ng = N / group_size;
  const int Nw = N / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    std::fill(result, result + N, 0);

    for (int k = 0; k < K; k++) {
      T* result_local = result;
      T xi = *x++;

      for (int n = 0; n < N; n += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;
        for (int ng = 0; ng < packs_in_group; ng++) {
          uint32_t wi = *w_local++;

#pragma clang loop unroll(full)
          for (int p = 0; p < pack_factor; p++) {
            (*result_local++) +=
                xi * (scale * static_cast<T>(wi & bitmask) + bias);
            wi >>= bits;
          }
        }
      }
    }

    result += N;
  }
}

template <typename T, int bits, int group_size>
void _qmm_t(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const int Kg = K / group_size;
  const int Kw = K / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    for (int n = 0; n < N; n++) {
      const T* x_local = x;
      T sum = 0;
      for (int k = 0; k < K; k += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw++) {
          uint32_t wi = *w_local++;

#pragma clang loop unroll(full)
          for (int p = 0; p < pack_factor; p++) {
            sum += (*x_local++) * (scale * static_cast<T>(wi & bitmask) + bias);
            wi >>= bits;
          }
        }
      }
      *result = sum;
      result++;
    }

    x += K;
  }
}

template <typename T>
void _qmm_dispatch_typed(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K,
    int group_size,
    int bits,
    bool transposed_w) {
  switch (bits) {
    case 2: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 2, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 2, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 2, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
    case 4: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 4, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 4, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 4, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
    case 8: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 8, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 8, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 8, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
  }
  std::ostringstream msg;
  msg << "Quantization type not supported. Provided bits=" << bits
      << " and group_size=" << group_size
      << ". The supported options are bits in "
      << "{2, 4, 8} and group_size in {64, 128}.";
  throw std::invalid_argument(msg.str());
}

void _qmm_dispatch(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int bits,
    int group_size,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);

  int w_els = w.ndim() > 2 ? w.shape(-1) * w.shape(-2) : 0;
  int g_els = w.ndim() > 2 ? scales.shape(-1) * scales.shape(-2) : 0;

  int batch_size = x.size() / x.shape(-1) / x.shape(-2);
  for (int i = 0; i < batch_size; i++) {
    switch (x.dtype()) {
      case float32:
        _qmm_dispatch_typed<float>(
            out.data<float>() + i * M * N,
            x.data<float>() + elem_to_loc(i * M * K, x),
            w.data<uint32_t>() + elem_to_loc(i * w_els, w),
            scales.data<float>() + elem_to_loc(i * g_els, scales),
            biases.data<float>() + elem_to_loc(i * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      case float16:
        _qmm_dispatch_typed<float16_t>(
            out.data<float16_t>() + i * M * N,
            x.data<float16_t>() + elem_to_loc(i * M * K, x),
            w.data<uint32_t>() + elem_to_loc(i * w_els, w),
            scales.data<float16_t>() + elem_to_loc(i * g_els, scales),
            biases.data<float16_t>() + elem_to_loc(i * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      case bfloat16:
        _qmm_dispatch_typed<bfloat16_t>(
            out.data<bfloat16_t>() + i * M * N,
            x.data<bfloat16_t>() + elem_to_loc(i * M * K, x),
            w.data<uint32_t>() + elem_to_loc(i * w_els, w),
            scales.data<bfloat16_t>() + elem_to_loc(i * g_els, scales),
            biases.data<bfloat16_t>() + elem_to_loc(i * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      default:
        throw std::invalid_argument(
            "[quantized_matmul] only floating types are supported");
    }
  }
}

void _bs_qmm_dispatch(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    int bits,
    int group_size,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);

  int w_els = w.shape(-1) * w.shape(-2);
  int g_els = scales.shape(-1) * scales.shape(-2);

  const uint32_t* lhs_indices_data = lhs_indices.data<uint32_t>();
  const uint32_t* rhs_indices_data = rhs_indices.data<uint32_t>();

  for (int i = 0; i < lhs_indices.size(); i++) {
    int x_idx = lhs_indices_data[elem_to_loc(i, lhs_indices)];
    int w_idx = rhs_indices_data[elem_to_loc(i, rhs_indices)];

    switch (x.dtype()) {
      case float32:
        _qmm_dispatch_typed<float>(
            out.data<float>() + i * M * N,
            x.data<float>() + elem_to_loc(x_idx * M * K, x),
            w.data<uint32_t>() + elem_to_loc(w_idx * w_els, w),
            scales.data<float>() + elem_to_loc(w_idx * g_els, scales),
            biases.data<float>() + elem_to_loc(w_idx * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      case float16:
        _qmm_dispatch_typed<float16_t>(
            out.data<float16_t>() + i * M * N,
            x.data<float16_t>() + elem_to_loc(x_idx * M * K, x),
            w.data<uint32_t>() + elem_to_loc(w_idx * w_els, w),
            scales.data<float16_t>() + elem_to_loc(w_idx * g_els, scales),
            biases.data<float16_t>() + elem_to_loc(w_idx * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      case bfloat16:
        _qmm_dispatch_typed<bfloat16_t>(
            out.data<bfloat16_t>() + i * M * N,
            x.data<bfloat16_t>() + elem_to_loc(x_idx * M * K, x),
            w.data<uint32_t>() + elem_to_loc(w_idx * w_els, w),
            scales.data<bfloat16_t>() + elem_to_loc(w_idx * g_els, scales),
            biases.data<bfloat16_t>() + elem_to_loc(w_idx * g_els, biases),
            M,
            N,
            K,
            bits,
            group_size,
            transposed_w);
        break;
      default:
        throw std::invalid_argument(
            "[quantized_matmul] only floating types are supported");
    }
  }
}

} // namespace

void QuantizedMatmul::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

  auto ensure_row_contiguous = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      return arr_copy;
    }
  };

  auto x = ensure_row_contiguous(x_pre);
  auto w = ensure_row_contiguous(w_pre);
  auto scales = ensure_row_contiguous(scales_pre);
  auto biases = ensure_row_contiguous(biases_pre);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  _qmm_dispatch(out, x, w, scales, biases, group_size_, bits_, transpose_);
}

void GatherQMM::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 6);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];
  auto& lhs_indices = inputs[4];
  auto& rhs_indices = inputs[5];

  auto ensure_row_contiguous_last_dims = [](const array& arr) {
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      return arr_copy;
    }
  };

  auto x = ensure_row_contiguous_last_dims(x_pre);
  auto w = ensure_row_contiguous_last_dims(w_pre);
  auto scales = ensure_row_contiguous_last_dims(scales_pre);
  auto biases = ensure_row_contiguous_last_dims(biases_pre);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  _bs_qmm_dispatch(
      out,
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      group_size_,
      bits_,
      transpose_);
}

template <typename T>
void quantize(
    const array& w_,
    array& out_,
    array& scales_,
    array& biases_,
    int bits,
    int group_size,
    bool compute_scale_bias) {
  const T* w = w_.data<T>();
  T* scales = scales_.data<T>();
  T* biases = biases_.data<T>();
  auto out = out_.data<uint32_t>();

  T n_bins = (1 << bits) - 1;
  T eps = 1e-7;
  int el_per_int = 32 / bits;
  int int_per_group = group_size / el_per_int;
  size_t n_groups = w_.size() / group_size;

  for (size_t i = 0; i < n_groups; ++i) {
    size_t w_idx = i * group_size;
    if (compute_scale_bias) {
      T w_min = std::numeric_limits<float>::infinity();
      T w_max = -w_min;
      for (int j = 0; j < group_size; ++j) {
        w_max = std::max(w_max, w[w_idx + j]);
        w_min = std::min(w_min, w[w_idx + j]);
      }
      bool mask = std::abs(w_min) > std::abs(w_max);
      T scale = std::max(T((w_max - w_min) / n_bins), eps);
      scale = mask ? scale : -scale;

      auto edge = mask ? w_min : w_max;
      auto q0 = std::rint(edge / scale);
      if (q0 == 0) {
        scales[i] = scale;
        biases[i] = 0;
      } else {
        scales[i] = edge / q0;
        biases[i] = edge;
      }
    }
    size_t out_idx = i * int_per_group;
    for (int j = 0; j < int_per_group; ++j) {
      uint32_t out_el = 0;
      for (int k = 0; k < el_per_int; ++k) {
        T w_el = w[w_idx + j * el_per_int + k];
        w_el = std::rint((w_el - biases[i]) / scales[i]);
        w_el = std::min(std::max(w_el, T(0)), n_bins);
        out_el |= static_cast<uint32_t>(w_el) << (k * bits);
      }
      out[out_idx + j] = out_el;
    }
  }
}

void fast::AffineQuantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  bool compute_scale_bias = inputs.size() == 1;

  auto ensure_row_contiguous = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      return arr_copy;
    }
  };
  auto w = ensure_row_contiguous(inputs[0]);

  auto& out = outputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& scales =
      compute_scale_bias ? outputs[1] : const_cast<array&>(inputs[1]);
  auto& biases =
      compute_scale_bias ? outputs[2] : const_cast<array&>(inputs[2]);
  if (compute_scale_bias) {
    scales.set_data(allocator::malloc_or_wait(scales.nbytes()));
    biases.set_data(allocator::malloc_or_wait(biases.nbytes()));
  }
  if (w.dtype() == float16) {
    quantize<float16_t>(
        w, out, scales, biases, bits_, group_size_, compute_scale_bias);
  } else if (w.dtype() == bfloat16) {
    quantize<bfloat16_t>(
        w, out, scales, biases, bits_, group_size_, compute_scale_bias);
  } else if (w.dtype() == float32) {
    quantize<float>(
        w, out, scales, biases, bits_, group_size_, compute_scale_bias);
  } else {
    throw std::runtime_error(
        "[fast::AffineQuantize::eval_cpu] Only supports floating point inputs");
  }
}

} // namespace mlx::core
