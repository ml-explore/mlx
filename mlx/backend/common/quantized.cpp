// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename T, int bits>
void extract_bits(const uint8_t* w_in, T* w_out) {
  assert(bits == 3 || bits == 6);
  if (bits == 3) {
    w_out[0] = static_cast<T>(w_in[0] & 0x7);
    w_out[1] = static_cast<T>((w_in[0] & 0x38) >> 3);
    w_out[2] = static_cast<T>(((w_in[0] & 0xc0) >> 6) + ((w_in[1] & 0x1) << 2));
    w_out[3] = static_cast<T>((w_in[1] & 0xe) >> 1);
    w_out[4] = static_cast<T>((w_in[1] & 0x70) >> 4);
    w_out[5] = static_cast<T>(((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0x3) << 1));
    w_out[6] = static_cast<T>((w_in[2] & 0x1c) >> 2);
    w_out[7] = static_cast<T>((w_in[2] & 0xe0) >> 5);
  } else if (bits == 6) {
    w_out[0] = static_cast<T>(w_in[0] & 0x3f);
    w_out[1] =
        static_cast<T>(((w_in[0] >> 6) & 0x03) + ((w_in[1] & 0x0f) << 2));
    w_out[2] =
        static_cast<T>(((w_in[1] >> 4) & 0x0f) + ((w_in[2] & 0x03) << 4));
    w_out[3] = static_cast<T>((w_in[2] >> 2) & 0x3f);
  }
}

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
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t* w_local = (const uint8_t*)w;
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
          if (bits == 3 || bits == 6) {
            T wl[pack_factor];
            extract_bits<T, bits>(w_local, wl);
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              (*result_local++) += xi * (scale * wl[p] + bias);
            }
            w_local += bytes_per_pack;

          } else {
            uint8_t wi = *w_local++;
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              (*result_local++) +=
                  xi * (scale * static_cast<T>(wi & bitmask) + bias);
              if (bits != 8) {
                wi >>= bits;
              }
            }
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
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t* w_local = (const uint8_t*)w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    for (int n = 0; n < N; n++) {
      const T* x_local = x;
      T sum = 0;
      for (int k = 0; k < K; k += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw++) {
          if (bits == 3 || bits == 6) {
            T wl[pack_factor];
            extract_bits<T, bits>(w_local, wl);
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              sum += x_local[p] * (scale * wl[p] + bias);
            }
            w_local += bytes_per_pack;
            x_local += pack_factor;

          } else {
            uint8_t wi = *w_local++;
#pragma clang loop unroll(full)
            for (int p = 0; p < pack_factor; p++) {
              sum +=
                  (*x_local++) * (scale * static_cast<T>(wi & bitmask) + bias);
              if (bits != 8) {
                wi >>= bits;
              }
            }
          }
        }
      }
      *result = sum;
      result++;
    }

    x += K;
  }
}

template <typename T, int bits, int group_size>
void _qmm_dispatch_transpose(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K,
    bool transposed_w) {
  if (transposed_w) {
    return _qmm_t<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
  } else {
    return _qmm<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
  }
}

template <typename T, int bits>
void _qmm_dispatch_group(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K,
    int group_size,
    bool transposed_w) {
  switch (group_size) {
    case 32:
      _qmm_dispatch_transpose<T, bits, 32>(
          result, x, w, scales, biases, M, N, K, transposed_w);
      break;
    case 64:
      _qmm_dispatch_transpose<T, bits, 64>(
          result, x, w, scales, biases, M, N, K, transposed_w);
      break;
    case 128:
      _qmm_dispatch_transpose<T, bits, 128>(
          result, x, w, scales, biases, M, N, K, transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "Quantization group size must be 32, 64 or 128.");
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
    case 2:
      _qmm_dispatch_group<T, 2>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
    case 3:
      _qmm_dispatch_group<T, 3>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
    case 4:
      _qmm_dispatch_group<T, 4>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
    case 6:
      _qmm_dispatch_group<T, 6>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
    case 8:
      _qmm_dispatch_group<T, 8>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
    default:
      throw std::invalid_argument("Quantization bits must be 2, 3, 4, 6 or 8.");
  }
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

void GatherQMM::eval_cpu(const std::vector<array>& inputs, array& out) {
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

template <typename T, typename U>
void quantize(
    const array& w_,
    array& out_,
    array& scales_,
    array& biases_,
    int bits,
    int group_size) {
  const T* w = w_.data<T>();

  auto out = out_.data<U>();
  T* scales = scales_.data<T>();
  T* biases = biases_.data<T>();

  T n_bins = (1 << bits) - 1;
  T eps = 1e-7;
  bool power_of_2_bits = is_power_of_2(bits);
  int el_per_int = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  // For 3/6 bits we read 3 uint8s at a time instead of 1 uint32
  int bytes_per_pack = power_of_2_bits ? 1 : 3;
  int int_per_group = group_size * bytes_per_pack / el_per_int;
  size_t n_groups = w_.size() / group_size;

  for (size_t i = 0; i < n_groups; ++i) {
    size_t w_idx = i * group_size;
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
    size_t out_idx = i * int_per_group;
    for (int j = 0; j < int_per_group / bytes_per_pack; ++j) {
      uint32_t out_el = 0;
      for (int k = 0; k < el_per_int; ++k) {
        T w_el = w[w_idx + j * el_per_int + k];
        w_el = std::rint((w_el - biases[i]) / scales[i]);
        w_el = std::min(std::max(w_el, T(0)), n_bins);
        out_el |= static_cast<uint32_t>(w_el) << (k * bits);
      }
      if (power_of_2_bits) {
        out[out_idx + j] = out_el;
      } else {
        out[out_idx + bytes_per_pack * j] = out_el & 0xff;
        out[out_idx + bytes_per_pack * j + 1] = (out_el & 0xff00) >> 8;
        out[out_idx + bytes_per_pack * j + 2] = (out_el & 0xff0000) >> 16;
      }
    }
  }
}

void fast::AffineQuantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
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

  auto& scales = outputs[1];
  auto& biases = outputs[2];
  scales.set_data(allocator::malloc_or_wait(scales.nbytes()));
  biases.set_data(allocator::malloc_or_wait(biases.nbytes()));
  if (w.dtype() == float16) {
    if (is_power_of_2(bits_)) {
      quantize<float16_t, uint32_t>(w, out, scales, biases, bits_, group_size_);
    } else {
      quantize<float16_t, uint8_t>(w, out, scales, biases, bits_, group_size_);
    }
  } else if (w.dtype() == bfloat16) {
    if (is_power_of_2(bits_)) {
      quantize<bfloat16_t, uint32_t>(
          w, out, scales, biases, bits_, group_size_);
    } else {
      quantize<bfloat16_t, uint8_t>(w, out, scales, biases, bits_, group_size_);
    }
  } else if (w.dtype() == float32) {
    if (is_power_of_2(bits_)) {
      quantize<float, uint32_t>(w, out, scales, biases, bits_, group_size_);
    } else {
      quantize<float, uint8_t>(w, out, scales, biases, bits_, group_size_);
    }
  } else {
    throw std::runtime_error(
        "[fast::AffineQuantize::eval_cpu] Only supports floating point inputs");
  }
}

} // namespace mlx::core
