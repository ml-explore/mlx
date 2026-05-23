// Copyright © 2023 Apple Inc.

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/unary.h"
#include "mlx/backend/cpu/unary_ops.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

array ensure_row_contiguous(
    const array& arr,
    cpu::CommandEncoder& encoder,
    Stream s) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    auto arr_cpy = contiguous_copy_cpu(arr, s);
    encoder.add_temporary(arr_cpy);
    return arr_cpy;
  }
};

const static float FP4_LUT[16] = {
    +0.0f,
    +0.5f,
    +1.0f,
    +1.5f,
    +2.0f,
    +3.0f,
    +4.0f,
    +6.0f,
    -0.0f,
    -0.5f,
    -1.0f,
    -1.5f,
    -2.0f,
    -3.0f,
    -4.0f,
    -6.0f};

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return static_cast<T>(detail::FromFP8{}(s));
  } else {
    using FOrI = union {
      bfloat16_t f;
      uint16_t i;
    };
    FOrI out;
    out.i = (s == 0 ? 0x40 : (static_cast<uint16_t>(s) << 7));
    return static_cast<T>(out.f);
  }
}

template <typename T, int bits>
void extract_bits(const uint8_t* w_in, T* w_out) {
  static_assert(bits == 3 || bits == 5 || bits == 6);
  if (bits == 3) {
    w_out[0] = static_cast<T>(w_in[0] & 0x7);
    w_out[1] = static_cast<T>((w_in[0] & 0x38) >> 3);
    w_out[2] = static_cast<T>(((w_in[0] & 0xc0) >> 6) + ((w_in[1] & 0x1) << 2));
    w_out[3] = static_cast<T>((w_in[1] & 0xe) >> 1);
    w_out[4] = static_cast<T>((w_in[1] & 0x70) >> 4);
    w_out[5] = static_cast<T>(((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0x3) << 1));
    w_out[6] = static_cast<T>((w_in[2] & 0x1c) >> 2);
    w_out[7] = static_cast<T>((w_in[2] & 0xe0) >> 5);
  } else if (bits == 5) {
    w_out[0] = static_cast<T>(w_in[0] & 0x1f);
    w_out[1] = static_cast<T>(((w_in[0] & 0xe0) >> 5) + ((w_in[1] & 0x3) << 3));
    w_out[2] = static_cast<T>((w_in[1] & 0x7c) >> 2);
    w_out[3] = static_cast<T>(((w_in[1] & 0x80) >> 7) + ((w_in[2] & 0xf) << 1));
    w_out[4] = static_cast<T>(((w_in[2] & 0xf0) >> 4) + ((w_in[3] & 0x1) << 4));
    w_out[5] = static_cast<T>((w_in[3] & 0x3e) >> 1);
    w_out[6] = static_cast<T>(((w_in[3] & 0xc0) >> 6) + ((w_in[4] & 0x7) << 2));
    w_out[7] = static_cast<T>((w_in[4] & 0xf8) >> 3);

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
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
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
          if constexpr (bits == 3 || bits == 5 || bits == 6) {
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

  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
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
          if constexpr (bits == 3 || bits == 5 || bits == 6) {
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

template <int bits, int S>
simd::Simd<uint32_t, S> extract_bits_simd(const uint32_t* w) {
  constexpr int bitmask = (1 << bits) - 1;
  simd::Simd<uint32_t, S> wi;
  if constexpr (bits == 4 && S == 8) {
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 4, 8, 12, 16, 20, 24, 28}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    wi = simd::Simd<uint32_t, S>(*w);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 8 && S == 8) {
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 8, 16, 24, 0, 8, 16, 24}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    auto l = simd::Simd<uint32_t, S / 2>(*w++);
    auto r = simd::Simd<uint32_t, S / 2>(*w);
    wi = simd::Simd<uint32_t, S>(l, r);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else {
    // Appease compiler.. but should never get here
    throw std::runtime_error("Unsupported combination for simd qmm.");
  }
  return wi;
}

template <typename T, int bits, int group_size>
void _qmm_t_simd(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  constexpr int S = simd::max_size<T>;
  static_assert(
      S % pack_factor == 0, "SIMD size must be divisible by pack factor");
  constexpr int packs_per_simd = S / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    for (int n = 0; n < N; n++) {
      simd::Simd<float, S> acc(0);
      auto x_local = x;
      for (int k = 0; k < K; k += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw += packs_per_simd) {
          auto wf = simd::Simd<float, S>(extract_bits_simd<bits, S>(w_local));
          w_local += packs_per_simd;
          wf = wf * scale;
          wf = wf + bias;
          simd::Simd<float, S> x_simd = simd::load<T, S>(x_local);
          acc = acc + x_simd * wf;
          x_local += S;
        }
      }

      *result = T(simd::sum(acc));
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
    // the simd size must be a multiple of the number of elements per word
    if constexpr (32 % bits == 0 && simd::max_size<T> % (32 / bits) == 0) {
      _qmm_t_simd<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
    } else {
      _qmm_t<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
    }
  } else {
    _qmm<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
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
    case 5:
      _qmm_dispatch_group<T, 5>(
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

template <typename T>
void _qmm_dispatch_typed(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int bits,
    int group_size,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.ndim() > 1 ? x.shape(-2) : 1;
  int N = out.shape(-1);
  int w_els = w.ndim() > 2 ? w.shape(-1) * w.shape(-2) : 0;
  int g_els = w.ndim() > 2 ? scales.shape(-1) * scales.shape(-2) : 0;
  int batch_size = x.size() / (K * M);

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<T>();
  auto biases_ptr = biases.data<T>();
  for (int i = 0; i < batch_size; i++) {
    _qmm_dispatch_typed<T>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(i * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(i * w_els, w.shape(), w.strides()),
        scales_ptr + elem_to_loc(i * g_els, scales.shape(), scales.strides()),
        biases_ptr + elem_to_loc(i * g_els, biases.shape(), biases.strides()),
        M,
        N,
        K,
        bits,
        group_size,
        transposed_w);
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
  switch (x.dtype()) {
    case float32:
      _qmm_dispatch_typed<float>(
          out, x, w, scales, biases, bits, group_size, transposed_w);
      break;
    case float16:
      _qmm_dispatch_typed<float16_t>(
          out, x, w, scales, biases, bits, group_size, transposed_w);
      break;
    case bfloat16:
      _qmm_dispatch_typed<bfloat16_t>(
          out, x, w, scales, biases, bits, group_size, transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

template <typename T, int group_size, int bits>
void fp_qmm(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K) {
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t* w_local = (const uint8_t*)w;
    const uint8_t* scales_local = scales;

    std::fill(result, result + N, 0);

    for (int k = 0; k < K; k++) {
      T* result_local = result;
      T xi = *x++;

      for (int n = 0; n < N; n += group_size) {
        T scale = dequantize_scale<T, group_size>(*scales_local++);
        for (int ng = 0; ng < packs_in_group; ng++) {
          if constexpr (bits == 4) {
            (*result_local++) +=
                xi * scale * static_cast<T>(FP4_LUT[w_local[0] & 0xf]);
            (*result_local++) +=
                xi * scale * static_cast<T>(FP4_LUT[(w_local[0] >> 4) & 0xf]);
          } else {
            (*result_local++) +=
                xi * scale * static_cast<T>(detail::FromFP8{}(w_local[0]));
          }
          w_local++;
        }
      }
    }
    result += N;
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_t(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K) {
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int packs_in_group = group_size / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint8_t* w_local = (const uint8_t*)w;
    const uint8_t* scales_local = scales;

    for (int n = 0; n < N; n++) {
      const T* x_local = x;
      T sum = 0;
      for (int k = 0; k < K; k += group_size) {
        T scale = dequantize_scale<T, group_size>(*scales_local++);

        T gsum = 0;
        for (int kw = 0; kw < packs_in_group; kw++) {
          if constexpr (bits == 4) {
            gsum += (*x_local++) * static_cast<T>(FP4_LUT[w_local[0] & 0xf]);
            gsum +=
                (*x_local++) * static_cast<T>(FP4_LUT[(w_local[0] >> 4) & 0xf]);
          } else {
            gsum +=
                (*x_local++) * static_cast<T>(detail::FromFP8{}(w_local[0]));
          }
          w_local++;
        }
        sum += scale * gsum;
      }
      *result = sum;
      result++;
    }

    x += K;
  }
}

template <int S, int bits>
simd::Simd<float, S> fp_extract_bits_simd(const uint32_t* w) {
  if constexpr (S == 8 && bits == 4) {
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 4, 8, 12, 16, 20, 24, 28}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    auto wi = simd::Simd<uint32_t, S>(*w);
    wi = wi >> shifts;
    wi = wi & 0xf;
    simd::Simd<float, S> w_out;
    for (int i = 0; i < S; ++i) {
      w_out[i] = FP4_LUT[wi[i]];
    }
    return w_out;
  } else if constexpr (S == 8 && bits == 8) {
    auto w_out = simd::load<uint8_t, S>(reinterpret_cast<const uint8_t*>(w));
    return detail::FromFP8{}(w_out);
  } else {
    // Appease compiler.. but should never get here
    throw std::runtime_error("Unsupported combination for simd qmm.");
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_t_simd(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K) {
  constexpr int pack_factor = get_pack_factor(bits, 32);
  constexpr int packs_in_group = group_size / pack_factor;
  constexpr int S = simd::max_size<T>;
  static_assert(
      S % pack_factor == 0, "SIMD size must be divisible by pack factor");
  constexpr int packs_per_simd = S / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const uint8_t* scales_local = scales;

    for (int n = 0; n < N; n++) {
      simd::Simd<float, S> acc(0);
      auto x_local = x;
      for (int k = 0; k < K; k += group_size) {
        T scale = dequantize_scale<T, group_size>(*scales_local++);

        simd::Simd<float, S> g_acc(0);
        for (int kw = 0; kw < packs_in_group; kw += packs_per_simd) {
          // Extract bits
          auto wf = fp_extract_bits_simd<S, bits>(w_local);
          w_local += packs_per_simd;
          simd::Simd<float, S> x_simd = simd::load<T, S>(x_local);
          g_acc = g_acc + x_simd * wf;
          x_local += S;
        }
        acc = acc + scale * g_acc;
      }

      *result = T(simd::sum(acc));
      result++;
    }
    x += K;
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_dispatch_transpose(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K,
    bool transposed_w) {
  if (transposed_w) {
    // the simd size must be a multiple of the number of elements per word
    if constexpr (simd::max_size<T> % 8 == 0) {
      fp_qmm_t_simd<T, group_size, bits>(result, x, w, scales, M, N, K);
    } else {
      fp_qmm_t<T, group_size, bits>(result, x, w, scales, M, N, K);
    }
  } else {
    fp_qmm<T, group_size, bits>(result, x, w, scales, M, N, K);
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_dispatch_mode(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.ndim() > 1 ? x.shape(-2) : 1;
  int N = out.shape(-1);
  int w_els = w.ndim() > 2 ? w.shape(-1) * w.shape(-2) : 0;
  int g_els = w.ndim() > 2 ? scales.shape(-1) * scales.shape(-2) : 0;
  int batch_size = x.size() / (K * M);

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<uint8_t>();
  for (int i = 0; i < batch_size; i++) {
    fp_qmm_dispatch_transpose<T, group_size, bits>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(i * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(i * w_els, w.shape(), w.strides()),
        scales_ptr + elem_to_loc(i * g_els, scales.shape(), scales.strides()),
        M,
        N,
        K,
        transposed_w);
  }
}

template <typename T>
void fp_qmm_dispatch_typed(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    int group_size,
    int bits,
    bool transposed_w) {
  if (bits == 8) {
    fp_qmm_dispatch_mode<T, 32, 8>(out, x, w, scales, transposed_w);
  } else if (group_size == 32) {
    fp_qmm_dispatch_mode<T, 32, 4>(out, x, w, scales, transposed_w);
  } else {
    fp_qmm_dispatch_mode<T, 16, 4>(out, x, w, scales, transposed_w);
  }
}

void fp_qmm_dispatch(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    int group_size,
    int bits,
    bool transposed_w) {
  switch (x.dtype()) {
    case bfloat16:
      fp_qmm_dispatch_typed<bfloat16_t>(
          out, x, w, scales, group_size, bits, transposed_w);
      break;
    case float16:
      fp_qmm_dispatch_typed<float16_t>(
          out, x, w, scales, group_size, bits, transposed_w);
      break;
    case float32:
      fp_qmm_dispatch_typed<float>(
          out, x, w, scales, group_size, bits, transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

template <typename T>
void _bs_qmm_dispatch_typed(
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

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<T>();
  auto biases_ptr = biases.data<T>();
  auto lhs_indices_ptr = lhs_indices.data<uint32_t>();
  auto rhs_indices_ptr = rhs_indices.data<uint32_t>();

  for (int i = 0; i < lhs_indices.size(); i++) {
    int x_idx = lhs_indices_ptr[elem_to_loc(
        i, lhs_indices.shape(), lhs_indices.strides())];
    int w_idx = rhs_indices_ptr[elem_to_loc(
        i, rhs_indices.shape(), rhs_indices.strides())];
    _qmm_dispatch_typed<T>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(x_idx * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(w_idx * w_els, w.shape(), w.strides()),
        scales_ptr +
            elem_to_loc(w_idx * g_els, scales.shape(), scales.strides()),
        biases_ptr +
            elem_to_loc(w_idx * g_els, biases.shape(), biases.strides()),
        M,
        N,
        K,
        bits,
        group_size,
        transposed_w);
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
  switch (x.dtype()) {
    case float32:
      _bs_qmm_dispatch_typed<float>(
          out,
          x,
          w,
          scales,
          biases,
          lhs_indices,
          rhs_indices,
          bits,
          group_size,
          transposed_w);
      break;
    case float16:
      _bs_qmm_dispatch_typed<float16_t>(
          out,
          x,
          w,
          scales,
          biases,
          lhs_indices,
          rhs_indices,
          bits,
          group_size,
          transposed_w);
      break;
    case bfloat16:
      _bs_qmm_dispatch_typed<bfloat16_t>(
          out,
          x,
          w,
          scales,
          biases,
          lhs_indices,
          rhs_indices,
          bits,
          group_size,
          transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}
template <typename T, int group_size, int bits>
void fp_bs_qmm_dispatch_mode(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);

  int w_els = w.shape(-1) * w.shape(-2);
  int g_els = scales.shape(-1) * scales.shape(-2);

  auto out_ptr = out.data<T>();
  auto x_ptr = x.data<T>();
  auto w_ptr = w.data<uint32_t>();
  auto scales_ptr = scales.data<uint8_t>();
  auto lhs_indices_ptr = lhs_indices.data<uint32_t>();
  auto rhs_indices_ptr = rhs_indices.data<uint32_t>();

  for (int i = 0; i < lhs_indices.size(); i++) {
    int x_idx = lhs_indices_ptr[elem_to_loc(
        i, lhs_indices.shape(), lhs_indices.strides())];
    int w_idx = rhs_indices_ptr[elem_to_loc(
        i, rhs_indices.shape(), rhs_indices.strides())];
    fp_qmm_dispatch_transpose<T, group_size, bits>(
        out_ptr + i * M * N,
        x_ptr + elem_to_loc(x_idx * M * K, x.shape(), x.strides()),
        w_ptr + elem_to_loc(w_idx * w_els, w.shape(), w.strides()),
        scales_ptr +
            elem_to_loc(w_idx * g_els, scales.shape(), scales.strides()),
        M,
        N,
        K,
        transposed_w);
  }
}

template <typename T>
void fp_bs_qmm_dispatch_typed(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    int group_size,
    int bits,
    bool transposed_w) {
  if (bits == 8) {
    fp_bs_qmm_dispatch_mode<T, 32, 8>(
        out, x, w, scales, lhs_indices, rhs_indices, transposed_w);
  } else if (group_size == 32) {
    fp_bs_qmm_dispatch_mode<T, 32, 4>(
        out, x, w, scales, lhs_indices, rhs_indices, transposed_w);
  } else {
    fp_bs_qmm_dispatch_mode<T, 16, 4>(
        out, x, w, scales, lhs_indices, rhs_indices, transposed_w);
  }
}

void fp_bs_qmm_dispatch(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    const array& lhs_indices,
    const array& rhs_indices,
    int group_size,
    int bits,
    bool transposed_w) {
  switch (x.dtype()) {
    case float32:
      fp_bs_qmm_dispatch_typed<float>(
          out,
          x,
          w,
          scales,
          lhs_indices,
          rhs_indices,
          group_size,
          bits,
          transposed_w);
      break;
    case float16:
      fp_bs_qmm_dispatch_typed<float16_t>(
          out,
          x,
          w,
          scales,
          lhs_indices,
          rhs_indices,
          group_size,
          bits,
          transposed_w);
      break;
    case bfloat16:
      fp_bs_qmm_dispatch_typed<bfloat16_t>(
          out,
          x,
          w,
          scales,
          lhs_indices,
          rhs_indices,
          group_size,
          bits,
          transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

} // namespace

namespace {

template <typename T>
void kquant_dequantize_dispatch(
    const uint8_t* w,
    T* out,
    size_t num_weights,
    const std::string& kquant_type);

template <typename T>
void kquant_qmm_cpu(
    T* result,
    const T* x,
    const uint8_t* w,
    int M,
    int N,
    int K,
    bool transpose_w,
    const std::string& kquant_type);

} // namespace

void QuantizedMatmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];

  auto& encoder = cpu::get_command_encoder(stream());
  auto x = ensure_row_contiguous(x_pre, encoder, stream());
  auto w = ensure_row_contiguous(w_pre, encoder, stream());
  auto scales = ensure_row_contiguous(scales_pre, encoder, stream());

  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_output_array(out);
  if (mode_ == QuantizationMode::Affine) {
    auto biases = ensure_row_contiguous(inputs[3], encoder, stream());
    encoder.set_input_array(biases);
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      scales = array::unsafe_weak_copy(scales),
                      biases = array::unsafe_weak_copy(biases),
                      group_size_ = group_size_,
                      bits_ = bits_,
                      transpose_ = transpose_]() mutable {
      _qmm_dispatch(out, x, w, scales, biases, group_size_, bits_, transpose_);
    });
  } else if (mode_ == QuantizationMode::KQuant) {
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      transpose_ = transpose_,
                      kquant_type = kquant_type_]() mutable {
      int K = x.shape(-1);
      int M = x.ndim() > 1 ? x.shape(-2) : 1;
      int N = out.shape(-1);
      int batch_size = x.size() / (K * M);
      size_t w_batch_els = w.ndim() > 2 ? w.shape(-1) * w.shape(-2) : 0;
      switch (x.dtype()) {
        case float32:
          for (int i = 0; i < batch_size; i++) {
            kquant_qmm_cpu<float>(
                out.data<float>() + i * M * N,
                x.data<float>() +
                    elem_to_loc(i * M * K, x.shape(), x.strides()),
                w.data<uint8_t>() +
                    elem_to_loc(i * w_batch_els, w.shape(), w.strides()),
                M,
                N,
                K,
                transpose_,
                kquant_type);
          }
          break;
        case float16:
          for (int i = 0; i < batch_size; i++) {
            kquant_qmm_cpu<float16_t>(
                out.data<float16_t>() + i * M * N,
                x.data<float16_t>() +
                    elem_to_loc(i * M * K, x.shape(), x.strides()),
                w.data<uint8_t>() +
                    elem_to_loc(i * w_batch_els, w.shape(), w.strides()),
                M,
                N,
                K,
                transpose_,
                kquant_type);
          }
          break;
        case bfloat16:
          for (int i = 0; i < batch_size; i++) {
            kquant_qmm_cpu<bfloat16_t>(
                out.data<bfloat16_t>() + i * M * N,
                x.data<bfloat16_t>() +
                    elem_to_loc(i * M * K, x.shape(), x.strides()),
                w.data<uint8_t>() +
                    elem_to_loc(i * w_batch_els, w.shape(), w.strides()),
                M,
                N,
                K,
                transpose_,
                kquant_type);
          }
          break;
        default:
          throw std::invalid_argument(
              "[quantized_matmul] only floating types are supported");
      }
    });
  } else {
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      scales = array::unsafe_weak_copy(scales),
                      group_size_ = group_size_,
                      bits_ = bits_,
                      transpose_ = transpose_]() mutable {
      fp_qmm_dispatch(out, x, w, scales, group_size_, bits_, transpose_);
    });
  }
}

void GatherQMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (mode_ == QuantizationMode::KQuant) {
    auto& encoder = cpu::get_command_encoder(stream());
    auto x = ensure_row_contiguous(inputs[0], encoder, stream());
    auto w = ensure_row_contiguous(inputs[1], encoder, stream());
    auto& lhs_indices = inputs[inputs.size() - 2];
    auto& rhs_indices = inputs[inputs.size() - 1];

    out.set_data(allocator::malloc(out.nbytes()));
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(lhs_indices);
    encoder.set_input_array(rhs_indices);
    encoder.set_output_array(out);
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      lhs_indices = array::unsafe_weak_copy(lhs_indices),
                      rhs_indices = array::unsafe_weak_copy(rhs_indices),
                      transpose_ = transpose_,
                      kquant_type = kquant_type_]() mutable {
      int K = x.shape(-1);
      int M = x.shape(-2);
      int N = out.shape(-1);
      int w_els = w.shape(-1) * w.shape(-2);
      auto lhs_ptr = lhs_indices.data<uint32_t>();
      auto rhs_ptr = rhs_indices.data<uint32_t>();

      auto gather_loop = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        for (int i = 0; i < lhs_indices.size(); i++) {
          int x_idx = lhs_ptr[elem_to_loc(
              i, lhs_indices.shape(), lhs_indices.strides())];
          int w_idx = rhs_ptr[elem_to_loc(
              i, rhs_indices.shape(), rhs_indices.strides())];
          kquant_qmm_cpu<T>(
              out.data<T>() + i * M * N,
              x.data<T>() + elem_to_loc(x_idx * M * K, x.shape(), x.strides()),
              w.data<uint8_t>() +
                  elem_to_loc(w_idx * w_els, w.shape(), w.strides()),
              M,
              N,
              K,
              transpose_,
              kquant_type);
        }
      };
      switch (x.dtype()) {
        case float32:
          gather_loop(static_cast<float*>(nullptr));
          break;
        case float16:
          gather_loop(static_cast<float16_t*>(nullptr));
          break;
        case bfloat16:
          gather_loop(static_cast<bfloat16_t*>(nullptr));
          break;
        default:
          throw std::invalid_argument(
              "[quantized_matmul] only floating types are supported");
      }
    });
    return;
  }
  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& lhs_indices = inputs[inputs.size() - 2];
  auto& rhs_indices = inputs[inputs.size() - 1];

  auto& encoder = cpu::get_command_encoder(stream());
  auto ensure_row_contiguous_last_dims = [s = stream(),
                                          &encoder](const array& arr) {
    auto stride_0 = arr.strides()[arr.ndim() - 2];
    auto stride_1 = arr.strides()[arr.ndim() - 1];
    if (stride_0 == arr.shape(-1) && stride_1 == 1) {
      return arr;
    } else {
      auto arr_cpy = array(arr.shape(), arr.dtype(), nullptr, {});
      copy_cpu(arr, arr_cpy, CopyType::General, s);
      encoder.add_temporary(arr_cpy);
      return arr_cpy;
    }
  };

  auto x = ensure_row_contiguous_last_dims(x_pre);
  auto w = ensure_row_contiguous_last_dims(w_pre);
  auto scales = ensure_row_contiguous_last_dims(scales_pre);

  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);
  if (mode_ == QuantizationMode::Affine) {
    auto biases = ensure_row_contiguous_last_dims(inputs[3]);
    encoder.set_input_array(biases);
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      scales = array::unsafe_weak_copy(scales),
                      biases = array::unsafe_weak_copy(biases),
                      lhs_indices = array::unsafe_weak_copy(lhs_indices),
                      rhs_indices = array::unsafe_weak_copy(rhs_indices),
                      group_size_ = group_size_,
                      bits_ = bits_,
                      transpose_ = transpose_]() mutable {
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
    });
  } else {
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      w = array::unsafe_weak_copy(w),
                      scales = array::unsafe_weak_copy(scales),
                      lhs_indices = array::unsafe_weak_copy(lhs_indices),
                      rhs_indices = array::unsafe_weak_copy(rhs_indices),
                      group_size_ = group_size_,
                      bits_ = bits_,
                      transpose_ = transpose_]() mutable {
      fp_bs_qmm_dispatch(
          out,
          x,
          w,
          scales,
          lhs_indices,
          rhs_indices,
          group_size_,
          bits_,
          transpose_);
    });
  }
}

uint8_t to_fp8_e8m0(float x) {
  if (!std::isfinite(x)) {
    return 0xFF;
  }
  if (x < 0.0f) {
    return 0x00;
  }
  float le = std::log2(x);
  int n = int(std::round(le));

  n = n < -127 ? -127 : n;
  n = n > 127 ? 127 : n;
  return static_cast<uint8_t>(n + 127);
}

uint8_t to_fp4_e2m1(float x) {
  if (std::isnan(x)) {
    return 0x7;
  }

  const uint8_t sign_bit = (std::signbit(x)) ? 0x8 : 0x0;
  x = std::abs(x);

  uint8_t bits;
  if (x > 5.0f) {
    bits = 0x7;
  } else if (x >= 3.5f) {
    bits = 0x6;
  } else if (x > 2.5f) {
    bits = 0x5;
  } else if (x >= 1.75f) {
    bits = 0x4;
  } else if (x > 1.25f) {
    bits = 0x3;
  } else if (x >= 0.75f) {
    bits = 0x2;
  } else if (x > 0.25f) {
    bits = 0x1;
  } else {
    bits = 0x0;
  }
  return bits | sign_bit;
}

template <typename T>
void fp_quantize_dequantize(
    const array& w_arr,
    array& out_arr,
    int bits,
    int group_size,
    size_t w_size) {
  auto w = w_arr.data<T>();
  auto out = out_arr.data<T>();

  size_t n_groups = w_size / group_size;

  for (size_t i = 0; i < n_groups; ++i) {
    size_t idx = i * group_size;
    float scale = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < group_size; ++j) {
      scale = std::max(scale, std::abs(w[idx + j]));
    }
    scale /= bits == 4 ? 6.0f : 448.0f;
    if (group_size == 16) {
      scale = dequantize_scale<float, 16>(detail::ToFP8()(scale));
    } else {
      scale = dequantize_scale<float, 32>(to_fp8_e8m0(scale));
    }

    for (int j = 0; j < group_size; ++j) {
      float w_el = scale == 0 ? 0.0f : w[idx + j] / scale;
      float output;
      if (bits == 8) {
        output = detail::FromFP8()(detail::ToFP8()(w_el));
      } else {
        output = FP4_LUT[to_fp4_e2m1(w_el)];
      }
      out[idx + j] = static_cast<T>(scale * output);
    }
  }
}

void dispatch_quantize_dequantize(
    const array& w,
    array& out,
    int bits,
    int group_size) {
  if (w.dtype() == float16) {
    fp_quantize_dequantize<float16_t>(w, out, bits, group_size, w.size());
  } else if (w.dtype() == bfloat16) {
    fp_quantize_dequantize<bfloat16_t>(w, out, bits, group_size, w.size());
  } else if (w.dtype() == float32) {
    fp_quantize_dequantize<float>(w, out, bits, group_size, w.size());
  } else {
    throw std::runtime_error(
        "[quantize_dequantize] Only supports floating point inputs");
  }
}

template <typename T, typename U>
void quantize(
    const T* w,
    U* out,
    T* scales,
    T* biases,
    int bits,
    int group_size,
    size_t w_size) {
  float n_bins = (1 << bits) - 1;
  float eps = 1e-7;

  bool power_of_2_bits = is_power_of_2(bits);
  int el_per_int = get_pack_factor(bits, 32);
  int bytes_per_pack = get_bytes_per_pack(bits);
  int int_per_group = group_size * bytes_per_pack / el_per_int;
  size_t n_groups = w_size / group_size;

  for (size_t i = 0; i < n_groups; ++i) {
    size_t w_idx = i * group_size;
    float w_min = std::numeric_limits<float>::infinity();
    float w_max = -w_min;
    for (int j = 0; j < group_size; ++j) {
      w_max = std::max(w_max, (float)w[w_idx + j]);
      w_min = std::min(w_min, (float)w[w_idx + j]);
    }
    bool mask = std::abs(w_min) > std::abs(w_max);
    float scale = std::max((w_max - w_min) / n_bins, eps);
    scale = mask ? scale : -scale;

    float edge = mask ? w_min : w_max;
    float q0 = std::rint(edge / scale);
    float bias = 0;
    if (q0 != 0) {
      scale = edge / q0;
      bias = edge;
    }
    size_t out_idx = i * int_per_group;
    for (int j = 0; j < int_per_group / bytes_per_pack; ++j) {
      uint64_t out_el = 0;
      for (int k = 0; k < el_per_int; ++k) {
        float w_el = w[w_idx + j * el_per_int + k];
        w_el = std::rint((w_el - bias) / scale);
        w_el = std::min(std::max(w_el, 0.0f), n_bins);
        out_el |= static_cast<uint64_t>(w_el) << (k * bits);
      }
      if (power_of_2_bits) {
        out[out_idx + j] = out_el;
      } else if (bits == 5) {
        out[out_idx + bytes_per_pack * j] = out_el & 0xff;
        out[out_idx + bytes_per_pack * j + 1] = (out_el & 0xff00) >> 8;
        out[out_idx + bytes_per_pack * j + 2] = (out_el & 0xff0000) >> 16;
        out[out_idx + bytes_per_pack * j + 3] = (out_el & 0xff000000) >> 24;
        out[out_idx + bytes_per_pack * j + 4] = (out_el & 0xff00000000) >> 32;
      } else {
        out[out_idx + bytes_per_pack * j] = out_el & 0xff;
        out[out_idx + bytes_per_pack * j + 1] = (out_el & 0xff00) >> 8;
        out[out_idx + bytes_per_pack * j + 2] = (out_el & 0xff0000) >> 16;
      }
    }
    scales[i] = static_cast<T>(scale);
    biases[i] = static_cast<T>(bias);
  }
}

template <typename T, typename U>
void dispatch_quantize(
    const array& w,
    array& out,
    array& scales,
    array& biases,
    int bits,
    int group_size) {
  auto w_ptr = w.data<T>();
  auto out_ptr = out.data<U>();
  auto scales_ptr = scales.data<T>();
  auto biases_ptr = biases.data<T>();
  quantize<T, U>(
      w_ptr, out_ptr, scales_ptr, biases_ptr, bits, group_size, w.size());
}

namespace {

inline float read_f16(const uint8_t* ptr) {
  _Float16 tmp;
  std::memcpy(&tmp, ptr, sizeof(_Float16));
  return static_cast<float>(tmp);
}

template <typename T>
void kquant_dequantize_q8_0(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 34;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    T* dst = out + b * block_weights;
    for (int i = 0; i < block_weights; i++) {
      dst[i] = static_cast<T>(d * static_cast<float>(qs[i]));
    }
  }
}

template <typename T>
void kquant_dequantize_q4_0(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 18;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qs = block + 2;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = (qs[j] & 0x0F) - 8;
      int x1 = (qs[j] >> 4) - 8;
      dst[j] = static_cast<T>(d * static_cast<float>(x0));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1));
    }
  }
}

template <typename T>
void kquant_dequantize_q4_1(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 20;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qs = block + 4;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int x0 = qs[j] & 0x0F;
      int x1 = qs[j] >> 4;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

template <typename T>
void kquant_dequantize_q5_0(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 22;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    const uint8_t* qh_bytes = block + 2;
    uint32_t qh = static_cast<uint32_t>(qh_bytes[0]) |
        (static_cast<uint32_t>(qh_bytes[1]) << 8) |
        (static_cast<uint32_t>(qh_bytes[2]) << 16) |
        (static_cast<uint32_t>(qh_bytes[3]) << 24);
    const uint8_t* qs = block + 6;
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      int xh_0 = ((qh >> j) << 4) & 0x10;
      int xh_1 = (qh >> (j + 12)) & 0x10;
      int x0 = (qs[j] & 0x0F) | xh_0;
      int x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0 - 16));
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1 - 16));
    }
  }
}

template <typename T>
void kquant_dequantize_q5_1(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 32;
  constexpr int block_bytes = 24;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float m = read_f16(block + 2);
    const uint8_t* qh_bytes = block + 4;
    const uint8_t* qs = block + 8;
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    T* dst = out + b * block_weights;
    for (int j = 0; j < 16; j++) {
      uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
      uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
      uint8_t x0 = (qs[j] & 0x0F) | xh_0;
      uint8_t x1 = (qs[j] >> 4) | xh_1;
      dst[j] = static_cast<T>(d * static_cast<float>(x0) + m);
      dst[j + 16] = static_cast<T>(d * static_cast<float>(x1) + m);
    }
  }
}

inline void kquant_unpack_q4k_scales(
    const uint8_t* scales_packed,
    float* sc,
    float* mn,
    float d,
    float dmin) {
  for (int i = 0; i < 8; i++) {
    uint8_t raw_sc, raw_m;
    if (i < 4) {
      raw_sc = scales_packed[i] & 0x3F;
      raw_m = scales_packed[i + 4] & 0x3F;
    } else {
      raw_sc =
          (scales_packed[i + 4] & 0x0F) | ((scales_packed[i - 4] >> 6) << 4);
      raw_m = (scales_packed[i + 4] >> 4) | ((scales_packed[i] >> 6) << 4);
    }
    sc[i] = d * static_cast<float>(raw_sc);
    mn[i] = dmin * static_cast<float>(raw_m);
  }
}

template <typename T>
void kquant_dequantize_q4_k(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 144;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qs = block + 16;

    float sc[8], mn[8];
    kquant_unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(qs[g * 32 + i] & 0x0F) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(qs[g * 32 + i] >> 4) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void kquant_dequantize_q5_k(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 176;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales_packed = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* qs = block + 48;

    float sc[8], mn[8];
    kquant_unpack_q4k_scales(scales_packed, sc, mn, d, dmin);

    T* dst = out + b * block_weights;
    for (int g = 0; g < 4; g++) {
      for (int i = 0; i < 32; i++) {
        uint8_t lo0 = qs[g * 32 + i] & 0x0F;
        uint8_t lo1 = qs[g * 32 + i] >> 4;
        uint8_t hi0 = (qh[i] >> (2 * g)) & 1;
        uint8_t hi1 = (qh[i] >> (2 * g + 1)) & 1;
        dst[(2 * g) * 32 + i] = static_cast<T>(
            sc[2 * g] * static_cast<float>(lo0 | (hi0 << 4)) - mn[2 * g]);
        dst[(2 * g + 1) * 32 + i] = static_cast<T>(
            sc[2 * g + 1] * static_cast<float>(lo1 | (hi1 << 4)) -
            mn[2 * g + 1]);
      }
    }
  }
}

template <typename T>
void kquant_dequantize_q6_k(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 210;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* ql_base = block;
    const uint8_t* qh_base = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    float d = read_f16(block + 208);

    T* dst = out + b * block_weights;
    for (int half = 0; half < 2; half++) {
      const uint8_t* ql = ql_base + half * 64;
      const uint8_t* qh = qh_base + half * 32;
      const int8_t* sc = scales + half * 8;
      T* out_half = dst + half * 128;

      for (int l = 0; l < 32; l++) {
        int is0 = l / 16;
        int8_t q1 =
            static_cast<int8_t>((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) -
            32;
        int8_t q2 = static_cast<int8_t>(
                        (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) -
            32;
        int8_t q3 =
            static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int8_t q4 =
            static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) -
            32;
        out_half[l] = static_cast<T>(
            d * static_cast<float>(sc[is0]) * static_cast<float>(q1));
        out_half[l + 32] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 2]) * static_cast<float>(q2));
        out_half[l + 64] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 4]) * static_cast<float>(q3));
        out_half[l + 96] = static_cast<T>(
            d * static_cast<float>(sc[is0 + 6]) * static_cast<float>(q4));
      }
    }
  }
}

inline void kquant_unpack_q3k_scales(const uint8_t* s, int32_t* sc) {
  for (int k = 0; k < 4; k++) {
    sc[k] = static_cast<int32_t>(s[k] & 0x0F) |
        (static_cast<int32_t>((s[8 + k]) & 0x03) << 4);
    sc[k + 4] = static_cast<int32_t>(s[k + 4] & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 2) & 0x03) << 4);
    sc[k + 8] = static_cast<int32_t>((s[k] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 4) & 0x03) << 4);
    sc[k + 12] = static_cast<int32_t>((s[k + 4] >> 4) & 0x0F) |
        (static_cast<int32_t>((s[8 + k] >> 6) & 0x03) << 4);
  }
  for (int i = 0; i < 16; i++) {
    sc[i] -= 32;
  }
}

template <typename T>
void kquant_dequantize_q3_k(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 110;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* hmask = block;
    const uint8_t* qs_full = block + 32;
    const uint8_t* scales_packed = block + 96;
    float d = read_f16(block + 108);

    int32_t sc[16];
    kquant_unpack_q3k_scales(scales_packed, sc);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t m = 1 << (outer_half * 4 + shift_idx);
        int is_left = outer_half * 8 + shift_idx * 2;
        float dl_left = d * static_cast<float>(sc[is_left]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          int h = (hmask[l] & m) ? 0 : 4;
          dst[out_idx++] = static_cast<T>(dl_left * static_cast<float>(q2 - h));
        }
        float dl_right = d * static_cast<float>(sc[is_left + 1]);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          int h = (hmask[l + 16] & m) ? 0 : 4;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2 - h));
        }
      }
    }
  }
}

template <typename T>
void kquant_dequantize_q2_k(const uint8_t* w, T* out, size_t num_weights) {
  constexpr int block_weights = 256;
  constexpr int block_bytes = 84;
  size_t num_blocks = num_weights / block_weights;
  for (size_t b = 0; b < num_blocks; b++) {
    const uint8_t* block = w + b * block_bytes;
    const uint8_t* scales_raw = block;
    const uint8_t* qs_full = block + 16;
    float d = read_f16(block + 80);
    float dmin = read_f16(block + 82);

    T* dst = out + b * block_weights;
    int out_idx = 0;
    int is_idx = 0;
    for (int outer_half = 0; outer_half < 2; outer_half++) {
      const uint8_t* qs_chunk = qs_full + outer_half * 32;
      for (int shift_idx = 0; shift_idx < 4; shift_idx++) {
        int shift = shift_idx * 2;
        uint8_t sc_byte_left = scales_raw[is_idx++];
        float dl_left = d * static_cast<float>(sc_byte_left & 0x0F);
        float ml_left = dmin * static_cast<float>(sc_byte_left >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_left * static_cast<float>(q2) - ml_left);
        }
        uint8_t sc_byte_right = scales_raw[is_idx++];
        float dl_right = d * static_cast<float>(sc_byte_right & 0x0F);
        float ml_right = dmin * static_cast<float>(sc_byte_right >> 4);
        for (int l = 0; l < 16; l++) {
          int q2 = (qs_chunk[l + 16] >> shift) & 3;
          dst[out_idx++] =
              static_cast<T>(dl_right * static_cast<float>(q2) - ml_right);
        }
      }
    }
  }
}

template <typename T>
void kquant_dequantize_dispatch(
    const uint8_t* w,
    T* out,
    size_t num_weights,
    const std::string& kquant_type) {
  if (kquant_type == "q8_0") {
    kquant_dequantize_q8_0(w, out, num_weights);
  } else if (kquant_type == "q4_0") {
    kquant_dequantize_q4_0(w, out, num_weights);
  } else if (kquant_type == "q4_1") {
    kquant_dequantize_q4_1(w, out, num_weights);
  } else if (kquant_type == "q5_0") {
    kquant_dequantize_q5_0(w, out, num_weights);
  } else if (kquant_type == "q5_1") {
    kquant_dequantize_q5_1(w, out, num_weights);
  } else if (kquant_type == "q4_k") {
    kquant_dequantize_q4_k(w, out, num_weights);
  } else if (kquant_type == "q5_k") {
    kquant_dequantize_q5_k(w, out, num_weights);
  } else if (kquant_type == "q6_k") {
    kquant_dequantize_q6_k(w, out, num_weights);
  } else if (kquant_type == "q3_k") {
    kquant_dequantize_q3_k(w, out, num_weights);
  } else if (kquant_type == "q2_k") {
    kquant_dequantize_q2_k(w, out, num_weights);
  } else {
    throw std::runtime_error(
        "[kquant_dequantize] Unsupported codec: " + kquant_type);
  }
}

template <typename T>
void kquant_qmm_cpu(
    T* result,
    const T* x,
    const uint8_t* w,
    int M,
    int N,
    int K,
    bool transpose_w,
    const std::string& kquant_type) {
  const auto* codec = kquant_codec_by_name(kquant_type);
  int w_rows = transpose_w ? N : K;
  int w_cols = transpose_w ? K : N;
  size_t weights_per_row = static_cast<size_t>(w_cols);
  size_t row_bytes =
      (weights_per_row / codec->weights_per_block) * codec->bytes_per_block;

  std::vector<float> w_dec(static_cast<size_t>(w_rows) * w_cols);
  for (int r = 0; r < w_rows; r++) {
    kquant_dequantize_dispatch(
        w + r * row_bytes, w_dec.data() + r * w_cols, w_cols, kquant_type);
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0.0f;
      if (transpose_w) {
        for (int k = 0; k < K; k++) {
          acc += static_cast<float>(x[m * K + k]) * w_dec[n * K + k];
        }
      } else {
        for (int k = 0; k < K; k++) {
          acc += static_cast<float>(x[m * K + k]) * w_dec[k * N + n];
        }
      }
      result[m * N + n] = static_cast<T>(acc);
    }
  }
}

} // namespace

void fast::Quantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (mode_ == QuantizationMode::KQuant) {
    if (!dequantize_) {
      throw std::runtime_error(
          "[fast::Quantize::eval_cpu] KQuant encode is GPU-only.");
    }
    auto& encoder = cpu::get_command_encoder(stream());
    auto w = ensure_row_contiguous(inputs[0], encoder, stream());
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));
    encoder.set_input_array(w);
    encoder.set_output_array(out);
    size_t num_weights = out.size();
    encoder.dispatch([w = array::unsafe_weak_copy(w),
                      out = array::unsafe_weak_copy(out),
                      num_weights,
                      kquant_type = kquant_type_]() mutable {
      auto w_ptr = w.data<uint8_t>();
      switch (out.dtype()) {
        case float32:
          kquant_dequantize_dispatch(
              w_ptr, out.data<float>(), num_weights, kquant_type);
          break;
        case float16:
          kquant_dequantize_dispatch(
              w_ptr, out.data<float16_t>(), num_weights, kquant_type);
          break;
        case bfloat16:
          kquant_dequantize_dispatch(
              w_ptr, out.data<bfloat16_t>(), num_weights, kquant_type);
          break;
        default:
          throw std::runtime_error(
              "[fast::Quantize::eval_cpu] KQuant dequantize only supports float types.");
      }
    });
    return;
  }
  auto& encoder = cpu::get_command_encoder(stream());
  auto w = ensure_row_contiguous(inputs[0], encoder, stream());
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& scales = outputs[1];
  auto& biases = outputs[2];
  scales.set_data(allocator::malloc(scales.nbytes()));
  biases.set_data(allocator::malloc(biases.nbytes()));
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  encoder.set_input_array(biases);
  encoder.set_output_array(out);
  encoder.dispatch([w = array::unsafe_weak_copy(w),
                    out = array::unsafe_weak_copy(out),
                    scales = array::unsafe_weak_copy(scales),
                    biases = array::unsafe_weak_copy(biases),
                    group_size_ = group_size_,
                    bits_ = bits_]() mutable {
    if (w.dtype() == float16) {
      if (is_power_of_2(bits_)) {
        dispatch_quantize<float16_t, uint32_t>(
            w, out, scales, biases, bits_, group_size_);
      } else {
        dispatch_quantize<float16_t, uint8_t>(
            w, out, scales, biases, bits_, group_size_);
      }
    } else if (w.dtype() == bfloat16) {
      if (is_power_of_2(bits_)) {
        dispatch_quantize<bfloat16_t, uint32_t>(
            w, out, scales, biases, bits_, group_size_);
      } else {
        dispatch_quantize<bfloat16_t, uint8_t>(
            w, out, scales, biases, bits_, group_size_);
      }
    } else if (w.dtype() == float32) {
      if (is_power_of_2(bits_)) {
        dispatch_quantize<float, uint32_t>(
            w, out, scales, biases, bits_, group_size_);
      } else {
        dispatch_quantize<float, uint8_t>(
            w, out, scales, biases, bits_, group_size_);
      }
    } else {
      throw std::runtime_error(
          "[fast::Quantize::eval_cpu] Only supports floating point inputs");
    }
  });
}

void fast::ConvertFP8::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  set_unary_output_data(in, out);
  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.dispatch([in = array::unsafe_weak_copy(in),
                    out = array::unsafe_weak_copy(out),
                    to_fp8 = to_fp8_]() mutable {
    if (to_fp8) {
      switch (in.dtype()) {
        case float16:
          unary_op<float16_t, uint8_t>(in, out, detail::ToFP8());
          break;
        case bfloat16:
          unary_op<bfloat16_t, uint8_t>(in, out, detail::ToFP8());
          break;
        default:
          unary_op<float, uint8_t>(in, out, detail::ToFP8());
          break;
      }
    } else {
      switch (out.dtype()) {
        case float16:
          unary_op<uint8_t, float16_t>(in, out, detail::FromFP8());
          break;
        case bfloat16:
          unary_op<uint8_t, bfloat16_t>(in, out, detail::FromFP8());
          break;
        default:
          unary_op<uint8_t, float>(in, out, detail::FromFP8());
          break;
      }
    }
  });
}

void QQMatmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = cpu::get_command_encoder(stream());

  bool w_quantized = (inputs[1].dtype() == uint32);
  if (w_quantized && inputs[0].shape(-2) == 1) {
    bool donate_x = inputs[0].is_donatable();
    auto x = ensure_row_contiguous(inputs[0], encoder, stream());
    auto w = ensure_row_contiguous(inputs[1], encoder, stream());
    auto scales = ensure_row_contiguous(inputs[2], encoder, stream());

    out.set_data(allocator::malloc(out.nbytes()));

    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
    if (!donate_x) {
      encoder.add_temporary(xhat);
    }
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_output_array(out);
    encoder.dispatch([out = array::unsafe_weak_copy(out),
                      x = array::unsafe_weak_copy(x),
                      xhat = array::unsafe_weak_copy(xhat),
                      w = array::unsafe_weak_copy(w),
                      scales = array::unsafe_weak_copy(scales),
                      group_size_ = group_size_,
                      bits_ = bits_]() mutable {
      dispatch_quantize_dequantize(x, xhat, bits_, group_size_);
      fp_qmm_dispatch(out, xhat, w, scales, group_size_, bits_, true);
    });
    return;
  } else {
    throw std::runtime_error("[QQMatmul] NYI for the general case");
  }
}

} // namespace mlx::core
