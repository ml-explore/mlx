// Copyright © 2023 Apple Inc.

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

inline constexpr short get_pack_factor(int bits, int wsize = 8) {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

inline constexpr short get_bytes_per_pack(int bits, int wsize = 8) {
  auto power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
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
    case 1:
      _qmm_dispatch_group<T, 1>(
          result, x, w, scales, biases, M, N, K, group_size, transposed_w);
      break;
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
      throw std::invalid_argument(
          "Quantization bits must be 1, 2, 3, 4, 5, 6 or 8.");
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
    float scale;
    float bias;

    if (bits == 1) {
      // Affine 1-bit: bit 0 -> w_min, bit 1 -> w_max
      scale = std::max(w_max - w_min, eps);
      bias = w_min;
    } else {
      scale = std::max((w_max - w_min) / n_bins, eps);
      scale = mask ? scale : -scale;

      float edge = mask ? w_min : w_max;
      float q0 = std::rint(edge / scale);
      bias = 0;
      if (q0 != 0) {
        scale = edge / q0;
        bias = edge;
      }
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

void fast::Quantize::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
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
