// Copyright © 2023-2026 Apple Inc.

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/backend/cpu/scratch.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading/common.h"
#include "mlx/backend/cpu/unary.h"
#include "mlx/backend/cpu/unary_ops.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <type_traits>
#include <vector>

// ISA-specific SIMD implementations (int8 kernels, dequant, LUT lookups).
// Highway kernels are compiled as runtime-dispatched translation units when x86
// CPU SIMD is enabled.
#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/quantized_highway.h"
#else
// No ISA-specific SIMD available. Provide no-op stubs to fall through to the
// generic SIMD paths in quantized.cpp which work on any platform via the
// portable Simd<T,N> layer.
#include "mlx/backend/cpu/quantized.h"
namespace mlx::core {

template <typename T, int bits, int group_size>
bool try_int8_simd_row(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int n_start,
    int n_end,
    int K,
    int groups_per_col,
    float* x_group_sums,
    const PreqAct* preq) {
  return false;
}

template <typename T, int bits, int group_size>
bool try_int8_preq_parallel(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int N,
    int K,
    int n_threads,
    int n_chunks,
    std::atomic<int>& steal_counter,
    int CHUNK_COLS) {
  return false;
}

template <int S>
simd::Simd<float, S> fp4_lut_lookup_simd(
    const simd::Simd<uint32_t, S>& wi,
    const float* FP4_LUT_ptr) {
  alignas(32) float tmp[S];
  for (int i = 0; i < S; ++i) {
    tmp[i] = FP4_LUT_ptr[wi[i]];
  }
  return simd::load<float, S>(tmp);
}

template <int S>
simd::Simd<float, S> fp8_lut_gather_simd(
    const uint32_t* w,
    const float* FP8_LUT_ptr) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(w);
  alignas(32) float tmp[S];
  for (int i = 0; i < S; ++i) {
    tmp[i] = FP8_LUT_ptr[bytes[i]];
  }
  return simd::load<float, S>(tmp);
}

} // namespace mlx::core
#endif // ISA dispatch

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
}

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

// FP8 E4M3 -> float32 lookup table (256 entries, 1KB).
// Precomputed from the FromFP8 decode: strip sign, interpret lower 7 bits
// as float16 mantissa/exponent (<<7), multiply by 256, apply sign.
// Used by FP quantized matmul for O(1) FP8 weight decoding.
// clang-format off
const static float FP8_LUT[256] = {
       0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
  0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
   0.03125f,  0.03515625f,  0.0390625f,  0.04296875f,  0.046875f,  0.05078125f,  0.0546875f,  0.05859375f,
    0.0625f,   0.0703125f,   0.078125f,   0.0859375f,   0.09375f,   0.1015625f,   0.109375f,   0.1171875f,
     0.125f,    0.140625f,    0.15625f,    0.171875f,    0.1875f,    0.203125f,    0.21875f,    0.234375f,
      0.25f,     0.28125f,     0.3125f,     0.34375f,     0.375f,     0.40625f,     0.4375f,     0.46875f,
       0.5f,      0.5625f,      0.625f,      0.6875f,      0.75f,      0.8125f,      0.875f,      0.9375f,
       1.0f,       1.125f,       1.25f,       1.375f,       1.5f,       1.625f,       1.75f,       1.875f,
       2.0f,        2.25f,        2.5f,        2.75f,        3.0f,        3.25f,        3.5f,        3.75f,
       4.0f,         4.5f,         5.0f,        5.5f,        6.0f,        6.5f,        7.0f,        7.5f,
       8.0f,         9.0f,        10.0f,       11.0f,       12.0f,       13.0f,       14.0f,       15.0f,
      16.0f,        18.0f,        20.0f,       22.0f,       24.0f,       26.0f,       28.0f,       30.0f,
      32.0f,        36.0f,        40.0f,       44.0f,       48.0f,       52.0f,       56.0f,       60.0f,
      64.0f,        72.0f,        80.0f,       88.0f,       96.0f,      104.0f,      112.0f,      120.0f,
     128.0f,       144.0f,       160.0f,      176.0f,      192.0f,      208.0f,      224.0f,      240.0f,
     256.0f,       288.0f,       320.0f,      352.0f,      384.0f,      416.0f,      448.0f,      480.0f,
      -0.0f,-0.001953125f,-0.00390625f,-0.005859375f,-0.0078125f,-0.009765625f,-0.01171875f,-0.013671875f,
 -0.015625f,-0.017578125f,-0.01953125f,-0.021484375f,-0.0234375f,-0.025390625f,-0.02734375f,-0.029296875f,
  -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
   -0.0625f,  -0.0703125f,  -0.078125f,  -0.0859375f,  -0.09375f,  -0.1015625f,  -0.109375f,  -0.1171875f,
    -0.125f,   -0.140625f,   -0.15625f,   -0.171875f,   -0.1875f,   -0.203125f,   -0.21875f,   -0.234375f,
     -0.25f,    -0.28125f,    -0.3125f,    -0.34375f,    -0.375f,    -0.40625f,    -0.4375f,    -0.46875f,
      -0.5f,     -0.5625f,     -0.625f,     -0.6875f,     -0.75f,     -0.8125f,     -0.875f,     -0.9375f,
      -1.0f,      -1.125f,      -1.25f,      -1.375f,      -1.5f,      -1.625f,      -1.75f,      -1.875f,
      -2.0f,       -2.25f,       -2.5f,       -2.75f,       -3.0f,       -3.25f,       -3.5f,       -3.75f,
      -4.0f,        -4.5f,       -5.0f,       -5.5f,       -6.0f,       -6.5f,       -7.0f,       -7.5f,
      -8.0f,        -9.0f,      -10.0f,      -11.0f,      -12.0f,      -13.0f,      -14.0f,      -15.0f,
     -16.0f,       -18.0f,      -20.0f,      -22.0f,      -24.0f,      -26.0f,      -28.0f,      -30.0f,
     -32.0f,       -36.0f,      -40.0f,      -44.0f,      -48.0f,      -52.0f,      -56.0f,      -60.0f,
     -64.0f,       -72.0f,      -80.0f,      -88.0f,      -96.0f,     -104.0f,     -112.0f,     -120.0f,
    -128.0f,      -144.0f,     -160.0f,     -176.0f,     -192.0f,     -208.0f,     -224.0f,     -240.0f,
    -256.0f,      -288.0f,     -320.0f,     -352.0f,     -384.0f,     -416.0f,     -448.0f,     -480.0f,
};
// clang-format on

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return static_cast<T>(FP8_LUT[s]);
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

// 3/5/6-bit values straddle byte boundaries, so they have no Highway kernel
// (those static_assert on bits == 4 || bits == 8) and are unpacked one element
// at a time. That leaves them ~8x the 4-bit path, matching 2-bit. Vectorising
// these widths is deliberately deferred -- they are rare in published models.
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

  // Accumulate in float -- see _qmm_t_cols for why T arithmetic is costly here.
  std::vector<float> acc(N);

  for (int m = 0; m < M; m++) {
    const uint8_t* w_local = (const uint8_t*)w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    std::fill(acc.begin(), acc.end(), 0.0f);

    for (int k = 0; k < K; k++) {
      float* result_local = acc.data();
      float xi = static_cast<float>(*x++);

      for (int n = 0; n < N; n += group_size) {
        float scale = static_cast<float>(*scales_local++);
        float bias = static_cast<float>(*biases_local++);
        for (int ng = 0; ng < packs_in_group; ng++) {
          if constexpr (bits == 3 || bits == 5 || bits == 6) {
            float wl[pack_factor];
            extract_bits<float, bits>(w_local, wl);
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
                  xi * (scale * static_cast<float>(wi & bitmask) + bias);
              if (bits != 8) {
                wi >>= bits;
              }
            }
          }
        }
      }
    }

    for (int n = 0; n < N; n++) {
      result[n] = static_cast<T>(acc[n]);
    }
    result += N;
  }
}

// Helper for computing a range of output columns in _qmm_t (used by threaded
// version)
template <typename T, int bits, int group_size>
void _qmm_t_cols(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int n_start,
    int n_end,
    int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int bytes_per_pack = get_bytes_per_pack(bits);
  constexpr int packs_in_group = group_size / pack_factor;

  // Compute strides for navigating to different output columns
  int groups_per_col = K / group_size;
  int bytes_per_col = groups_per_col * packs_in_group * bytes_per_pack;

  // Advance to starting column
  const uint8_t* w_base = (const uint8_t*)w + n_start * bytes_per_col;
  const T* scales_base = scales + n_start * groups_per_col;
  const T* biases_base = biases + n_start * groups_per_col;

  // Compute in float. This file is built without F16C, so float16_t is the
  // software _MLX_Float16 and each T operation below would be a library call.
  // The row is shared by every column, so convert it once rather than per
  // output element.
  std::vector<float> x_storage;
  const float* x_f;
  if constexpr (std::is_same_v<T, float>) {
    x_f = x;
  } else {
    x_storage.resize(K);
    for (int k = 0; k < K; k++) {
      x_storage[k] = static_cast<float>(x[k]);
    }
    x_f = x_storage.data();
  }

  for (int n = n_start; n < n_end; n++) {
    const uint8_t* w_local = w_base;
    const T* scales_local = scales_base;
    const T* biases_local = biases_base;
    const float* x_local = x_f;
    float sum = 0;

    for (int k = 0; k < K; k += group_size) {
      float scale = static_cast<float>(*scales_local++);
      float bias = static_cast<float>(*biases_local++);

      for (int kw = 0; kw < packs_in_group; kw++) {
        if constexpr (bits == 3 || bits == 5 || bits == 6) {
          float wl[pack_factor];
          extract_bits<float, bits>(w_local, wl);
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
            sum += (*x_local++) *
                (scale * static_cast<float>(wi & bitmask) + bias);
            if (bits != 8) {
              wi >>= bits;
            }
          }
        }
      }
    }
    result[n - n_start] = static_cast<T>(sum);

    // Move to next column
    w_base += bytes_per_col;
    scales_base += groups_per_col;
    biases_base += groups_per_col;
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
  auto& pool = cpu::ThreadPool::instance();

  // Adaptive threshold for M-parallelization based on work size (N*K).
  // Larger work per row justifies parallelization overhead.
  // Formula: threshold = max(8, N * K / 1000000) for faster parallelization
  // on typical prompt sizes (7-15 tokens) with reasonable N (e.g., 2048+).
  constexpr int64_t WORK_PER_PIXEL_THRESHOLD = 1000000;
  int m_parallel_threshold =
      std::max(8, static_cast<int>(N * K / WORK_PER_PIXEL_THRESHOLD));
  if (M >= m_parallel_threshold && pool.max_threads() > 1) {
    int m_threads = std::min(pool.max_threads(), M);
    pool.parallel_for(m_threads, [&](int tid, int nth) {
      int m_chunk = (M + nth - 1) / nth;
      int m_start = m_chunk * tid;
      int m_end = std::min(m_start + m_chunk, M);
      for (int m = m_start; m < m_end; m++) {
        _qmm_t_cols<T, bits, group_size>(
            result + m * N, x + m * K, w, scales, biases, 0, N, K);
      }
    });
    return;
  }

  // Single row (M=1) or small M: parallelize over N if beneficial
  int n_threads = std::min(pool.max_threads(), std::max(1, N / 64));

  for (int m = 0; m < M; m++) {
    if (n_threads > 1) {
      // Parallel path: parallelize over output columns N
      pool.parallel_for(n_threads, [&](int tid, int nth) {
        int chunk = (N + nth - 1) / nth;
        int n_start = chunk * tid;
        int n_end = std::min(n_start + chunk, N);
        if (n_start < n_end) {
          _qmm_t_cols<T, bits, group_size>(
              result + n_start, x, w, scales, biases, n_start, n_end, K);
        }
      });
      result += N;
    } else {
      // Same kernel, single-threaded over all columns.
      _qmm_t_cols<T, bits, group_size>(result, x, w, scales, biases, 0, N, K);
      result += N;
    }
    x += K;
  }
}

// Extract S elements starting at element offset `elem_off` within packed words.
// For S >= pack_factor, elem_off must be 0 and this degenerates to the standard
// version. For S < pack_factor (e.g., S=4, pack_factor=8), elem_off
// selects which S-element chunk within the word to extract.
template <int bits, int S>
simd::Simd<uint32_t, S> extract_bits_simd(const uint32_t* w, int elem_off = 0) {
  constexpr int bitmask = (1 << bits) - 1;
  simd::Simd<uint32_t, S> wi;
  if constexpr (S == 1) {
    // Scalar: extract single value from packed word
    wi = simd::Simd<uint32_t, 1>((*w >> (elem_off * bits)) & bitmask);
  } else if constexpr (bits == 4 && S == 4) {
    // 4-lane path: extract 4 4-bit values from part of one uint32_t word.
    // elem_off selects which group of 4 nibbles: 0=lower, 1=upper
    uint32_t word = *w;
    if (elem_off == 1) {
      word >>= 16; // shift to get upper 4 nibbles
    }
    alignas(16) constexpr uint32_t shifts_[] = {0, 4, 8, 12};
    auto shifts = simd::load<uint32_t, S>(shifts_);
    wi = simd::Simd<uint32_t, S>(word);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 2 && S == 4) {
    // 4-lane path: extract 4 2-bit values from part of one uint32_t word.
    // Each uint32 has 16 2-bit values. We extract 4 at a time.
    uint32_t word = *w;
    word >>= (elem_off * S * bits);
    alignas(16) constexpr uint32_t shifts_[] = {0, 2, 4, 6};
    auto shifts = simd::load<uint32_t, S>(shifts_);
    wi = simd::Simd<uint32_t, S>(word);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 8 && S == 4) {
    // 4-lane path: extract 4 8-bit values from one uint32_t word.
    alignas(16) constexpr uint32_t shifts_[] = {0, 8, 16, 24};
    auto shifts = simd::load<uint32_t, S>(shifts_);
    wi = simd::Simd<uint32_t, S>(*w);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 2 && S == 8) {
    // Highway SIMD: 8 2-bit values from one uint32_t word (half of 16 total)
    // elem_off selects which half: 0=lower 8, 1=upper 8
    uint32_t word = *w;
    if (elem_off == 1) {
      word >>= 16;
    }
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 2, 4, 6, 8, 10, 12, 14}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    wi = simd::Simd<uint32_t, S>(word);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 4 && S == 8) {
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 4, 8, 12, 16, 20, 24, 28}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    wi = simd::Simd<uint32_t, S>(*w);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else if constexpr (bits == 8 && S == 8) {
    // 8 8-bit values from 2 uint32_t words
    // Broadcast w[0] to lanes 0-3, w[1] to lanes 4-7, then shift and mask
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 8, 16, 24, 0, 8, 16, 24}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    alignas(32)
        uint32_t words[8] = {w[0], w[0], w[0], w[0], w[1], w[1], w[1], w[1]};
    wi = simd::load<uint32_t, S>(words);
    wi = wi >> shifts;
    wi = wi & bitmask;
  } else {
    // Appease compiler.. but should never get here
    throw std::runtime_error("Unsupported combination for simd qmm.");
  }
  return wi;
}

// Process NC output columns simultaneously, loading x once and reusing across
// all columns. This amortizes x load cost and improves register utilization.
template <typename T, int bits, int group_size, int NC>
void _qmm_t_simd_multi_col(
    T* result,
    const T* x,
    const uint32_t* w_ptrs[NC],
    const T* scales_ptrs[NC],
    const T* biases_ptrs[NC],
    const float* x_group_sums,
    int K) {
  constexpr int pack_factor = 32 / bits;
  constexpr int S = simd::max_size<float>;
  constexpr int iters_per_word = (S >= pack_factor) ? 1 : (pack_factor / S);
  constexpr int words_per_iter = (S >= pack_factor) ? (S / pack_factor) : 0;

  simd::Simd<float, S> acc[NC];
  for (int c = 0; c < NC; c++)
    acc[c] = simd::Simd<float, S>(0);

  // Scalar accumulator for deferred bias contributions
  float bias_acc[NC] = {};

  auto x_local = x;
  int groups = K / group_size;
  for (int g = 0; g < groups; g++) {
    // Load scale and bias for each column
    float scale_f[NC], bias_f[NC];
    for (int c = 0; c < NC; c++) {
      scale_f[c] = static_cast<float>(*scales_ptrs[c]++);
      bias_f[c] = static_cast<float>(*biases_ptrs[c]++);
    }

    // Per-group accumulator for raw (unscaled) dot product
    simd::Simd<float, S> g_acc[NC];
    for (int c = 0; c < NC; c++)
      g_acc[c] = simd::Simd<float, S>(0);

    int elem = 0;
    {
      // 2xS unrolled loop with extract_bits_simd
      for (; elem + S * 2 <= group_size; elem += S * 2) {
        int elem_off_0 = 0, elem_off_1 = 0;
        if constexpr (S < pack_factor) {
          elem_off_0 = (elem / S) % iters_per_word;
          elem_off_1 = ((elem + S) / S) % iters_per_word;
        }

        simd::Simd<float, S> x_simd_0 = load_as_float<T, S>(x_local);
        simd::Simd<float, S> x_simd_1 = load_as_float<T, S>(x_local + S);
        x_local += S * 2;

#pragma clang loop unroll(full)
        for (int c = 0; c < NC; c++) {
          auto w_raw_0 = simd::Simd<float, S>(
              extract_bits_simd<bits, S>(w_ptrs[c], elem_off_0));
          if constexpr (S >= pack_factor) {
            w_ptrs[c] += words_per_iter;
          } else {
            if (elem_off_0 == iters_per_word - 1) {
              w_ptrs[c] += 1;
            }
          }
          g_acc[c] = simd::fma(x_simd_0, w_raw_0, g_acc[c]);

          auto w_raw_1 = simd::Simd<float, S>(
              extract_bits_simd<bits, S>(w_ptrs[c], elem_off_1));
          if constexpr (S >= pack_factor) {
            w_ptrs[c] += words_per_iter;
          } else {
            if (elem_off_1 == iters_per_word - 1) {
              w_ptrs[c] += 1;
            }
          }
          g_acc[c] = simd::fma(x_simd_1, w_raw_1, g_acc[c]);
        }
      }

      // Handle remaining elements
      for (; elem < group_size; elem += S) {
        int elem_off = 0;
        if constexpr (S < pack_factor) {
          elem_off = (elem / S) % iters_per_word;
        }

        simd::Simd<float, S> x_simd = load_as_float<T, S>(x_local);
        x_local += S;

#pragma clang loop unroll(full)
        for (int c = 0; c < NC; c++) {
          auto w_raw = simd::Simd<float, S>(
              extract_bits_simd<bits, S>(w_ptrs[c], elem_off));
          if constexpr (S >= pack_factor) {
            w_ptrs[c] += words_per_iter;
          } else {
            if (elem_off == iters_per_word - 1) {
              w_ptrs[c] += 1;
            }
          }
          g_acc[c] = simd::fma(x_simd, w_raw, g_acc[c]);
        }
      }
    }

    // Per-group fixup: acc += scale * g_acc, bias_acc += bias * x_group_sum
    float xgs = x_group_sums[g];
#pragma clang loop unroll(full)
    for (int c = 0; c < NC; c++) {
      acc[c] = simd::fma(simd::Simd<float, S>(scale_f[c]), g_acc[c], acc[c]);
      bias_acc[c] += bias_f[c] * xgs;
    }
  }

  for (int c = 0; c < NC; c++) {
    result[c] = T(simd::sum(acc[c]) + bias_acc[c]);
  }
}

// Single column accumulator with bias deferral - helper for tiled version tail
template <typename T, int bits, int group_size>
float _qmm_t_simd_single_col(
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    const float* x_group_sums,
    int K) {
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  constexpr int S = simd::max_size<float>;
  constexpr int iters_per_word = (S >= pack_factor) ? 1 : (pack_factor / S);
  constexpr int words_per_iter = (S >= pack_factor) ? (S / pack_factor) : 0;

  int groups_per_col = K / group_size;

  simd::Simd<float, S> acc(0);
  float bias_acc = 0;
  auto x_local = x;

  for (int g = 0; g < groups_per_col; g++) {
    float scale_f = static_cast<float>(scales[g]);
    float bias_f = static_cast<float>(biases[g]);
    const uint32_t* w_local = w + g * packs_in_group;

    simd::Simd<float, S> g_acc(0);
    for (int elem = 0; elem < group_size; elem += S) {
      int elem_off = 0;
      if constexpr (S < pack_factor) {
        elem_off = (elem / S) % iters_per_word;
      }

      auto w_raw =
          simd::Simd<float, S>(extract_bits_simd<bits, S>(w_local, elem_off));
      if constexpr (S >= pack_factor) {
        w_local += words_per_iter;
      } else {
        if (elem_off == iters_per_word - 1) {
          w_local += 1;
        }
      }

      simd::Simd<float, S> x_simd = load_as_float<T, S>(x_local);
      g_acc = simd::fma(x_simd, w_raw, g_acc);
      x_local += S;
    }

    acc = simd::fma(simd::Simd<float, S>(scale_f), g_acc, acc);
    bias_acc += bias_f * x_group_sums[g];
  }

  return simd::sum(acc) + bias_acc;
}

} // namespace

// Row processor: precomputes x_group_sums for bias deferral, then dispatches
// to multi-col kernel for main body and single-col for tail.
// If preq is non-null, uses pre-quantized activation data instead of
// quantizing locally (eliminates redundant quantization across threads).
template <typename T, int bits, int group_size>
void _qmm_t_simd_row(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int n_start,
    int n_end,
    int K,
    const PreqAct* preq) {
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  constexpr int S = simd::max_size<float>;
  constexpr int NC = 8;

  int groups_per_col = K / group_size;
  int packs_per_col = groups_per_col * packs_in_group;

  // Precompute group sums for x (once per row, reused for all N columns)
  // Stack-allocate for typical sizes, heap for large K
  constexpr int STACK_GROUPS = 128; // covers K up to 8192 with group_size=64
  float xgs_stack[STACK_GROUPS];
  std::unique_ptr<float[]> xgs_heap;
  float* x_group_sums;
  if (preq) {
    // Use pre-quantized group sums (already computed once by caller)
    x_group_sums = const_cast<float*>(preq->x_group_sums);
  } else if (groups_per_col <= STACK_GROUPS) {
    x_group_sums = xgs_stack;
  } else {
    xgs_heap.reset(new float[groups_per_col]);
    x_group_sums = xgs_heap.get();
  }

  // Try ISA-specific int8 dot-product path (no-op stub returns false when
  // unavailable).
  if (try_int8_simd_row<T, bits, group_size>(
          result,
          x,
          w,
          scales,
          biases,
          n_start,
          n_end,
          K,
          groups_per_col,
          x_group_sums,
          preq))
    return;

  // Float path: compute x_group_sums only (no int8 quantization)
  // Skip if using pre-quantized data (group_sums already set at top)
  if (!preq) {
    for (int g = 0; g < groups_per_col; g++) {
      simd::Simd<float, S> sum_acc(0);
      const T* xg = x + g * group_size;
      for (int e = 0; e < group_size; e += S) {
        sum_acc = sum_acc + load_as_float<T, S>(xg + e);
      }
      x_group_sums[g] = simd::sum(sum_acc);
    }
  }

  const uint32_t* w_base = w + n_start * packs_per_col;
  const T* scales_base = scales + n_start * groups_per_col;
  const T* biases_base = biases + n_start * groups_per_col;

  int n = n_start;

  // Process NC columns at a time (8 columns for better x-cache reuse)
  for (; n + NC <= n_end; n += NC) {
    const uint32_t* wp[NC];
    const T* sp[NC];
    const T* bp[NC];
    for (int c = 0; c < NC; c++) {
      wp[c] = w_base + c * packs_per_col;
      sp[c] = scales_base + c * groups_per_col;
      bp[c] = biases_base + c * groups_per_col;
    }
    _qmm_t_simd_multi_col<T, bits, group_size, NC>(
        result + (n - n_start), x, wp, sp, bp, x_group_sums, K);
    w_base += NC * packs_per_col;
    scales_base += NC * groups_per_col;
    biases_base += NC * groups_per_col;
  }

  // Handle remaining columns with NC=4 (if >= 4 remaining)
  constexpr int NC4 = 4;
  for (; n + NC4 <= n_end; n += NC4) {
    const uint32_t* wp[NC4];
    const T* sp[NC4];
    const T* bp[NC4];
    for (int c = 0; c < NC4; c++) {
      wp[c] = w_base + c * packs_per_col;
      sp[c] = scales_base + c * groups_per_col;
      bp[c] = biases_base + c * groups_per_col;
    }
    _qmm_t_simd_multi_col<T, bits, group_size, NC4>(
        result + (n - n_start), x, wp, sp, bp, x_group_sums, K);
    w_base += NC4 * packs_per_col;
    scales_base += NC4 * groups_per_col;
    biases_base += NC4 * groups_per_col;
  }

  // Handle remaining columns one at a time
  for (; n < n_end; n++) {
    result[n - n_start] =
        T(_qmm_t_simd_single_col<T, bits, group_size>(
            x, w_base, scales_base, biases_base, x_group_sums, K));
    w_base += packs_per_col;
    scales_base += groups_per_col;
    biases_base += groups_per_col;
  }
}

uint8_t to_fp8_e8m0_round_up(float x);

namespace {

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
  constexpr int S = simd::max_size<float>;
  static_assert(
      S % pack_factor == 0 || pack_factor % S == 0,
      "SIMD size and pack factor must be divisible");
#if defined(MLX_USE_HIGHWAY_KERNELS)
  if constexpr (bits == 4 || bits == 8) {
    if (qmm_t_int8_highway(
            result,
            x,
            w,
            scales,
            biases,
            quantized_highway_dtype<T>(),
            bits,
            group_size,
            M,
            N,
            K)) {
      return;
    }
  }
#endif
  auto& pool = cpu::ThreadPool::instance();

  // N-parallel threading: split columns across threads.
  // Each thread processes ALL M rows for its column chunk.
  // Weight data per thread fits in L2 cache for much better locality
  // than M-parallel where every thread reads the entire weight matrix.
  //
  // For M=1 generation, use larger minimum columns per thread (128 vs 64).
  // Fewer threads reduces sync overhead and improves L2 cache utilization
  // per thread. For a small model (N=2048): 16 threads x 128 cols vs 32 x 64.
  int min_cols_per_thread = (M == 1) ? 128 : 64;
  int n_threads =
      std::min(pool.max_threads(), std::max(1, N / min_cols_per_thread));

  // Work-stealing chunk size: 64 columns (8 multi-col kernel invocations).
  // Threads start at their assigned chunk and steal more via atomic counter.
  // This dynamically balances load when threads have different cache warmth
  // or the system has asymmetric load from background processes.
  constexpr int CHUNK_COLS = 64;
  int n_chunks = (N + CHUNK_COLS - 1) / CHUNK_COLS;

  // Local steal counter -- must NOT be a pool member because multiple CPU
  // streams can call parallel_for concurrently (e.g., stream_generate uses
  // a separate generation_stream). A shared counter would be corrupted.
  alignas(64) std::atomic<int> steal_counter{0};

  if (n_threads > 1 && M == 1) {
    // M==1 fast path (generation): pre-quantize activation vector ONCE
    // and share across all threads. Eliminates redundant quantization
    // (16 threads each quantizing the same x -> 1 quantization shared).
    constexpr int STACK_GROUPS = 128;
    int groups_per_col = K / group_size;

    // Try ISA-specific int8 pre-quantization path (no-op stub returns false).
    if (try_int8_preq_parallel<T, bits, group_size>(
            result,
            x,
            w,
            scales,
            biases,
            N,
            K,
            n_threads,
            n_chunks,
            steal_counter,
            CHUNK_COLS))
      return;

    // Float fallback: pre-compute group_sums only
    {
      float xgs_buf[STACK_GROUPS];
      std::unique_ptr<float[]> xgs_heap;
      float* x_group_sums = xgs_buf;
      if (groups_per_col > STACK_GROUPS) {
        xgs_heap.reset(new float[groups_per_col]);
        x_group_sums = xgs_heap.get();
      }
      for (int g = 0; g < groups_per_col; g++) {
        simd::Simd<float, S> sum_acc(0);
        const T* xg = x + g * group_size;
        for (int e = 0; e < group_size; e += S) {
          sum_acc = sum_acc + load_as_float<T, S>(xg + e);
        }
        x_group_sums[g] = simd::sum(sum_acc);
      }

      PreqAct preq{nullptr, nullptr, x_group_sums};
      steal_counter.store(n_threads, std::memory_order_relaxed);
      pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
        int my_chunk = tid;
        while (my_chunk < n_chunks) {
          int n_start = std::min(my_chunk * CHUNK_COLS, N);
          int n_end = std::min(n_start + CHUNK_COLS, N);
          if (n_start < n_end) {
            _qmm_t_simd_row<T, bits, group_size>(
                result + n_start,
                x,
                w,
                scales,
                biases,
                n_start,
                n_end,
                K,
                &preq);
          }
          my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
        }
      });
      return;
    }
  }

  if (n_threads > 1) {
    // M>1: each thread processes all M rows for its column chunk.
    // Threads quantize per-row independently (pre-quantization would
    // require M parallel_for calls or large buffers).
    // Work stealing: threads start at their assigned chunk, steal more.
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        int n_start = std::min(my_chunk * CHUNK_COLS, N);
        int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          for (int m = 0; m < M; m++) {
            _qmm_t_simd_row<T, bits, group_size>(
                result + m * N + n_start,
                x + m * K,
                w,
                scales,
                biases,
                n_start,
                n_end,
                K);
          }
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    for (int m = 0; m < M; m++) {
      _qmm_t_simd_row<T, bits, group_size>(
          result + m * N, x + m * K, w, scales, biases, 0, N, K);
    }
  }
}

// Dequantize a row of packed quantized weights to f32.
// W row layout: K/pack_factor uint32 words, each containing pack_factor values.
// Dequant: f32_val = scale * nibble + bias, with scale/bias per group.
template <int bits, int group_size>
void _dequant_row(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int K) {
  constexpr int pack_factor = 32 / bits;
  constexpr int bitmask = (1 << bits) - 1;
  int k = 0;
  for (int g = 0; g < K / group_size; g++) {
    float scale = scales_row[g];
    float bias = biases_row[g];
    for (int j = 0; j < group_size; j++, k++) {
      int word_idx = k / pack_factor;
      int elem_idx = k % pack_factor;
      uint32_t val = (w_row[word_idx] >> (elem_idx * bits)) & bitmask;
      out[k] = scale * static_cast<float>(val) + bias;
    }
  }
}

template <int bits, int group_size>
void _dequant_row_for_blas(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int K) {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  if constexpr (bits == 4 || bits == 8) {
    _dequant_row_highway<bits, group_size>(
        w_row, scales_row, biases_row, out, K);
    return;
  }
#endif
  _dequant_row<bits, group_size>(w_row, scales_row, biases_row, out, K);
}

// Dequantize + BLAS path for large-M quantized matmul (prompt processing).
// For large M, OpenBLAS SGEMM is ~2x faster than custom quantized kernel
// because BLAS has superior micro-kernels and cache tiling.
// The dequantization overhead is <2% of total time for M >= 32.
template <typename T, int bits, int group_size>
void _qmm_t_blas(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int pack_factor = 32 / bits;
  const int w_row_words = K / pack_factor;
  const int num_groups = K / group_size;

  auto& pool = cpu::ThreadPool::instance();

  // The dequantized weight and the f32 result both scale with N, so a
  // 128k-entry vocabulary would ask for ~2 GB of scratch in one go. Walk the
  // output columns in blocks instead: scratch then stays bounded and small
  // enough that cpu::scratch_buffer retains it across calls, rather than
  // thrashing a multi-gigabyte allocation between the layers and the lm_head.
  constexpr size_t kColumnScratchFloats = size_t{32} << 20; // 128 MB
  const size_t floats_per_column = static_cast<size_t>(K) + M;
  const int n_block = static_cast<int>(std::min<size_t>(
      N, std::max<size_t>(kColumnScratchFloats / floats_per_column, 64)));

  // Scratch: x_f32[M*K] + W_f32[n_block*K] + result_f32[M*n_block].
  // For gate_proj (N=8192, K=2048, M=2048) one block covers all of N and the
  // layout matches the untiled path: 16+64+64 = 144MB.
  size_t x_f32_size = static_cast<size_t>(M) * K;
  size_t w_f32_size = static_cast<size_t>(n_block) * K;
  size_t r_f32_size = static_cast<size_t>(M) * n_block;
  float* x_f32 = cpu::scratch_buffer(x_f32_size + w_f32_size + r_f32_size);
  float* w_f32 = x_f32 + x_f32_size;
  float* r_f32 = w_f32 + w_f32_size;

  constexpr int SW = simd::max_size<T>;

  // Convert x to f32 once -- every column block multiplies against it.
  {
    size_t i = 0;
    for (; i + SW <= x_f32_size; i += SW) {
      simd::store(x_f32 + i, simd::Simd<float, SW>(simd::load<T, SW>(x + i)));
    }
    for (; i < x_f32_size; i++) {
      x_f32[i] = static_cast<float>(x[i]);
    }
  }

  std::vector<float> scales_f32(static_cast<size_t>(n_block) * num_groups);
  std::vector<float> biases_f32(static_cast<size_t>(n_block) * num_groups);

  for (int n0 = 0; n0 < N; n0 += n_block) {
    const int nb = std::min(n_block, N - n0);

    // Step 1: Convert this block's scales and biases to f32 (small: nb *
    // K/group_size each), then dequantize its rows of W in parallel. Each
    // thread dequantizes a disjoint range of rows.
    const T* scales_blk = scales + static_cast<size_t>(n0) * num_groups;
    const T* biases_blk = biases + static_cast<size_t>(n0) * num_groups;
    size_t block_groups = static_cast<size_t>(nb) * num_groups;
    for (size_t i = 0; i < block_groups; i++) {
      scales_f32[i] = static_cast<float>(scales_blk[i]);
      biases_f32[i] = static_cast<float>(biases_blk[i]);
    }

    int n_threads = std::min(pool.max_threads(), std::max(1, nb / 64));
    cpu::parallel_for_range(n_threads, nb, [&](size_t begin, size_t end) {
      for (size_t n = begin; n < end; n++) {
        _dequant_row_for_blas<bits, group_size>(
            w + (static_cast<size_t>(n0) + n) * w_row_words,
            scales_f32.data() + n * num_groups,
            biases_f32.data() + n * num_groups,
            w_f32 + n * K,
            K);
      }
    });

    // Step 2: SGEMM -- result block = x_f32 @ W_f32^T
    // Split M across threads, each thread calls single-threaded BLAS
    int m_threads = std::min(pool.max_threads(), std::max(1, M / 8));

    if (m_threads > 1) {
      pool.parallel_for(m_threads, [&](int tid, int nth) {
        size_t m_chunk = (M + nth - 1) / nth;
        size_t m_start = m_chunk * tid;
        size_t m_end = std::min(m_start + m_chunk, (size_t)M);
        if (m_start < m_end) {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasTrans,
              m_end - m_start,
              nb,
              K,
              1.0f,
              x_f32 + m_start * K,
              K,
              w_f32,
              K,
              0.0f,
              r_f32 + m_start * nb,
              nb);
        }
      });
    } else {
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          M,
          nb,
          K,
          1.0f,
          x_f32,
          K,
          w_f32,
          K,
          0.0f,
          r_f32,
          nb);
    }

    // Step 3: Convert this block's result into its output columns.
    for (int m = 0; m < M; m++) {
      const float* src = r_f32 + static_cast<size_t>(m) * nb;
      T* dst = result + static_cast<size_t>(m) * N + n0;
      int j = 0;
      for (; j + SW <= nb; j += SW) {
        simd::store(dst + j, simd::Simd<T, SW>(simd::load<float, SW>(src + j)));
      }
      for (; j < nb; j++) {
        dst[j] = static_cast<T>(src[j]);
      }
    }
  }
}

// Minimum M for the dequant+BLAS path. Below this the custom kernel wins:
// dequantizing to f32 inflates weight streaming 8x (4-bit) so BLAS only pays
// off once the GEMM is deep enough to be compute-bound rather than
// weight-bandwidth-bound. Overridable for tuning via MLX_QMM_BLAS_THRESHOLD.
inline int qmm_blas_m_threshold() {
  static int threshold = []() {
    if (int t = env::get_var("MLX_QMM_BLAS_THRESHOLD", 0); t > 0) {
      return t;
    }
    // Measured on Zen 4 (7950X, N=K=4096, 4-bit): the custom kernel holds
    // ~500-550 GFLOP/s for all M while the dequant+BLAS path only crosses
    // that around M ~= 700 (55 GFLOP/s at M=32, 625 at M=1024, 713 at
    // M=2048). Below the crossover BLAS loses because f32 dequant inflates
    // weight streaming 8x.
    return 768;
  }();
  return threshold;
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
    // For very large M (long prompts), dequantize weights to f32 and use
    // BLAS, whose cache tiling wins once the GEMM is compute-bound.
    if (M >= qmm_blas_m_threshold()) {
      _qmm_t_blas<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
      return;
    }
#if defined(MLX_USE_HIGHWAY_KERNELS)
    if constexpr (bits == 4 || bits == 8) {
      _qmm_t_simd<T, bits, group_size>(result, x, w, scales, biases, M, N, K);
      return;
    }
#endif
    // SIMD path: S must be >= 4 and must divide pack_factor or vice versa
    constexpr int pack_factor = 32 / bits;
    constexpr int S = simd::max_size<float>;
    if constexpr (
        S >= 4 && 32 % bits == 0 &&
        (S % pack_factor == 0 || pack_factor % S == 0) && group_size % S == 0) {
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
  if (x.size() == 0 || out.size() == 0) {
    return;
  }
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
    int K,
    float scale_factor) {
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
        T scale = static_cast<T>(
            static_cast<float>(
                dequantize_scale<T, group_size>(*scales_local++)) *
            scale_factor);
        for (int ng = 0; ng < packs_in_group; ng++) {
          if constexpr (bits == 4) {
            (*result_local++) +=
                xi * scale * static_cast<T>(FP4_LUT[w_local[0] & 0xf]);
            (*result_local++) +=
                xi * scale * static_cast<T>(FP4_LUT[(w_local[0] >> 4) & 0xf]);
          } else {
            (*result_local++) +=
                xi * scale * static_cast<T>(FP8_LUT[w_local[0]]);
          }
          w_local++;
        }
      }
    }
    result += N;
  }
}

// Helper for computing a range of output columns in fp_qmm_t (used by threaded
// version)
template <typename T, int group_size, int bits>
void fp_qmm_t_cols(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int n_start,
    int n_end,
    int K,
    float scale_factor) {
  constexpr int pack_factor = get_pack_factor(bits, 8);
  constexpr int packs_in_group = group_size / pack_factor;

  // Compute strides for navigating to different output columns
  int groups_per_col = K / group_size;
  int bytes_per_col = groups_per_col * packs_in_group;

  // Advance to starting column
  const uint8_t* w_base = (const uint8_t*)w + n_start * bytes_per_col;
  const uint8_t* scales_base = scales + n_start * groups_per_col;

  for (int n = n_start; n < n_end; n++) {
    const uint8_t* w_local = w_base;
    const uint8_t* scales_local = scales_base;
    const T* x_local = x;
    T sum = 0;

    for (int k = 0; k < K; k += group_size) {
      T scale = static_cast<T>(
          static_cast<float>(dequantize_scale<T, group_size>(*scales_local++)) *
          scale_factor);

      T gsum = 0;
      for (int kw = 0; kw < packs_in_group; kw++) {
        if constexpr (bits == 4) {
          gsum += (*x_local++) * static_cast<T>(FP4_LUT[w_local[0] & 0xf]);
          gsum +=
              (*x_local++) * static_cast<T>(FP4_LUT[(w_local[0] >> 4) & 0xf]);
        } else {
          gsum += (*x_local++) * static_cast<T>(FP8_LUT[w_local[0]]);
        }
        w_local++;
      }
      sum += scale * gsum;
    }
    result[n - n_start] = sum;

    // Move to next column
    w_base += bytes_per_col;
    scales_base += groups_per_col;
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
    int K,
    float scale_factor) {
  // Check if parallelization over N is beneficial
  auto& pool = cpu::ThreadPool::instance();
  // Each thread needs enough work to amortize wake-up overhead (~64 columns
  // min)
  int n_threads = std::min(pool.max_threads(), std::max(1, N / 64));

  for (int m = 0; m < M; m++) {
    if (n_threads > 1) {
      // Parallel path: parallelize over output columns N
      pool.parallel_for(n_threads, [&](int tid, int nth) {
        int chunk = (N + nth - 1) / nth;
        int n_start = chunk * tid;
        int n_end = std::min(n_start + chunk, N);
        if (n_start < n_end) {
          fp_qmm_t_cols<T, group_size, bits>(
              result + n_start, x, w, scales, n_start, n_end, K, scale_factor);
        }
      });
      result += N;
    } else {
      // Sequential path (original implementation)
      constexpr int pack_factor = get_pack_factor(bits, 8);
      constexpr int packs_in_group = group_size / pack_factor;

      const uint8_t* w_local = (const uint8_t*)w;
      const uint8_t* scales_local = scales;

      for (int n = 0; n < N; n++) {
        const T* x_local = x;
        T sum = 0;
        for (int k = 0; k < K; k += group_size) {
          T scale = static_cast<T>(
              static_cast<float>(
                  dequantize_scale<T, group_size>(*scales_local++)) *
              scale_factor);

          T gsum = 0;
          for (int kw = 0; kw < packs_in_group; kw++) {
            if constexpr (bits == 4) {
              gsum += (*x_local++) * static_cast<T>(FP4_LUT[w_local[0] & 0xf]);
              gsum += (*x_local++) *
                  static_cast<T>(FP4_LUT[(w_local[0] >> 4) & 0xf]);
            } else {
              gsum += (*x_local++) * static_cast<T>(FP8_LUT[w_local[0]]);
            }
            w_local++;
          }
          sum += scale * gsum;
        }
        *result = sum;
        result++;
      }
    }
    x += K;
  }
}

template <int group_size, int bits>
void fp_dequant_rows_for_blas(
    const uint32_t* w,
    const uint8_t* scales,
    float* out,
    int row_start,
    int row_end,
    int K,
    float scale_factor) {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  if constexpr (bits == 4 || bits == 8) {
    fp_dequant_rows_highway(
        w,
        scales,
        out,
        bits,
        group_size,
        row_start,
        row_end,
        K,
        scale_factor,
        FP4_LUT,
        FP8_LUT);
    return;
  }
#endif

  constexpr int pack_factor = get_pack_factor(bits, 32);
  const int groups_per_row = K / group_size;
  const int packs_per_row = groups_per_row * (group_size / pack_factor);

  for (int row = row_start; row < row_end; ++row) {
    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(
        w + static_cast<size_t>(row) * packs_per_row);
    const uint8_t* scales_row =
        scales + static_cast<size_t>(row) * groups_per_row;
    float* out_row = out + static_cast<size_t>(row - row_start) * K;

    for (int g = 0; g < groups_per_row; ++g) {
      const float scale =
          static_cast<float>(
              dequantize_scale<float, group_size>(scales_row[g])) *
          scale_factor;
      const uint8_t* w_group =
          w_row + static_cast<size_t>(g) * (group_size * bits / 8);
      float* out_group = out_row + static_cast<size_t>(g) * group_size;

      if constexpr (bits == 4) {
        for (int k = 0; k < group_size; k += 2) {
          const uint8_t packed = w_group[k / 2];
          out_group[k] = scale * FP4_LUT[packed & 0x0F];
          out_group[k + 1] = scale * FP4_LUT[(packed >> 4) & 0x0F];
        }
      } else {
        for (int k = 0; k < group_size; ++k) {
          out_group[k] = scale * FP8_LUT[w_group[k]];
        }
      }
    }
  }
}

// Prompt path for fp-quantized weights. Direct packed kernels avoid
// materializing weights and are best for decode/few-row shapes; long prompts
// are faster when we dequantize each weight block once and let BLAS tile the
// dense GEMM. Overridable for tuning via MLX_FP_QMM_BLAS_THRESHOLD.
inline int fp_qmm_blas_m_threshold() {
  static int threshold = env::get_var("MLX_FP_QMM_BLAS_THRESHOLD", 512);
  return threshold;
}

inline bool fp_qmm_blas_n_split_enabled() {
  static bool value = env::get_var("MLX_FP_QMM_BLAS_N_SPLIT", 1) != 0;
  return value;
}

inline int fp_qmm_blas_n_split_cols() {
  static int cols =
      std::clamp(env::get_var("MLX_FP_QMM_BLAS_N_SPLIT_COLS", 512), 64, 4096);
  return cols;
}

template <typename T, int group_size, int bits>
void fp_qmm_t_blas(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K,
    float scale_factor) {
  auto& pool = cpu::ThreadPool::instance();

  const bool use_n_split = fp_qmm_blas_n_split_enabled() &&
      pool.max_threads() > 1 && M >= 128 && N >= 1024;
  const size_t x_f32_size = static_cast<size_t>(M) * K;
  float* x_f32 = nullptr;
  float* w_f32 = nullptr;
  float* r_f32 = nullptr;
  int n_block = 0;
  if (use_n_split) {
    x_f32 = cpu::scratch_buffer(x_f32_size);
  } else {
    constexpr size_t kColumnScratchFloats = size_t{32} << 20; // 128 MB
    const size_t floats_per_column = static_cast<size_t>(K) + M;
    n_block = static_cast<int>(std::min<size_t>(
        N, std::max<size_t>(kColumnScratchFloats / floats_per_column, 64)));
    const size_t w_f32_size = static_cast<size_t>(n_block) * K;
    const size_t r_f32_size = static_cast<size_t>(M) * n_block;
    x_f32 = cpu::scratch_buffer(x_f32_size + w_f32_size + r_f32_size);
    w_f32 = x_f32 + x_f32_size;
    r_f32 = w_f32 + w_f32_size;
  }

  constexpr int SW = simd::max_size<T>;
  size_t i = 0;
  for (; i + SW <= x_f32_size; i += SW) {
    simd::store(x_f32 + i, simd::Simd<float, SW>(simd::load<T, SW>(x + i)));
  }
  for (; i < x_f32_size; ++i) {
    x_f32[i] = static_cast<float>(x[i]);
  }
  if (use_n_split) {
    // Balance chunk count across threads to avoid straggler rounds.
    const int max_threads = pool.max_threads();
    const int chunk_target =
        std::min(N, ((fp_qmm_blas_n_split_cols() + 63) / 64) * 64);
    const int rounds = std::max(
        1, (N + max_threads * chunk_target - 1) / (max_threads * chunk_target));
    int chunk_cols = (N + max_threads * rounds - 1) / (max_threads * rounds);
    chunk_cols = std::min(N, ((chunk_cols + 63) / 64) * 64);
    const int n_chunks = (N + chunk_cols - 1) / chunk_cols;
    const int n_threads = std::min(max_threads, n_chunks);
    std::atomic<int> steal_counter{n_threads};

    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int chunk = tid;
      while (chunk < n_chunks) {
        const int n0 = chunk * chunk_cols;
        const int nb = std::min(chunk_cols, N - n0);
        const size_t local_w_f32_size = static_cast<size_t>(nb) * K;
        const size_t local_r_f32_size = static_cast<size_t>(M) * nb;
        float* local_w_f32 =
            cpu::scratch_buffer(local_w_f32_size + local_r_f32_size, 1);
        float* local_r_f32 = local_w_f32 + local_w_f32_size;

        fp_dequant_rows_for_blas<group_size, bits>(
            w, scales, local_w_f32, n0, n0 + nb, K, scale_factor);

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M,
            nb,
            K,
            1.0f,
            x_f32,
            K,
            local_w_f32,
            K,
            0.0f,
            local_r_f32,
            nb);
        for (int m = 0; m < M; ++m) {
          const float* src = local_r_f32 + static_cast<size_t>(m) * nb;
          T* dst = result + static_cast<size_t>(m) * N + n0;
          int j = 0;
          for (; j + SW <= nb; j += SW) {
            simd::store(
                dst + j, simd::Simd<T, SW>(simd::load<float, SW>(src + j)));
          }
          for (; j < nb; ++j) {
            dst[j] = static_cast<T>(src[j]);
          }
        }
        chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
    return;
  }

  for (int n0 = 0; n0 < N; n0 += n_block) {
    const int nb = std::min(n_block, N - n0);

    int n_threads = std::min(pool.max_threads(), std::max(1, nb / 64));
    cpu::parallel_for_range(n_threads, nb, [&](size_t begin, size_t end) {
      fp_dequant_rows_for_blas<group_size, bits>(
          w,
          scales,
          w_f32 + begin * K,
          n0 + static_cast<int>(begin),
          n0 + static_cast<int>(end),
          K,
          scale_factor);
    });
    int m_threads = std::min(pool.max_threads(), std::max(1, M / 8));
    if (m_threads > 1) {
      pool.parallel_for(m_threads, [&](int tid, int nth) {
        size_t m_chunk = (M + nth - 1) / nth;
        size_t m_start = m_chunk * tid;
        size_t m_end = std::min(m_start + m_chunk, static_cast<size_t>(M));
        if (m_start < m_end) {
          cblas_sgemm(
              CblasRowMajor,
              CblasNoTrans,
              CblasTrans,
              m_end - m_start,
              nb,
              K,
              1.0f,
              x_f32 + m_start * K,
              K,
              w_f32,
              K,
              0.0f,
              r_f32 + m_start * nb,
              nb);
        }
      });
    } else {
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          M,
          nb,
          K,
          1.0f,
          x_f32,
          K,
          w_f32,
          K,
          0.0f,
          r_f32,
          nb);
    }
    for (int m = 0; m < M; ++m) {
      const float* src = r_f32 + static_cast<size_t>(m) * nb;
      T* dst = result + static_cast<size_t>(m) * N + n0;
      int j = 0;
      for (; j + SW <= nb; j += SW) {
        simd::store(dst + j, simd::Simd<T, SW>(simd::load<float, SW>(src + j)));
      }
      for (; j < nb; ++j) {
        dst[j] = static_cast<T>(src[j]);
      }
    }
  }
}

#if !defined(MLX_USE_HIGHWAY_KERNELS)
template <int S, int bits>
simd::Simd<float, S> fp_extract_bits_simd(const uint32_t* w, int elem_off = 0) {
  if constexpr (S == 1 && bits == 4) {
    // Scalar: extract single 4-bit FP4 value
    return simd::Simd<float, 1>(FP4_LUT[(*w >> (elem_off * 4)) & 0xf]);
  } else if constexpr (S == 1 && bits == 8) {
    // Scalar: extract single 8-bit FP8 value via LUT
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(w);
    return simd::Simd<float, 1>(FP8_LUT[bytes[elem_off]]);
  } else if constexpr (S == 4 && bits == 4) {
    // 4-lane path: extract 4 FP4 values from part of one uint32_t word.
    uint32_t word = *w;
    if (elem_off == 1) {
      word >>= 16;
    }
    alignas(16) constexpr uint32_t shifts_[] = {0, 4, 8, 12};
    auto shifts = simd::load<uint32_t, S>(shifts_);
    auto wi = simd::Simd<uint32_t, S>(word);
    wi = wi >> shifts;
    wi = wi & 0xf;
    // Scalar LUT lookup for this fallback path.
    alignas(16) float tmp[S];
    for (int i = 0; i < S; ++i) {
      tmp[i] = FP4_LUT[wi[i]];
    }
    return simd::load<float, S>(tmp);
  } else if constexpr (S == 4 && bits == 8) {
    // 4-lane path: load 4 FP8 values from one uint32_t word via LUT.
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(w);
    alignas(16) float tmp[S];
    for (int i = 0; i < S; ++i) {
      tmp[i] = FP8_LUT[bytes[i]];
    }
    return simd::load<float, S>(tmp);
  } else if constexpr (S == 8 && bits == 4) {
    // Extract 8 4-bit indices, then SIMD LUT lookup
    constexpr std::array<uint32_t, 8> shifts_ = {{0, 4, 8, 12, 16, 20, 24, 28}};
    auto shifts(*(simd::Simd<uint32_t, S>*)&shifts_);
    auto wi = simd::Simd<uint32_t, S>(*w);
    wi = wi >> shifts;
    wi = wi & 0xf;
    return fp4_lut_lookup_simd<S>(wi, FP4_LUT);
  } else if constexpr (S == 8 && bits == 8) {
    // 8 FP8 values via SIMD LUT gather
    return fp8_lut_gather_simd<S>(w, FP8_LUT);
  } else {
    // Appease compiler.. but should never get here
    throw std::runtime_error("Unsupported combination for simd qmm.");
  }
}

// Process NC output columns simultaneously for FP quantized matmul.
// Loads x once per SIMD chunk and reuses across all columns.
template <typename T, int group_size, int bits, int NC>
void fp_qmm_t_simd_multi_col(
    T* result,
    const T* x,
    const uint32_t* w_ptrs[NC],
    const uint8_t* scales_ptrs[NC],
    int K,
    float scale_factor) {
  constexpr int pack_factor = get_pack_factor(bits, 32);
  constexpr int S = simd::max_size<float>;
  constexpr int iters_per_word = (S >= pack_factor) ? 1 : (pack_factor / S);
  constexpr int words_per_iter = (S >= pack_factor) ? (S / pack_factor) : 0;

  simd::Simd<float, S> acc[NC];
  for (int c = 0; c < NC; c++)
    acc[c] = simd::Simd<float, S>(0);

  auto x_local = x;
  for (int k = 0; k < K; k += group_size) {
    // Load and dequantize scale for each column
    float scale_f[NC];
    for (int c = 0; c < NC; c++) {
      scale_f[c] = static_cast<float>(
                       dequantize_scale<T, group_size>(*scales_ptrs[c]++)) *
          scale_factor;
    }

    simd::Simd<float, S> g_acc[NC];
    for (int c = 0; c < NC; c++)
      g_acc[c] = simd::Simd<float, S>(0);

    for (int elem = 0; elem < group_size; elem += S) {
      // Load x once, reuse for all NC columns
      simd::Simd<float, S> x_simd = load_as_float<T, S>(x_local);
      x_local += S;

      int elem_off = 0;
      if constexpr (S < pack_factor) {
        elem_off = (elem / S) % iters_per_word;
      }

      // Process each column with the same x_simd
#pragma clang loop unroll(full)
      for (int c = 0; c < NC; c++) {
        auto wf = fp_extract_bits_simd<S, bits>(w_ptrs[c], elem_off);
        if constexpr (S >= pack_factor) {
          w_ptrs[c] += words_per_iter;
        } else {
          if (elem_off == iters_per_word - 1) {
            w_ptrs[c] += 1;
          }
        }
        g_acc[c] = simd::fma(x_simd, wf, g_acc[c]);
      }
    }

    // Accumulate group results: acc += scale * g_acc
#pragma clang loop unroll(full)
    for (int c = 0; c < NC; c++) {
      acc[c] = simd::fma(simd::Simd<float, S>(scale_f[c]), g_acc[c], acc[c]);
    }
  }

  for (int c = 0; c < NC; c++) {
    result[c] = T(simd::sum(acc[c]));
  }
}
#endif

// Helper for FP quantized matmul row processing (used by parallel version).
// Processes multiple output columns, using multi-column kernel where possible.
template <typename T, int group_size, int bits>
void fp_qmm_t_simd_row(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int n_start,
    int n_end,
    int K,
    float scale_factor) {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  fp_qmm_t_highway_row(
      result,
      x,
      w,
      scales,
      quantized_highway_dtype<T>(),
      bits,
      group_size,
      n_start,
      n_end,
      K,
      scale_factor,
      FP4_LUT,
      FP8_LUT);
  return;
#else

  constexpr int pack_factor = get_pack_factor(bits, 32);
  constexpr int packs_in_group = group_size / pack_factor;
  constexpr int S = simd::max_size<float>;
  constexpr int iters_per_word = (S >= pack_factor) ? 1 : (pack_factor / S);
  constexpr int words_per_iter = (S >= pack_factor) ? (S / pack_factor) : 0;
  constexpr int NC = 4; // Process 4 columns at once

  // Compute stride per output column
  int groups_per_col = K / group_size;
  int packs_per_col = groups_per_col * packs_in_group;

  // Advance pointers to starting column
  const uint32_t* w_base = w + n_start * packs_per_col;
  const uint8_t* scales_base = scales + n_start * groups_per_col;

  int n = n_start;

  // Process NC columns at a time
  for (; n + NC <= n_end; n += NC) {
    const uint32_t* wp[NC];
    const uint8_t* sp[NC];
    for (int c = 0; c < NC; c++) {
      wp[c] = w_base + c * packs_per_col;
      sp[c] = scales_base + c * groups_per_col;
    }
    fp_qmm_t_simd_multi_col<T, group_size, bits, NC>(
        result + (n - n_start), x, wp, sp, K, scale_factor);
    w_base += NC * packs_per_col;
    scales_base += NC * groups_per_col;
  }

  // Handle remaining columns one at a time
  for (; n < n_end; n++) {
    const uint32_t* w_local = w_base;
    const uint8_t* scales_local = scales_base;

    simd::Simd<float, S> acc(0);
    auto x_local = x;
    for (int k = 0; k < K; k += group_size) {
      T scale = static_cast<T>(
          static_cast<float>(dequantize_scale<T, group_size>(*scales_local++)) *
          scale_factor);

      simd::Simd<float, S> g_acc(0);
      for (int elem = 0; elem < group_size; elem += S) {
        int elem_off = 0;
        if constexpr (S < pack_factor) {
          elem_off = (elem / S) % iters_per_word;
        }
        auto wf = fp_extract_bits_simd<S, bits>(w_local, elem_off);
        if constexpr (S >= pack_factor) {
          w_local += words_per_iter;
        } else {
          if (elem_off == iters_per_word - 1) {
            w_local += 1;
          }
        }
        simd::Simd<float, S> x_simd = load_as_float<T, S>(x_local);
        g_acc = simd::fma(x_simd, wf, g_acc);
        x_local += S;
      }
      acc = simd::fma(
          simd::Simd<float, S>(static_cast<float>(scale)), g_acc, acc);
    }

    result[n - n_start] = T(simd::sum(acc));

    // Move to next column
    w_base += packs_per_col;
    scales_base += groups_per_col;
  }
#endif
}

template <typename T, int group_size, int bits>
void fp_qmm_t_simd(
    T* result,
    const T* x,
    const uint32_t* w,
    const uint8_t* scales,
    int M,
    int N,
    int K,
    float scale_factor) {
#if defined(MLX_USE_HIGHWAY_KERNELS)
  fp_qmm_t_highway(
      result,
      x,
      w,
      scales,
      quantized_highway_dtype<T>(),
      bits,
      group_size,
      M,
      N,
      K,
      scale_factor,
      FP4_LUT,
      FP8_LUT);
  return;
#else
  constexpr int pack_factor = get_pack_factor(bits, 32);
  constexpr int S = simd::max_size<float>;
  static_assert(
      S % pack_factor == 0 || pack_factor % S == 0,
      "SIMD size and pack factor must be divisible");
  constexpr int iters_per_word = (S >= pack_factor) ? 1 : (pack_factor / S);
  constexpr int words_per_iter = (S >= pack_factor) ? (S / pack_factor) : 0;

  auto& pool = cpu::ThreadPool::instance();

  // N-parallel threading: split columns across threads.
  // Each thread processes ALL M rows for its column chunk.
  // Weight data per thread fits in L2 cache for much better locality.
  int min_cols_per_thread = (M == 1) ? 128 : 64;
  int n_threads =
      std::min(pool.max_threads(), std::max(1, N / min_cols_per_thread));

  if (n_threads > 1) {
    // Work stealing: 64-column chunks for dynamic load balancing.
    // Local counter -- safe for concurrent parallel_for from multiple streams.
    constexpr int CHUNK_COLS = 64;
    int n_chunks = (N + CHUNK_COLS - 1) / CHUNK_COLS;
    alignas(64) std::atomic<int> steal_counter{0};
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        int n_start = std::min(my_chunk * CHUNK_COLS, N);
        int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          for (int m = 0; m < M; m++) {
            fp_qmm_t_simd_row<T, group_size, bits>(
                result + m * N + n_start,
                x + m * K,
                w,
                scales,
                n_start,
                n_end,
                K,
                scale_factor);
          }
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
  } else {
    for (int m = 0; m < M; m++) {
      fp_qmm_t_simd_row<T, group_size, bits>(
          result + m * N, x + m * K, w, scales, 0, N, K, scale_factor);
    }
  }
#endif
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
    bool transposed_w,
    float scale_factor) {
  if (transposed_w) {
    const int blas_threshold = fp_qmm_blas_m_threshold();
    if (blas_threshold > 0 && M >= blas_threshold) {
      fp_qmm_t_blas<T, group_size, bits>(
          result, x, w, scales, M, N, K, scale_factor);
      return;
    }
#if defined(MLX_USE_HIGHWAY_KERNELS)
    if constexpr (bits == 4 || bits == 8) {
      fp_qmm_t_simd<T, group_size, bits>(
          result, x, w, scales, M, N, K, scale_factor);
      return;
    }
#endif
    // SIMD path: S must be >= 4 and must divide pack_factor or vice versa
    constexpr int fp_pack_factor = get_pack_factor(bits, 32);
    constexpr int S = simd::max_size<float>;
    if constexpr (
        S >= 4 && (S % fp_pack_factor == 0 || fp_pack_factor % S == 0) &&
        group_size % S == 0) {
      fp_qmm_t_simd<T, group_size, bits>(
          result, x, w, scales, M, N, K, scale_factor);
    } else {
      fp_qmm_t<T, group_size, bits>(
          result, x, w, scales, M, N, K, scale_factor);
    }
  } else {
    fp_qmm<T, group_size, bits>(result, x, w, scales, M, N, K, scale_factor);
  }
}

template <typename T, int group_size, int bits>
void fp_qmm_dispatch_mode(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    bool transposed_w,
    float scale_factor) {
  if (x.size() == 0 || out.size() == 0) {
    return;
  }
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
        transposed_w,
        scale_factor);
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
    bool transposed_w,
    float scale_factor) {
  if (bits == 8) {
    fp_qmm_dispatch_mode<T, 32, 8>(
        out, x, w, scales, transposed_w, scale_factor);
  } else if (group_size == 32) {
    fp_qmm_dispatch_mode<T, 32, 4>(
        out, x, w, scales, transposed_w, scale_factor);
  } else {
    fp_qmm_dispatch_mode<T, 16, 4>(
        out, x, w, scales, transposed_w, scale_factor);
  }
}

void fp_qmm_dispatch(
    array& out,
    const array& x,
    const array& w,
    const array& scales,
    int group_size,
    int bits,
    bool transposed_w,
    float scale_factor = 1.0f) {
  switch (x.dtype()) {
    case bfloat16:
      fp_qmm_dispatch_typed<bfloat16_t>(
          out, x, w, scales, group_size, bits, transposed_w, scale_factor);
      break;
    case float16:
      fp_qmm_dispatch_typed<float16_t>(
          out, x, w, scales, group_size, bits, transposed_w, scale_factor);
      break;
    case float32:
      fp_qmm_dispatch_typed<float>(
          out, x, w, scales, group_size, bits, transposed_w, scale_factor);
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

  // Precompute offsets for batching
  int total = lhs_indices.size();
  std::vector<size_t> x_offsets(total);
  std::vector<int> w_idxs(total);
  for (int i = 0; i < total; i++) {
    int x_idx = lhs_indices_ptr[elem_to_loc(
        i, lhs_indices.shape(), lhs_indices.strides())];
    w_idxs[i] = rhs_indices_ptr[elem_to_loc(
        i, rhs_indices.shape(), rhs_indices.strides())];
    x_offsets[i] = elem_to_loc(x_idx * M * K, x.shape(), x.strides());
  }

  // Batch consecutive same-expert entries with contiguous x data
  int i = 0;
  while (i < total) {
    int batch_count = 1;
    size_t x_stride = static_cast<size_t>(M) * K;
    while (i + batch_count < total && w_idxs[i + batch_count] == w_idxs[i] &&
           x_offsets[i + batch_count] ==
               x_offsets[i] + batch_count * x_stride) {
      batch_count++;
    }
    int w_idx = w_idxs[i];
    _qmm_dispatch_typed<T>(
        out_ptr + i * M * N,
        x_ptr + x_offsets[i],
        w_ptr + elem_to_loc(w_idx * w_els, w.shape(), w.strides()),
        scales_ptr +
            elem_to_loc(w_idx * g_els, scales.shape(), scales.strides()),
        biases_ptr +
            elem_to_loc(w_idx * g_els, biases.shape(), biases.strides()),
        M * batch_count,
        N,
        K,
        bits,
        group_size,
        transposed_w);
    i += batch_count;
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
    bool transposed_w,
    float scale_factor) {
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

  int total = lhs_indices.size();
  // Precompute all offsets
  std::vector<size_t> x_offsets(total);
  std::vector<size_t> w_offsets(total);
  std::vector<size_t> s_offsets(total);
  for (int i = 0; i < total; i++) {
    int x_idx = lhs_indices_ptr[elem_to_loc(
        i, lhs_indices.shape(), lhs_indices.strides())];
    int w_idx = rhs_indices_ptr[elem_to_loc(
        i, rhs_indices.shape(), rhs_indices.strides())];
    x_offsets[i] = elem_to_loc(x_idx * M * K, x.shape(), x.strides());
    w_offsets[i] = elem_to_loc(w_idx * w_els, w.shape(), w.strides());
    s_offsets[i] = elem_to_loc(w_idx * g_els, scales.shape(), scales.strides());
  }

  // For MoE dispatch (many entries, M=1, transposed weights with SIMD support):
  // Use outer-level parallelism -- distribute all entries across threads with
  // a single parallel_for, instead of running nested parallel_for per entry.
  constexpr int fp_pack_factor = get_pack_factor(bits, 32);
  constexpr int S = simd::max_size<float>;
  auto& pool = cpu::ThreadPool::instance();

  if (total >= 4 && total <= pool.max_threads() && M == 1 && transposed_w &&
      S >= 4 && (S % fp_pack_factor == 0 || fp_pack_factor % S == 0) &&
      group_size % S == 0 && N >= 1024) {
    constexpr int CHUNK_COLS = 64;
    int chunks_per_entry = (N + CHUNK_COLS - 1) / CHUNK_COLS;
    int total_chunks = total * chunks_per_entry;
    int n_threads = std::min(pool.max_threads(), total_chunks);
    alignas(64) std::atomic<int> steal_counter{0};
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int chunk = tid;
      while (chunk < total_chunks) {
        int j = chunk / chunks_per_entry;
        int n_start = (chunk - j * chunks_per_entry) * CHUNK_COLS;
        int n_end = std::min(n_start + CHUNK_COLS, N);
        fp_qmm_t_simd_row<T, group_size, bits>(
            out_ptr + j * N + n_start,
            x_ptr + x_offsets[j],
            w_ptr + w_offsets[j],
            scales_ptr + s_offsets[j],
            n_start,
            n_end,
            K,
            scale_factor);
        chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
    return;
  }
  if (total > pool.max_threads() && M == 1 && transposed_w && S >= 4 &&
      (S % fp_pack_factor == 0 || fp_pack_factor % S == 0) &&
      group_size % S == 0) {
    int n_threads = std::min(pool.max_threads(), total);
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      int chunk = (total + nth - 1) / nth;
      int start = chunk * tid;
      int end = std::min(start + chunk, total);
      for (int j = start; j < end; j++) {
        fp_qmm_t_simd_row<T, group_size, bits>(
            out_ptr + j * N,
            x_ptr + x_offsets[j],
            w_ptr + w_offsets[j],
            scales_ptr + s_offsets[j],
            0,
            N,
            K,
            scale_factor);
      }
    });
    return;
  }

  // Fallback: sequential with batching for contiguous same-expert entries
  for (int i = 0; i < total;) {
    int batch_count = 1;
    size_t x_stride = static_cast<size_t>(M) * K;
    while (
        i + batch_count < total && w_offsets[i + batch_count] == w_offsets[i] &&
        s_offsets[i + batch_count] == s_offsets[i] &&
        x_offsets[i + batch_count] == x_offsets[i] + batch_count * x_stride) {
      batch_count++;
    }
    fp_qmm_dispatch_transpose<T, group_size, bits>(
        out_ptr + i * M * N,
        x_ptr + x_offsets[i],
        w_ptr + w_offsets[i],
        scales_ptr + s_offsets[i],
        M * batch_count,
        N,
        K,
        transposed_w,
        scale_factor);
    i += batch_count;
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
    bool transposed_w,
    float scale_factor) {
  if (bits == 8) {
    fp_bs_qmm_dispatch_mode<T, 32, 8>(
        out,
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        transposed_w,
        scale_factor);
  } else if (group_size == 32) {
    fp_bs_qmm_dispatch_mode<T, 32, 4>(
        out,
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        transposed_w,
        scale_factor);
  } else {
    fp_bs_qmm_dispatch_mode<T, 16, 4>(
        out,
        x,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        transposed_w,
        scale_factor);
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
    bool transposed_w,
    float scale_factor = 1.0f) {
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
          transposed_w,
          scale_factor);
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
          transposed_w,
          scale_factor);
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
          transposed_w,
          scale_factor);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

// Quantize then dequantize x in-place, simulating quantization noise.
// For nvfp4 (group_size=16, bits=4): scale via fp8_e4m3, values via FP4 LUT.
// For mxfp8 (group_size=32, bits=8): scale via fp8_e8m0 (exponent-only), values
// via fp8_e4m3.
template <typename T>
void quantize_dequantize_fp_groups(
    T* data,
    int g_start,
    int g_end,
    int group_size,
    int bits,
    QuantizationMode mode,
    float global_scale) {
  detail::ToFP8 to_fp8;
  detail::FromFP8 from_fp8;

  float maxval = (bits == 4) ? 6.0f : 448.0f;

  for (int g = g_start; g < g_end; g++) {
    T* gp = data + static_cast<size_t>(g) * group_size;

    // Per-group absolute max
    float amax = 0;
    for (int i = 0; i < group_size; i++) {
      amax = std::max(amax, std::abs(static_cast<float>(gp[i])));
    }
    float raw_scale = amax / maxval;

    float scale;
    if (mode == QuantizationMode::Nvfp4) {
      // fp8_e4m3 scale
      float scale_encode =
          global_scale > 0.0f ? (448.0f * 6.0f) / global_scale : 1.0f;
      uint8_t scale_fp8 = to_fp8(raw_scale * scale_encode);
      scale = from_fp8(scale_fp8) / scale_encode;
    } else {
      // MX modes use the same upward-rounded E8M0 scale as quantize().
      if (raw_scale == 0.0f) {
        scale = 1.0f; // 2^0 when all values are zero
      } else {
        scale = dequantize_scale<float, 32>(to_fp8_e8m0_round_up(raw_scale));
      }
    }

    if (scale == 0.0f) {
      for (int i = 0; i < group_size; i++) {
        gp[i] = static_cast<T>(0.0f);
      }
      continue;
    }

    float inv_scale = 1.0f / scale;

    if (bits == 4) {
      // Round to nearest FP4 LUT value, then multiply by scale
      for (int i = 0; i < group_size; i++) {
        float normalized = static_cast<float>(gp[i]) * inv_scale;
        int best_idx = 0;
        float best_dist = std::abs(normalized - FP4_LUT[0]);
        for (int j = 1; j < 16; j++) {
          float dist = std::abs(normalized - FP4_LUT[j]);
          if (dist < best_dist) {
            best_dist = dist;
            best_idx = j;
          }
        }
        gp[i] = static_cast<T>(FP4_LUT[best_idx] * scale);
      }
    } else {
      // Round-trip through fp8_e4m3
      for (int i = 0; i < group_size; i++) {
        float normalized = static_cast<float>(gp[i]) * inv_scale;
        uint8_t fp8 = to_fp8(normalized);
        gp[i] = static_cast<T>(from_fp8(fp8) * scale);
      }
    }
  }
}

template <typename T>
void quantize_dequantize_fp(
    T* data,
    size_t size,
    int group_size,
    int bits,
    QuantizationMode mode,
    float global_scale) {
  int n_groups = size / group_size;
  auto& pool = cpu::ThreadPool::instance();
  int n_threads = cpu::effective_threads(size, pool.max_threads());
  n_threads = std::min(n_threads, n_groups);
  if (n_threads > 1) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      int chunk = (n_groups + nth - 1) / nth;
      int g_start = chunk * tid;
      int g_end = std::min(g_start + chunk, n_groups);
      if (g_start < g_end) {
        quantize_dequantize_fp_groups(
            data, g_start, g_end, group_size, bits, mode, global_scale);
      }
    });
  } else {
    quantize_dequantize_fp_groups(
        data, 0, n_groups, group_size, bits, mode, global_scale);
  }
}

void quantize_dequantize_fp_dispatch(
    array& x,
    int group_size,
    int bits,
    QuantizationMode mode,
    float global_scale = 0.0f) {
  switch (x.dtype()) {
    case float32:
      quantize_dequantize_fp(
          x.data<float>(), x.size(), group_size, bits, mode, global_scale);
      break;
    case float16:
      quantize_dequantize_fp(
          x.data<float16_t>(), x.size(), group_size, bits, mode, global_scale);
      break;
    case bfloat16:
      quantize_dequantize_fp(
          x.data<bfloat16_t>(), x.size(), group_size, bits, mode, global_scale);
      break;
    default:
      throw std::runtime_error(
          "[QQMatmul] Only floating point types are supported");
  }
}

// Quantize float weights into the packed fp-quant layout consumed by the
// fp_qmm kernels: per output row (of length K), K/group_size scale bytes
// (fp8_e4m3 for nvfp4, fp8_e8m0 for mx modes) and K packed values (two FP4
// LUT indices per byte, low nibble first, for bits=4; one fp8_e4m3 byte per
// value for bits=8). The scale byte is decoded back before normalizing so
// the packed values are exactly consistent with what the matmul kernels
// reconstruct. Scalar per element but parallelized over rows: this path only
// runs when qqmm/gather_qqmm is called with unquantized weights (QAT-style
// use); inference passes pre-quantized weights and never reaches it.
template <typename T, int group_size, int bits>
void fp_quantize_w_rows(
    const T* w,
    uint8_t* w_q,
    uint8_t* scales,
    size_t row_start,
    size_t row_end,
    int K,
    float scale_encode) {
  detail::ToFP8 to_fp8;
  detail::FromFP8 from_fp8;
  constexpr float maxval = (bits == 4) ? 6.0f : 448.0f;
  constexpr int bytes_per_group = (bits == 4) ? group_size / 2 : group_size;
  const int groups_per_row = K / group_size;
  const size_t bytes_per_row =
      static_cast<size_t>(groups_per_row) * bytes_per_group;

  for (size_t r = row_start; r < row_end; r++) {
    const T* wr = w + r * K;
    uint8_t* qr = w_q + r * bytes_per_row;
    uint8_t* sr = scales + r * groups_per_row;

    for (int g = 0; g < groups_per_row; g++) {
      const T* gp = wr + g * group_size;

      float amax = 0;
      for (int i = 0; i < group_size; i++) {
        amax = std::max(amax, std::abs(static_cast<float>(gp[i])));
      }
      float raw_scale = amax / maxval;

      uint8_t scale_byte;
      float scale;
      if constexpr (group_size == 16) {
        // nvfp4: fp8_e4m3 scale (optionally global-scaled)
        scale_byte = to_fp8(raw_scale * scale_encode);
        scale = from_fp8(scale_byte) / scale_encode;
      } else {
        // mx modes: fp8_e8m0 scale (pure exponent)
        if (raw_scale == 0.0f) {
          scale_byte = 127; // 2^0
          scale = 1.0f;
        } else {
          scale_byte = to_fp8_e8m0_round_up(raw_scale);
          scale = dequantize_scale<float, 32>(scale_byte);
        }
      }
      sr[g] = scale_byte;

      float inv_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
      uint8_t* qg = qr + static_cast<size_t>(g) * bytes_per_group;
      if constexpr (bits == 4) {
        for (int i = 0; i < group_size; i += 2) {
          uint8_t nib[2];
          for (int j = 0; j < 2; j++) {
            float normalized = static_cast<float>(gp[i + j]) * inv_scale;
            int best_idx = 0;
            float best_dist = std::abs(normalized - FP4_LUT[0]);
            for (int c = 1; c < 16; c++) {
              float dist = std::abs(normalized - FP4_LUT[c]);
              if (dist < best_dist) {
                best_dist = dist;
                best_idx = c;
              }
            }
            nib[j] = static_cast<uint8_t>(best_idx);
          }
          qg[i / 2] = static_cast<uint8_t>(nib[0] | (nib[1] << 4));
        }
      } else {
        for (int i = 0; i < group_size; i++) {
          float normalized = static_cast<float>(gp[i]) * inv_scale;
          qg[i] = to_fp8(normalized);
        }
      }
    }
  }
}

template <typename T>
void fp_quantize_w_typed(
    const array& w,
    array& w_q,
    array& scales,
    int group_size,
    int bits,
    float scale_encode) {
  int K = w.shape(-1);
  size_t n_rows = w.size() / K;
  auto w_ptr = w.data<T>();
  auto w_q_ptr = reinterpret_cast<uint8_t*>(w_q.data<uint32_t>());
  auto scales_ptr = scales.data<uint8_t>();

  auto run_rows = [&](size_t start, size_t end) {
    if (bits == 8) {
      fp_quantize_w_rows<T, 32, 8>(
          w_ptr, w_q_ptr, scales_ptr, start, end, K, scale_encode);
    } else if (group_size == 32) {
      fp_quantize_w_rows<T, 32, 4>(
          w_ptr, w_q_ptr, scales_ptr, start, end, K, scale_encode);
    } else {
      fp_quantize_w_rows<T, 16, 4>(
          w_ptr, w_q_ptr, scales_ptr, start, end, K, scale_encode);
    }
  };

  auto& pool = cpu::ThreadPool::instance();
  int n_threads = cpu::effective_threads(w.size(), pool.max_threads());
  n_threads = static_cast<int>(
      std::min<size_t>(static_cast<size_t>(n_threads), n_rows));
  if (n_threads > 1) {
    pool.parallel_for(n_threads, [&](int tid, int nth) {
      size_t chunk = (n_rows + nth - 1) / nth;
      size_t start = chunk * tid;
      size_t end = std::min(start + chunk, n_rows);
      if (start < end) {
        run_rows(start, end);
      }
    });
  } else {
    run_rows(0, n_rows);
  }
}

void fp_quantize_w_dispatch(
    const array& w,
    array& w_q,
    array& scales,
    int group_size,
    int bits,
    float scale_encode) {
  switch (w.dtype()) {
    case float32:
      fp_quantize_w_typed<float>(
          w, w_q, scales, group_size, bits, scale_encode);
      break;
    case float16:
      fp_quantize_w_typed<float16_t>(
          w, w_q, scales, group_size, bits, scale_encode);
      break;
    case bfloat16:
      fp_quantize_w_typed<bfloat16_t>(
          w, w_q, scales, group_size, bits, scale_encode);
      break;
    default:
      throw std::runtime_error(
          "[QQMatmul] Only floating point weights can be quantized");
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

// Smallest E8M0 >= x, so a block's largest elements do not saturate.
uint8_t to_fp8_e8m0_round_up(float x) {
  uint8_t bits = to_fp8_e8m0(x);
  if (bits < 0xFE && dequantize_scale<float, 32>(bits) < x) {
    bits += 1;
  }
  return bits;
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
      scale = dequantize_scale<float, 32>(to_fp8_e8m0_round_up(scale));
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

namespace {

// Shared setup for QQMatmul / GatherQQMM: when w arrives unquantized (float),
// allocate packed w_q + scales temporaries in the layout the fp_qmm kernels
// consume. The actual quantization runs inside the dispatch lambda because
// the nvfp4 scale encoding depends on global_scale_w's data.
std::pair<array, array> make_fp_quant_temporaries(
    const array& w,
    int group_size,
    int bits,
    cpu::CommandEncoder& encoder) {
  auto wq_shape = w.shape();
  wq_shape.back() = w.shape(-1) * bits / 32;
  auto sc_shape = w.shape();
  sc_shape.back() = w.shape(-1) / group_size;

  size_t wq_bytes = (w.size() / 32) * bits * sizeof(uint32_t);
  size_t sc_bytes = w.size() / group_size;
  array w_q(allocator::malloc(wq_bytes), std::move(wq_shape), uint32);
  array w_scales(allocator::malloc(sc_bytes), std::move(sc_shape), uint8);
  encoder.add_temporary(w_q);
  encoder.add_temporary(w_scales);
  return {std::move(w_q), std::move(w_scales)};
}

} // namespace

void QQMatmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = cpu::get_command_encoder(stream());

  auto x = ensure_row_contiguous(inputs[0], encoder, stream());
  auto w_src = ensure_row_contiguous(inputs[1], encoder, stream());
  bool w_quantized = w_src.dtype() == uint32;
  size_t base_size = w_quantized ? 3 : 2;
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
  array global_scale_x(0.0f, float32);
  array global_scale_w(0.0f, float32);
  if (has_global_scales) {
    global_scale_x =
        ensure_row_contiguous(inputs[base_size], encoder, stream());
    global_scale_w =
        ensure_row_contiguous(inputs[base_size + 1], encoder, stream());
  }

  auto [w, scales] = w_quantized
      ? std::make_pair(
            w_src, ensure_row_contiguous(inputs[2], encoder, stream()))
      : make_fp_quant_temporaries(w_src, group_size_, bits_, encoder);

  out.set_data(allocator::malloc(out.nbytes()));

  // Create a copy of x for quantize-dequantize (simulates quantization noise)
  auto x_hat = array(x.shape(), x.dtype(), nullptr, {});
  copy_cpu(x, x_hat, CopyType::General, stream());
  encoder.add_temporary(x_hat);

  encoder.set_input_array(x_hat);
  encoder.set_input_array(w_src);
  encoder.set_input_array(scales);
  if (has_global_scales) {
    encoder.set_input_array(global_scale_x);
    encoder.set_input_array(global_scale_w);
  }
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out),
                    x_hat = array::unsafe_weak_copy(x_hat),
                    w_src = array::unsafe_weak_copy(w_src),
                    w = array::unsafe_weak_copy(w),
                    scales = array::unsafe_weak_copy(scales),
                    global_scale_x = array::unsafe_weak_copy(global_scale_x),
                    global_scale_w = array::unsafe_weak_copy(global_scale_w),
                    has_global_scales,
                    w_quantized,
                    group_size = group_size_,
                    bits = bits_,
                    mode = mode_]() mutable {
    float x_global_scale = 0.0f;
    float w_scale_factor = 1.0f;
    float w_scale_encode = 1.0f;
    if (has_global_scales) {
      x_global_scale = global_scale_x.data<float>()[0];
      w_scale_factor = global_scale_w.data<float>()[0] / (448.0f * 6.0f);
      w_scale_encode = w_scale_factor != 0.0f ? 1.0f / w_scale_factor : 1.0f;
    }
    // Quantize float weights on the fly into the packed fp-quant layout
    if (!w_quantized) {
      fp_quantize_w_dispatch(
          w_src, w, scales, group_size, bits, w_scale_encode);
    }
    // Quantize-dequantize x in-place to simulate input quantization noise
    quantize_dequantize_fp_dispatch(
        x_hat, group_size, bits, mode, x_global_scale);
    // Compute x_hat @ dequant(w) using existing FP quantized matmul
    fp_qmm_dispatch(
        out, x_hat, w, scales, group_size, bits, true, w_scale_factor);
  });
}

void GatherQQMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = cpu::get_command_encoder(stream());

  // Inputs: x, w, lhs_indices, rhs_indices, [scales_w], [gs_x, gs_w]
  auto& x_pre = inputs[0];
  auto w_src = ensure_row_contiguous(inputs[1], encoder, stream());
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];
  bool w_quantized = w_src.dtype() == uint32;
  size_t base_size = w_quantized ? 5 : 4;
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
  array global_scale_x(0.0f, float32);
  array global_scale_w(0.0f, float32);
  if (has_global_scales) {
    global_scale_x =
        ensure_row_contiguous(inputs[base_size], encoder, stream());
    global_scale_w =
        ensure_row_contiguous(inputs[base_size + 1], encoder, stream());
  }

  auto [w, scales] = w_quantized
      ? std::make_pair(
            w_src, ensure_row_contiguous(inputs[4], encoder, stream()))
      : make_fp_quant_temporaries(w_src, group_size_, bits_, encoder);

  out.set_data(allocator::malloc(out.nbytes()));

  // Contiguous copy of x, quantize-dequantized in place before the gathered
  // matmul (simulates activation quantization noise, matching the GPU path).
  auto x_hat = array(x_pre.shape(), x_pre.dtype(), nullptr, {});
  copy_cpu(x_pre, x_hat, CopyType::General, stream());
  encoder.add_temporary(x_hat);

  encoder.set_input_array(x_hat);
  encoder.set_input_array(w_src);
  encoder.set_input_array(scales);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  if (has_global_scales) {
    encoder.set_input_array(global_scale_x);
    encoder.set_input_array(global_scale_w);
  }
  encoder.set_output_array(out);
  encoder.dispatch([out = array::unsafe_weak_copy(out),
                    x_hat = array::unsafe_weak_copy(x_hat),
                    w_src = array::unsafe_weak_copy(w_src),
                    w = array::unsafe_weak_copy(w),
                    scales = array::unsafe_weak_copy(scales),
                    lhs_indices = array::unsafe_weak_copy(lhs_indices),
                    rhs_indices = array::unsafe_weak_copy(rhs_indices),
                    global_scale_x = array::unsafe_weak_copy(global_scale_x),
                    global_scale_w = array::unsafe_weak_copy(global_scale_w),
                    has_global_scales,
                    w_quantized,
                    group_size = group_size_,
                    bits = bits_,
                    mode = mode_]() mutable {
    float x_global_scale = 0.0f;
    float w_scale_factor = 1.0f;
    float w_scale_encode = 1.0f;
    if (has_global_scales) {
      x_global_scale = global_scale_x.data<float>()[0];
      w_scale_factor = global_scale_w.data<float>()[0] / (448.0f * 6.0f);
      w_scale_encode = w_scale_factor != 0.0f ? 1.0f / w_scale_factor : 1.0f;
    }
    if (!w_quantized) {
      fp_quantize_w_dispatch(
          w_src, w, scales, group_size, bits, w_scale_encode);
    }
    quantize_dequantize_fp_dispatch(
        x_hat, group_size, bits, mode, x_global_scale);
    // Gathered matmul over (lhs, rhs) index pairs; reuses the fp gathered
    // QMM path (MoE-style outer parallelism for M == 1, expert batching
    // otherwise).
    fp_bs_qmm_dispatch(
        out,
        x_hat,
        w,
        scales,
        lhs_indices,
        rhs_indices,
        group_size,
        bits,
        true,
        w_scale_factor);
  });
}

} // namespace mlx::core
