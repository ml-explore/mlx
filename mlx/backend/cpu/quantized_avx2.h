// Copyright © 2026 Apple Inc.
//
// AVX2 implementations for quantized matmul.
// Included conditionally by quantized.cpp inside the ISA dispatch chain.
// All functions use ISA-neutral names so callers don't depend on the arch.

#pragma once

#ifdef __AVX2__

#include "mlx/backend/cpu/quantized.h"

#include <immintrin.h>

namespace mlx::core {

constexpr bool has_simd_qmm = true;

// ---------------------------------------------------------------------------
// LoadAsFloat specializations: hardware-accelerated float16/bfloat16 -> float
// ---------------------------------------------------------------------------

template <>
struct LoadAsFloat<float16_t, 8> {
  static inline simd::Simd<float, 8> apply(const float16_t* ptr) {
    return simd::Simd<float, 8>(simd::load<float16_t, 8>(ptr));
  }
};

template <>
struct LoadAsFloat<bfloat16_t, 8> {
  static inline simd::Simd<float, 8> apply(const bfloat16_t* ptr) {
    return simd::Simd<float, 8>(simd::load<bfloat16_t, 8>(ptr));
  }
};

// ---------------------------------------------------------------------------
// Bit extraction helpers
// ---------------------------------------------------------------------------

// Batch-extract 32 packed 4-bit values from 4 uint32 words into 4 float SIMD
// vectors. Each byte holds two 4-bit elements: bits [3:0] = even index,
// bits [7:4] = odd index. Separates even/odd with mask+shift, interleaves
// back into sequential order, then converts to float.
inline void extract_bits_batch_4bit(
    const uint32_t* w,
    simd::Simd<float, 8>& out0,
    simd::Simd<float, 8>& out1,
    simd::Simd<float, 8>& out2,
    simd::Simd<float, 8>& out3) {
  __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
  __m128i mask = _mm_set1_epi8(0x0F);

  __m128i even_elems = _mm_and_si128(packed, mask);
  __m128i odd_elems = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

  __m128i seq_lo = _mm_unpacklo_epi8(even_elems, odd_elems);
  __m128i seq_hi = _mm_unpackhi_epi8(even_elems, odd_elems);

  out0 = simd::Simd<float, 8>(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(seq_lo)));
  out1 = simd::Simd<float, 8>(
      _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(seq_lo, 8))));
  out2 = simd::Simd<float, 8>(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(seq_hi)));
  out3 = simd::Simd<float, 8>(
      _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(seq_hi, 8))));
}

// Batch-extract 32 packed 8-bit values from 8 uint32 words into 4 float SIMD
// vectors. Each byte is one element -- just zero-extend to int32 and convert.
inline void extract_bits_batch_8bit(
    const uint32_t* w,
    simd::Simd<float, 8>& out0,
    simd::Simd<float, 8>& out1,
    simd::Simd<float, 8>& out2,
    simd::Simd<float, 8>& out3) {
  __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
  __m128i lo = _mm256_castsi256_si128(packed);
  __m128i hi = _mm256_extracti128_si256(packed, 1);

  out0 = simd::Simd<float, 8>(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)));
  out1 = simd::Simd<float, 8>(
      _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(lo, 8))));
  out2 = simd::Simd<float, 8>(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)));
  out3 = simd::Simd<float, 8>(
      _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(hi, 8))));
}

// Unpack 32 packed 4-bit nibbles into one __m256i of 32x uint8 (range 0..15).
// Used by unsigned maddubs path.
inline __m256i nibbles_to_uint8(const uint32_t* w) {
  __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
  __m128i mask = _mm_set1_epi8(0x0F);
  __m128i even = _mm_and_si128(packed, mask);
  __m128i odd = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
  __m128i seq_lo = _mm_unpacklo_epi8(even, odd);
  __m128i seq_hi = _mm_unpackhi_epi8(even, odd);
  return _mm256_set_m128i(seq_hi, seq_lo);
}

// Load 32 packed 8-bit weights, center (subtract 128), clamp to [-127,127].
inline __m256i bytes_to_int8_centered(const uint32_t* w) {
  __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
  __m256i centered = _mm256_sub_epi8(raw, _mm256_set1_epi8(-128));
  return _mm256_max_epi8(centered, _mm256_set1_epi8(-127));
}

// ---------------------------------------------------------------------------
// Activation quantization: fused sum + amax + quantize-to-int8
// Consolidates two identical copies from _qmm_t_simd_row and _qmm_t_simd.
// ---------------------------------------------------------------------------

template <typename T, int group_size>
void quantize_activation_int8(
    const T* x,
    int K,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums) {
  alignas(32) float x_f32_tmp[128]; // group_size temp buffer, L1-hot
  const __m256i perm_fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  int groups_per_col = K / group_size;

  for (int g = 0; g < groups_per_col; g++) {
    const T* xg = x + g * group_size;
    simd::Simd<float, 8> sum_acc(0);
    __m256 vmax = _mm256_setzero_ps();

    // Pass 1: convert to f32, compute sum + amax
    for (int e = 0; e < group_size; e += 8) {
      __m256 xf = load_as_float<T, 8>(xg + e).value;
      _mm256_store_ps(x_f32_tmp + e, xf);
      sum_acc = sum_acc + simd::Simd<float, 8>(xf);
      vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, xf));
    }
    x_group_sums[g] = simd::sum(sum_acc);

    // Horizontal max
    __m128 hi128 = _mm_max_ps(
        _mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    __m128 hi64 = _mm_movehl_ps(hi128, hi128);
    hi128 = _mm_max_ps(hi128, hi64);
    __m128 hi32 = _mm_shuffle_ps(hi128, hi128, 1);
    hi128 = _mm_max_ss(hi128, hi32);
    float amax = _mm_cvtss_f32(hi128);

    float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;
    x_scales[g] = amax / 127.0f;

    // Pass 2: quantize f32 -> int8
    __m256 vinv = _mm256_set1_ps(inv_scale);
    for (int e = 0; e < group_size; e += 32) {
      __m256i i0 = _mm256_cvtps_epi32(
          _mm256_mul_ps(_mm256_load_ps(x_f32_tmp + e), vinv));
      __m256i i1 = _mm256_cvtps_epi32(
          _mm256_mul_ps(_mm256_load_ps(x_f32_tmp + e + 8), vinv));
      __m256i i2 = _mm256_cvtps_epi32(
          _mm256_mul_ps(_mm256_load_ps(x_f32_tmp + e + 16), vinv));
      __m256i i3 = _mm256_cvtps_epi32(
          _mm256_mul_ps(_mm256_load_ps(x_f32_tmp + e + 24), vinv));
      __m256i p01 = _mm256_packs_epi32(i0, i1);
      __m256i p23 = _mm256_packs_epi32(i2, i3);
      __m256i bytes = _mm256_packs_epi16(p01, p23);
      bytes = _mm256_permutevar8x32_epi32(bytes, perm_fix);
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(x_q + g * group_size + e), bytes);
    }
  }
}

// ---------------------------------------------------------------------------
// Int8 maddubs multi-col kernel with int32 accumulation.
// 4-bit: unsigned maddubs -- raw weights [0,15] as first operand.
// 8-bit: sign trick -- centered weights [-127,127] with abs(x) + sign_epi8.
// ---------------------------------------------------------------------------

template <typename T, int bits, int group_size, int NC>
void _qmm_t_int8_multi_col(
    T* result,
    const int8_t* x_q,
    const float* x_scales,
    const uint32_t* w_ptrs[NC],
    const T* scales_ptrs[NC],
    const T* biases_ptrs[NC],
    const float* x_group_sums,
    int K) {
  constexpr int S = 8;
  constexpr int w_advance = 32 / (32 / bits);
  const __m256i ones_16 = _mm256_set1_epi16(1);

  simd::Simd<float, S> acc[NC];
  for (int c = 0; c < NC; c++)
    acc[c] = simd::Simd<float, S>(0);
  float bias_acc[NC] = {};

  int groups = K / group_size;
  for (int g = 0; g < groups; g++) {
    float scale_f[NC], bias_f[NC];
    for (int c = 0; c < NC; c++) {
      scale_f[c] = static_cast<float>(*scales_ptrs[c]++);
      bias_f[c] = static_cast<float>(*biases_ptrs[c]++);
    }

    __m256i ig_acc[NC];
    for (int c = 0; c < NC; c++)
      ig_acc[c] = _mm256_setzero_si256();

    const int8_t* xq_local = x_q + g * group_size;

    for (int elem = 0; elem < group_size; elem += 32) {
      __m256i x_i8 =
          _mm256_load_si256(reinterpret_cast<const __m256i*>(xq_local + elem));

      if constexpr (bits == 4) {
        for (int c = 0; c < NC; c++) {
          __m256i w_u8 = nibbles_to_uint8(w_ptrs[c]);
          w_ptrs[c] += w_advance;
          __m256i prod16 = _mm256_maddubs_epi16(w_u8, x_i8);
          __m256i dot32 = _mm256_madd_epi16(prod16, ones_16);
          ig_acc[c] = _mm256_add_epi32(ig_acc[c], dot32);
        }
      } else {
        constexpr int CENTER = (1 << (bits - 1));
        __m256i abs_x = _mm256_abs_epi8(x_i8);
        for (int c = 0; c < NC; c++) {
          __m256i w_centered = bytes_to_int8_centered(w_ptrs[c]);
          w_ptrs[c] += w_advance;
          __m256i w_signed = _mm256_sign_epi8(w_centered, x_i8);
          __m256i prod16 = _mm256_maddubs_epi16(abs_x, w_signed);
          __m256i dot32 = _mm256_madd_epi16(prod16, ones_16);
          ig_acc[c] = _mm256_add_epi32(ig_acc[c], dot32);
        }
      }
    }

    float xs = x_scales[g];
    float xgs = x_group_sums[g];
    for (int c = 0; c < NC; c++) {
      float combined = scale_f[c] * xs;
      acc[c] = simd::fma(
          simd::Simd<float, S>(combined),
          simd::Simd<float, S>(_mm256_cvtepi32_ps(ig_acc[c])),
          acc[c]);
      if constexpr (bits == 4) {
        bias_acc[c] += bias_f[c] * xgs;
      } else {
        constexpr int CENTER = (1 << (bits - 1));
        bias_acc[c] += (scale_f[c] * CENTER + bias_f[c]) * xgs;
      }
    }
  }

  for (int c = 0; c < NC; c++) {
    result[c] = T(simd::sum(acc[c]) + bias_acc[c]);
  }
}

// ---------------------------------------------------------------------------
// Int8 maddubs single-col kernel (for tail columns).
// ---------------------------------------------------------------------------

template <typename T, int bits, int group_size>
float _qmm_t_int8_single_col(
    const int8_t* x_q,
    const float* x_scales,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    const float* x_group_sums,
    int K) {
  constexpr int S = 8;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const __m256i ones_16 = _mm256_set1_epi16(1);

  int groups_per_col = K / group_size;

  simd::Simd<float, S> acc(0);
  float bias_acc = 0;

  for (int g = 0; g < groups_per_col; g++) {
    float scale_f = static_cast<float>(scales[g]);
    float bias_f = static_cast<float>(biases[g]);
    const uint32_t* w_local = w + g * packs_in_group;
    const int8_t* xq_local = x_q + g * group_size;

    __m256i ig_acc = _mm256_setzero_si256();
    for (int elem = 0; elem < group_size; elem += 32) {
      __m256i x_i8 =
          _mm256_load_si256(reinterpret_cast<const __m256i*>(xq_local + elem));

      if constexpr (bits == 4) {
        __m256i w_u8 = nibbles_to_uint8(w_local + (elem * bits / 32));
        __m256i prod16 = _mm256_maddubs_epi16(w_u8, x_i8);
        __m256i dot32 = _mm256_madd_epi16(prod16, ones_16);
        ig_acc = _mm256_add_epi32(ig_acc, dot32);
      } else {
        __m256i abs_x = _mm256_abs_epi8(x_i8);
        __m256i w_centered =
            bytes_to_int8_centered(w_local + (elem * bits / 32));
        __m256i w_signed = _mm256_sign_epi8(w_centered, x_i8);
        __m256i prod16 = _mm256_maddubs_epi16(abs_x, w_signed);
        __m256i dot32 = _mm256_madd_epi16(prod16, ones_16);
        ig_acc = _mm256_add_epi32(ig_acc, dot32);
      }
    }

    float combined = scale_f * x_scales[g];
    acc = simd::fma(
        simd::Simd<float, S>(combined),
        simd::Simd<float, S>(_mm256_cvtepi32_ps(ig_acc)),
        acc);
    if constexpr (bits == 4) {
      bias_acc += bias_f * x_group_sums[g];
    } else {
      constexpr int CENTER = (1 << (bits - 1));
      bias_acc += (scale_f * CENTER + bias_f) * x_group_sums[g];
    }
  }

  return simd::sum(acc) + bias_acc;
}

// ---------------------------------------------------------------------------
// SIMD-optimized batch dequantization
// ---------------------------------------------------------------------------

template <int bits, int group_size>
void _dequant_row_simd(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int K) {
  constexpr int pack_factor = 32 / bits;
  constexpr int words_per_batch = 32 / pack_factor;
  constexpr int batches_per_group = group_size / 32;

  int k = 0;
  for (int g = 0; g < K / group_size; g++) {
    auto vscale = simd::Simd<float, 8>(scales_row[g]);
    auto vbias = simd::Simd<float, 8>(biases_row[g]);

    const uint32_t* w_ptr = w_row + (k / pack_factor);
    for (int b = 0; b < batches_per_group; b++) {
      simd::Simd<float, 8> v0, v1, v2, v3;
      if constexpr (bits == 4) {
        extract_bits_batch_4bit(w_ptr, v0, v1, v2, v3);
      } else {
        extract_bits_batch_8bit(w_ptr, v0, v1, v2, v3);
      }
      simd::store(out + k, simd::fma(v0, vscale, vbias));
      simd::store(out + k + 8, simd::fma(v1, vscale, vbias));
      simd::store(out + k + 16, simd::fma(v2, vscale, vbias));
      simd::store(out + k + 24, simd::fma(v3, vscale, vbias));
      k += 32;
      w_ptr += words_per_batch;
    }
  }
}

// ---------------------------------------------------------------------------
// Fast-path dispatch: batch-extract in multi-col kernel
// Returns number of elements processed (0 if not applicable).
// ---------------------------------------------------------------------------

template <typename T, int bits, int group_size, int NC>
int try_batch_extract_multi_col(
    const T*& x_local,
    const uint32_t* w_ptrs[NC],
    simd::Simd<float, 8> g_acc[NC],
    int group_size_val) {
  constexpr int S = 8;
  if constexpr ((bits == 4 || bits == 8) && S == 8 && group_size >= 32) {
    int elem = 0;
    for (; elem + 32 <= group_size; elem += 32) {
      simd::Simd<float, S> x0 = load_as_float<T, S>(x_local);
      simd::Simd<float, S> x1 = load_as_float<T, S>(x_local + 8);
      simd::Simd<float, S> x2 = load_as_float<T, S>(x_local + 16);
      simd::Simd<float, S> x3 = load_as_float<T, S>(x_local + 24);
      x_local += 32;

#pragma clang loop unroll(full)
      for (int c = 0; c < NC; c++) {
        simd::Simd<float, S> w0, w1, w2, w3;
        if constexpr (bits == 4) {
          extract_bits_batch_4bit(w_ptrs[c], w0, w1, w2, w3);
          w_ptrs[c] += 4;
        } else {
          extract_bits_batch_8bit(w_ptrs[c], w0, w1, w2, w3);
          w_ptrs[c] += 8;
        }
        g_acc[c] = simd::fma(x0, w0, g_acc[c]);
        g_acc[c] = simd::fma(x1, w1, g_acc[c]);
        g_acc[c] = simd::fma(x2, w2, g_acc[c]);
        g_acc[c] = simd::fma(x3, w3, g_acc[c]);
      }
    }
    return elem;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Int8 path for _qmm_t_simd_row: try AVX2 int8 maddubs path.
// Returns true if handled (caller should return), false to fall through.
// ---------------------------------------------------------------------------

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
    int packs_per_col,
    float* x_group_sums,
    const PreqAct* preq) {
  constexpr int S = simd::max_size<float>;
  constexpr int NC = 8;
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;

  if constexpr ((bits == 4 || bits == 8) && S == 8 && group_size >= 32) {
    if (!env::enable_tf32())
      return false;
    if (K > INT8_MAX_K)
      return false;

    // When pre-quantized data isn't available, each call pays O(K) quantization
    // overhead. Only worthwhile when processing enough columns to amortize it.
    // With preq (M==1 parallel path), quantization was done once by the caller.
    int n_cols = n_end - n_start;
    if (!preq && n_cols < K)
      return false;

    // Use pre-quantized data if available, otherwise quantize locally
    alignas(32) int8_t x_q_local[INT8_MAX_K];
    float x_scales_local[STACK_GROUPS];
    std::unique_ptr<float[]> xs_heap;
    const int8_t* x_q;
    const float* x_scales;

    if (preq) {
      x_q = static_cast<const int8_t*>(preq->x_q);
      x_scales = preq->x_scales;
    } else {
      float* x_scales_w;
      if (groups_per_col <= STACK_GROUPS) {
        x_scales_w = x_scales_local;
      } else {
        xs_heap.reset(new float[groups_per_col]);
        x_scales_w = xs_heap.get();
      }

      // Fused pass: compute x_group_sums AND quantize x to int8
      quantize_activation_int8<T, group_size>(
          x, K, x_q_local, x_scales_w, x_group_sums);

      x_q = x_q_local;
      x_scales = x_scales_w;
    }

    // Dispatch to int8 maddubs kernels
    const uint32_t* w_base = w + n_start * packs_per_col;
    const T* scales_base = scales + n_start * groups_per_col;
    const T* biases_base = biases + n_start * groups_per_col;
    int n = n_start;

    for (; n + NC <= n_end; n += NC) {
      const uint32_t* wp[NC];
      const T* sp[NC];
      const T* bp[NC];
      for (int c = 0; c < NC; c++) {
        wp[c] = w_base + c * packs_per_col;
        sp[c] = scales_base + c * groups_per_col;
        bp[c] = biases_base + c * groups_per_col;
      }
      _qmm_t_int8_multi_col<T, bits, group_size, NC>(
          result + (n - n_start), x_q, x_scales, wp, sp, bp, x_group_sums, K);
      w_base += NC * packs_per_col;
      scales_base += NC * groups_per_col;
      biases_base += NC * groups_per_col;
    }

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
      _qmm_t_int8_multi_col<T, bits, group_size, NC4>(
          result + (n - n_start), x_q, x_scales, wp, sp, bp, x_group_sums, K);
      w_base += NC4 * packs_per_col;
      scales_base += NC4 * groups_per_col;
      biases_base += NC4 * groups_per_col;
    }

    for (; n < n_end; n++) {
      result[n - n_start] =
          T(_qmm_t_int8_single_col<T, bits, group_size>(
              x_q,
              x_scales,
              w_base,
              scales_base,
              biases_base,
              x_group_sums,
              K));
      w_base += packs_per_col;
      scales_base += groups_per_col;
      biases_base += groups_per_col;
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Int8 pre-quantization + parallel dispatch for M==1 path in _qmm_t_simd.
// Returns true if handled.
// ---------------------------------------------------------------------------

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
  constexpr int S = simd::max_size<float>;
  constexpr int INT8_MAX_K = 16384;
  constexpr int STACK_GROUPS = 128;

  if constexpr ((bits == 4 || bits == 8) && S == 8 && group_size >= 32) {
    if (!env::enable_tf32())
      return false;
    if (K > INT8_MAX_K)
      return false;

    int groups_per_col = K / group_size;

    alignas(32) int8_t x_q[INT8_MAX_K];
    float x_scales_buf[STACK_GROUPS];
    float x_group_sums_buf[STACK_GROUPS];
    std::unique_ptr<float[]> xs_heap, xgs_heap;
    float* x_scales = x_scales_buf;
    float* x_group_sums = x_group_sums_buf;
    if (groups_per_col > STACK_GROUPS) {
      xs_heap.reset(new float[groups_per_col]);
      xgs_heap.reset(new float[groups_per_col]);
      x_scales = xs_heap.get();
      x_group_sums = xgs_heap.get();
    }

    // Quantize x to int8 (once, on main thread)
    quantize_activation_int8<T, group_size>(x, K, x_q, x_scales, x_group_sums);

    auto& pool = cpu::ThreadPool::instance();
    PreqAct preq{x_q, x_scales, x_group_sums};
    steal_counter.store(n_threads, std::memory_order_relaxed);
    pool.parallel_for(n_threads, [&](int tid, int /*nth*/) {
      int my_chunk = tid;
      while (my_chunk < n_chunks) {
        int n_start = std::min(my_chunk * CHUNK_COLS, N);
        int n_end = std::min(n_start + CHUNK_COLS, N);
        if (n_start < n_end) {
          _qmm_t_simd_row<T, bits, group_size>(
              result + n_start, x, w, scales, biases, n_start, n_end, K, &preq);
        }
        my_chunk = steal_counter.fetch_add(1, std::memory_order_relaxed);
      }
    });
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// FP quantized LUT lookups
// ---------------------------------------------------------------------------

// FP4 LUT lookup via SIMD register-based permutation.
// Requires the FP4_LUT array to be visible (defined in quantized.cpp).
template <int S>
simd::Simd<float, S> fp4_lut_lookup_simd(
    const simd::Simd<uint32_t, S>& wi,
    const float* FP4_LUT_ptr) {
  if constexpr (S == 8) {
    // LUT has 16 entries but vpermps handles only 8 indices (0-7).
    // Split: vpermps for low half (0-7), high half (8-15), blend.
    __m256 lut_lo = _mm256_setr_ps(
        FP4_LUT_ptr[0],
        FP4_LUT_ptr[1],
        FP4_LUT_ptr[2],
        FP4_LUT_ptr[3],
        FP4_LUT_ptr[4],
        FP4_LUT_ptr[5],
        FP4_LUT_ptr[6],
        FP4_LUT_ptr[7]);
    __m256 lut_hi = _mm256_setr_ps(
        FP4_LUT_ptr[8],
        FP4_LUT_ptr[9],
        FP4_LUT_ptr[10],
        FP4_LUT_ptr[11],
        FP4_LUT_ptr[12],
        FP4_LUT_ptr[13],
        FP4_LUT_ptr[14],
        FP4_LUT_ptr[15]);
    __m256i idx_lo = _mm256_and_si256(wi.value, _mm256_set1_epi32(0x7));
    __m256 val_lo = _mm256_permutevar8x32_ps(lut_lo, idx_lo);
    __m256 val_hi = _mm256_permutevar8x32_ps(lut_hi, idx_lo);
    __m256i hi_mask = _mm256_slli_epi32(wi.value, 28);
    return simd::Simd<float, S>(
        _mm256_blendv_ps(val_lo, val_hi, _mm256_castsi256_ps(hi_mask)));
  }
  // Should not be reached for other S values on AVX2
  return simd::Simd<float, S>(0);
}

// FP8 LUT gather via SIMD gather instruction.
template <int S>
simd::Simd<float, S> fp8_lut_gather_simd(
    const uint32_t* w,
    const float* FP8_LUT_ptr) {
  if constexpr (S == 8) {
    auto indices = _mm256_cvtepu8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w)));
    return simd::Simd<float, S>(_mm256_i32gather_ps(FP8_LUT_ptr, indices, 4));
  }
  return simd::Simd<float, S>(0);
}

} // namespace mlx::core

#endif // __AVX2__
