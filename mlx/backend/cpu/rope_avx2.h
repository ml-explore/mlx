// Copyright © 2026 Apple Inc.
//
// AVX2 implementation for traditional RoPE (Rotary Position Embedding).
// Included conditionally by rope.cpp inside the ISA dispatch chain.

#pragma once

#ifdef __AVX2__

#include "mlx/backend/cpu/simd/simd.h"

#include <immintrin.h>

constexpr bool has_simd_rope = true;

// SIMD vectorized traditional RoPE for 4 pairs (8 elements) at a time.
// Handles interleaved [x0,y0,x1,y1,...] layout by shuffling within pairs.
// Returns the number of half_dims pairs processed (caller handles scalar tail).
template <typename T, bool forward>
inline int rope_traditional_simd(
    const T* x_in,
    T* x_out,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  using namespace simd;
  // Permute index to duplicate each of 4 values: [v0,v0,v1,v1,v2,v2,v3,v3]
  static const __m256i dup_idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  // Sign vector for rotation:
  //   forward: out = vals*cos + swapped*sin*[-1,1,-1,1,...]
  //   reverse: out = vals*cos + swapped*sin*[1,-1,1,-1,...]
  static const __m256 fwd_signs = _mm256_setr_ps(-1, 1, -1, 1, -1, 1, -1, 1);
  static const __m256 rev_signs = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
  const __m256 signs = forward ? fwd_signs : rev_signs;

  int j = 0;
  for (; j + 4 <= half_dims; j += 4) {
    // Load 8 interleaved values -> 8 floats: [x0,y0,x1,y1,x2,y2,x3,y3]
    __m256 vals = Simd<float, 8>(load<T, 8>(x_in + 2 * j)).value;

    // Swap within pairs: [y0,x0,y1,x1,y2,x2,y3,x3]
    __m256 swapped = _mm256_shuffle_ps(vals, vals, _MM_SHUFFLE(2, 3, 0, 1));

    // Load 4 cos/sin, duplicate to pair layout
    __m256 c = _mm256_permutevar8x32_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(cos_t + j)), dup_idx);
    __m256 s = _mm256_permutevar8x32_ps(
        _mm256_castps128_ps256(_mm_loadu_ps(sin_t + j)), dup_idx);

    // out = vals * cos + swapped * sin * signs
    __m256 result = _mm256_fmadd_ps(
        _mm256_mul_ps(swapped, signs), s, _mm256_mul_ps(vals, c));

    store(x_out + 2 * j, Simd<T, 8>(Simd<float, 8>(result)));
  }
  return j;
}

#endif // __AVX2__
