// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/common/simd/scalar_math.h"
#include "mlx/backend/common/simd/type.h"

namespace mlx::core::simd {

constexpr float inf = std::numeric_limits<float>::infinity();

/**
 * Compute exp(x) in an optimizer friendly way as follows:
 *
 * First change the problem to computing 2**y where y = x / ln(2).
 *
 * Now we will compute 2**y as 2**y1 * 2**y2 where y1 is the integer part
 * `ipart` and y2 is fractional part. For the integer part we perform bit
 * shifting and for the fractional part we use a polynomial approximation.
 *
 * The algorithm and constants of the polynomial taken from
 * https://github.com/akohlmey/fastermath/blob/master/src/exp.c which took them
 * from Cephes math library.
 *
 * Note: The implementation below is a general fast exp. There could be faster
 *       implementations for numbers strictly < 0.
 */
template <typename T, int N>
Simd<T, N> exp(Simd<T, N> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else {
    Simd<float, N> x_init = in;
    auto x = x_init * 1.442695f; // multiply with log_2(e)
    Simd<float, N> ipart, fpart;
    ipart = floor(x + 0.5);
    fpart = x - ipart;

    x = 1.535336188319500e-4f;
    x = fma(x, fpart, 1.339887440266574e-3f);
    x = fma(x, fpart, 9.618437357674640e-3f);
    x = fma(x, fpart, 5.550332471162809e-2f);
    x = fma(x, fpart, 2.402264791363012e-1f);
    x = fma(x, fpart, 6.931472028550421e-1f);
    x = fma(x, fpart, 1.000000000000000f);

    // generate 2**ipart in the floating point representation using integer
    // bitshifting
    Simd<int, N> epart = (Simd<int, N>(ipart) + 127) << 23;

    // Deal with NaN and Inf
    auto result = select(isnan(x_init), x_init, (*(Simd<float, N>*)&epart) * x);
    result = select(x_init > 88.0f, Simd<float, N>(inf), result);
    result = select(x_init < -88.0f, Simd<float, N>(0), result);
    return Simd<T, N>(result);
  }
}

template <typename T, int N>
Simd<T, N> erfinv(Simd<T, N> in) {
  Simd<T, N> out;
  for (int i = 0; i < N; ++i) {
    out[i] = fast_erfinv(in[i]);
  }
  return out;
}

} // namespace mlx::core::simd
