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

/* Implementation from:
 * https://github.com/JishinMaster/simd_utils/blob/3c1433a86fb38edcc9b02039f3c9a65b16640976/neon_mathfun.h#L357
 * which originally came from the Cephes math library.
 */
template <bool Sine, typename T, int N>
Simd<T, N> sincos(Simd<T, N> in) {
  auto sign_mask_sin = in < 0;
  in = abs(in);
  Simd<float, N> x = in;

  // scale by 4/Pi
  auto y = x * 1.27323954473516f;

  // store the integer part of y in mm0
  Simd<uint32_t, N> emm2 = y;

  // j=(j+1) & (~1) (see the cephes sources)
  emm2 = emm2 + 1;
  emm2 = emm2 & ~1;

  y = emm2;

  // Get the polynom selection mask. There is one polynom for 0 <= x <= Pi/4
  // and another one for Pi/4<x<=Pi/2. Both branches will be computed.
  auto poly_mask = (emm2 & 2) != 0;

  // The magic pass: "Extended precision modular arithmetic"
  // x = ((x - y * DP1) - y * DP2) - y * DP3
  x = fma(y, Simd<float, N>(-0.78515625f), x);
  x = fma(y, Simd<float, N>(-2.4187564849853515625e-4f), x);
  x = fma(y, Simd<float, N>(-3.77489497744594108e-8f), x);

  sign_mask_sin = sign_mask_sin ^ ((emm2 & 4) != 0);
  auto sign_mask_cos = ((emm2 - 2) & 4) != 0;

  // Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
  // and the second polynom      (Pi/4 <= x <= 0) in y2
  auto z = x * x;

  auto y1 =
      fma(z, Simd<float, N>(2.443315711809948e-5f), -1.388731625493765e-3f);
  auto y2 = fma(z, Simd<float, N>(-1.9515295891e-4f), 8.3321608736e-3f);
  y1 = fma(y1, z, 4.166664568298827e-2f);
  y2 = fma(y2, z, -1.6666654611e-1f);
  y1 = y1 * z;
  y2 = y2 * z;
  y1 = y1 * z;
  y2 = fma(x, y2, x);
  y1 = fma(z, Simd<float, N>(-0.5f), y1);
  y1 = y1 + 1.0f;

  if constexpr (Sine) {
    auto ys = select(poly_mask, y1, y2);
    return select(sign_mask_sin, -ys, ys);
  } else {
    auto yc = select(poly_mask, y2, y1);
    return select(sign_mask_cos, yc, -yc);
  }
}

template <typename T, int N>
Simd<T, N> sin(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::sin(x.value);
  } else {
    return sincos<true>(x);
  }
}

template <typename T, int N>
Simd<T, N> cos(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::cos(x.value);
  } else {
    return sincos<false>(x);
  }
}

} // namespace mlx::core::simd
