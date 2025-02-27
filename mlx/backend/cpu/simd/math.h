// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/cpu/simd/type.h"

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

template <typename T, int N>
Simd<T, N> erf(Simd<T, N> x) {
  // https://github.com/pytorch/pytorch/blob/abf28982a8cb43342e7669d859de9543fd804cc9/aten/src/ATen/cpu/vec/vec256/vec256_float.h#L175
  Simd<float, N> v = x;
  auto t = recip(fma(Simd<float, N>(0.3275911f), abs(v), 1.0f));
  auto r = fma(Simd<float, N>(1.061405429f), t, -1.453152027f);
  r = fma(r, t, 1.421413741f);
  r = fma(r, t, -0.284496736f);
  r = fma(r, t, 0.254829592f);
  auto e = -exp(-v * v);
  auto result = Simd<T, N>(fma(e * t, r, 1.0f));
  return select(x > 0, result, -result);
}

template <typename T, int N>
Simd<T, N> erfinv(Simd<T, N> a_) {
  Simd<float, N> a = a_;
  auto t = fma(a, 0.0f - a, 1.0f);
  t = log(t);
  auto lhs = [](auto t) {
    Simd<float, N> p;
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    return fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  };
  auto rhs = [](auto t) {
    Simd<float, N> p;
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    return fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  };
  auto thresh = 6.125f;
  // Compute both branches and select if N > 1
  if constexpr (N == 1) {
    if ((abs(t) > thresh).value) { // maximum ulp error = 2.35793
      return a * lhs(t);
    } else { // maximum ulp error = 2.35002
      return a * rhs(t);
    }
  } else {
    return a * select(abs(t) > thresh, lhs(t), rhs(t));
  }
}

} // namespace mlx::core::simd
