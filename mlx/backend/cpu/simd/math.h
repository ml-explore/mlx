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

// Clenshaw recurrence for Chebyshev series evaluation.
template <int M>
float chbevl_scalar(float x, const float (&coeffs)[M]) {
  float b0 = coeffs[0];
  float b1 = 0.0f;
  float b2;
  for (int i = 1; i < M; i++) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + coeffs[i];
  }
  return 0.5f * (b0 - b2);
}

inline float i0e_scalar(float x) {
  // Cephes Chebyshev coefficients for exp(-x) I0(x), interval [0, 8].
  static const float A_i0[30] = {
      -4.41534164647933937950e-18f, 3.33079451882223809783e-17f,
      -2.43127984654795469359e-16f, 1.71539128555513303061e-15f,
      -1.16853328779934516808e-14f, 7.67618549860493561688e-14f,
      -4.85644678311192946090e-13f, 2.95505266312963983461e-12f,
      -1.72682629144155570723e-11f, 9.67580903537323691224e-11f,
      -5.18979560163526290666e-10f, 2.65982372468238665035e-09f,
      -1.30002500998624804212e-08f, 6.04699502254191894932e-08f,
      -2.67079385394061173391e-07f, 1.11738753912010371815e-06f,
      -4.41673835845875056359e-06f, 1.64484480707288970893e-05f,
      -5.75419501008210370398e-05f, 1.88502885095841655729e-04f,
      -5.76375574538582365885e-04f, 1.63947561694133579842e-03f,
      -4.32430999505057594430e-03f, 1.05464603945949983183e-02f,
      -2.37374148058994688156e-02f, 4.93052842396707084878e-02f,
      -9.49010970480476444210e-02f, 1.71620901522208775349e-01f,
      -3.04682672343198398683e-01f, 6.76795274409476084995e-01f,
  };
  // Cephes Chebyshev coefficients for exp(-x) sqrt(x) I0(x), interval [8, inf].
  static const float B_i0[25] = {
      -7.23318048787475395456e-18f, -4.83050448594418207126e-18f,
      4.46562142029675999901e-17f,  3.46122286769746109310e-17f,
      -2.82762398051658348494e-16f, -3.42548561967721913462e-16f,
      1.77256013305652638360e-15f,  3.81168066935262242075e-15f,
      -9.55484669882830764870e-15f, -4.15056934728722208663e-14f,
      1.54008621752140982691e-14f,  3.85277838274214270114e-13f,
      7.18012445138366623367e-13f,  -1.79417853150680611778e-12f,
      -1.32158118404477131188e-11f, -3.14991652796324136454e-11f,
      1.18891471078464383424e-11f,  4.94060238822496958910e-10f,
      3.39623202570838634515e-09f,  2.26666899049817806459e-08f,
      2.04891858946906374183e-07f,  2.89137052083475648297e-06f,
      6.88975834691682398426e-05f,  3.36911647825569408990e-03f,
      8.04490411014108831608e-01f,
  };

  float ax = std::abs(x);
  if (ax <= 8.0f) {
    float y = ax * 0.5f - 2.0f;
    return chbevl_scalar(y, A_i0);
  }
  return chbevl_scalar(32.0f / ax - 2.0f, B_i0) / std::sqrt(ax);
}

inline float i1e_scalar(float x) {
  // Cephes Chebyshev coefficients for exp(-x) I1(x) / x, interval [0, 8].
  static const float A_i1[29] = {
      2.77791411276104639959e-18f,  -2.11142121435816608115e-17f,
      1.55363195773620046921e-16f,  -1.10559694773538630805e-15f,
      7.60068429473540693410e-15f,  -5.04218550472791168711e-14f,
      3.22379336594557470981e-13f,  -1.98397439776494371520e-12f,
      1.17361862988909016308e-11f,  -6.66348972350202774223e-11f,
      3.62559028155211703701e-10f,  -1.88724975172282928790e-09f,
      9.38153738649577178388e-09f,  -4.44505912879632808065e-08f,
      2.00329475355213526229e-07f,  -8.56872026469545474066e-07f,
      3.47025130813767847674e-06f,  -1.32731636560394358279e-05f,
      4.78156510755005422638e-05f,  -1.61760815825896745588e-04f,
      5.12285956168575772895e-04f,  -1.51357245063125314899e-03f,
      4.15642294431288815669e-03f,  -1.05640848946261981558e-02f,
      2.47264490306265168283e-02f,  -5.29459812080949914269e-02f,
      1.02643658689847095384e-01f,  -1.76416518357834055153e-01f,
      2.52587186443633654823e-01f,
  };
  // Cephes Chebyshev coefficients for exp(-x) sqrt(x) I1(x), interval [8, inf].
  static const float B_i1[25] = {
      7.51729631084210481353e-18f,  4.41434832307170791151e-18f,
      -4.65030536848935832153e-17f, -3.20952592199342395980e-17f,
      2.96262899764595013876e-16f,  3.30820231092092828324e-16f,
      -1.88035477551078244854e-15f, -3.81440307243700780478e-15f,
      1.04202769841288027642e-14f,  4.27244001671195135429e-14f,
      -2.10154184277266431302e-14f, -4.08355111109219731823e-13f,
      -7.19855177624590851209e-13f, 2.03562854414708950722e-12f,
      1.41258074366137813316e-11f,  3.25260358301548823856e-11f,
      -1.89749581235054123450e-11f, -5.58974346219658380687e-10f,
      -3.83538038596423702205e-09f, -2.63146884688951950684e-08f,
      -2.51223623787020892529e-07f, -3.88256480887769039346e-06f,
      -1.10588938762623716291e-04f, -9.76109749136146840777e-03f,
      7.78576235018280120474e-01f,
  };

  float ax = std::abs(x);
  float result;
  if (ax <= 8.0f) {
    float y = ax * 0.5f - 2.0f;
    result = chbevl_scalar(y, A_i1) * ax;
  } else {
    result = chbevl_scalar(32.0f / ax - 2.0f, B_i1) / std::sqrt(ax);
  }
  return x < 0.0f ? -result : result;
}

template <typename T, int N>
Simd<T, N> i0e(Simd<T, N> x) {
  Simd<T, N> result;
  for (int i = 0; i < N; i++) {
    result[i] = static_cast<T>(i0e_scalar(static_cast<float>(x[i])));
  }
  return result;
}

template <typename T>
Simd<T, 1> i0e(Simd<T, 1> x) {
  return Simd<T, 1>(static_cast<T>(i0e_scalar(static_cast<float>(x.value))));
}

template <typename T, int N>
Simd<T, N> i1e(Simd<T, N> x) {
  Simd<T, N> result;
  for (int i = 0; i < N; i++) {
    result[i] = static_cast<T>(i1e_scalar(static_cast<float>(x[i])));
  }
  return result;
}

template <typename T>
Simd<T, 1> i1e(Simd<T, 1> x) {
  return Simd<T, 1>(static_cast<T>(i1e_scalar(static_cast<float>(x.value))));
}

} // namespace mlx::core::simd
