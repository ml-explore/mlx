// Copyright © 2025 Apple Inc.

#pragma once
#include <metal_math>

/*
 * Modified Bessel function of the first kind, order zero: I0(x).
 * Uses the Cephes polynomial approximation in two domains.
 *
 * Domain 1: |x| <= 3.75  →  polynomial in (x/3.75)^2
 * Domain 2: |x|  > 3.75  →  exp(|x|) / sqrt(|x|) * polynomial in (3.75/|x|)
 *
 * Reference: Cephes Math Library (netlib.org/cephes)
 */
float i0_impl(float x) {
  float y = metal::abs(x);

  if (y <= 3.75f) {
    float t = y / 3.75f;
    t = t * t;
    return 1.0f +
        t * (3.5156229f +
             t * (3.0899424f +
                  t * (1.2067492f +
                       t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
  } else {
    float t = 3.75f / y;
    float p = 0.00392377f;
    p = metal::fma(p, t, -0.01647633f);
    p = metal::fma(p, t, 0.02635537f);
    p = metal::fma(p, t, -0.02057706f);
    p = metal::fma(p, t, 0.00916281f);
    p = metal::fma(p, t, -0.00157565f);
    p = metal::fma(p, t, 0.00225319f);
    p = metal::fma(p, t, 0.01328592f);
    p = metal::fma(p, t, 0.39894228f);
    return (metal::precise::exp(y) / metal::precise::sqrt(y)) * p;
  }
}
