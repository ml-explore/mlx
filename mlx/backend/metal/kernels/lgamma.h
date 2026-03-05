// Copyright © 2025 Apple Inc.

#pragma once

#include <metal_math>

/*
 * Log-gamma function via Lanczos approximation (g=5, 7-term).
 * Coefficients from Numerical Recipes; ~10 digits accuracy,
 * sufficient for float32.
 *
 * Uses reflection formula for x < 0.5.
 */
float lgamma_impl(float x) {
  // Poles at non-positive integers
  if (x <= 0.0f && x == metal::floor(x)) {
    return metal::numeric_limits<float>::infinity();
  }

  // Reflection formula for x < 0.5:
  //   lgamma(x) = log(pi) - log(|sin(pi*x)|) - lgamma(1-x)
  if (x < 0.5f) {
    float sin_pi_x = metal::precise::sin(M_PI_F * x);
    return 1.1447298858494002f // log(pi)
        - metal::precise::log(metal::abs(sin_pi_x)) - lgamma_impl(1.0f - x);
  }

  // Lanczos g=5, 7-term (Numerical Recipes)
  float z = x - 1.0f;
  float t = z + 5.5f; // z + g + 0.5
  float s = 1.000000000190015f;
  s = metal::fma(76.18009172947146f, 1.0f / (z + 1.0f), s);
  s = metal::fma(-86.50532032941677f, 1.0f / (z + 2.0f), s);
  s = metal::fma(24.01409824083091f, 1.0f / (z + 3.0f), s);
  s = metal::fma(-1.231739572450155f, 1.0f / (z + 4.0f), s);
  s = metal::fma(1.208650973866179e-3f, 1.0f / (z + 5.0f), s);
  s = metal::fma(-5.395239384953e-6f, 1.0f / (z + 6.0f), s);

  return 0.9189385332046727f // 0.5 * log(2*pi)
      + (z + 0.5f) * metal::precise::log(t) - t +
      metal::precise::log(s);
}

/*
 * Digamma (psi) function: d/dx[lgamma(x)].
 * Uses asymptotic expansion for x >= 10, recurrence to shift
 * small x, and reflection formula for negative x.
 *
 * Asymptotic coefficients are Bernoulli-number derived (float32-adequate).
 */
float digamma_impl(float x) {
  float result = 0.0f;

  // Reflection for negative x:
  //   digamma(x) = digamma(1-x) - pi/tan(pi*x)
  if (x < 0.0f) {
    result = -M_PI_F / metal::precise::tan(M_PI_F * x);
    x = 1.0f - x;
  }

  // Recurrence: shift x to >= 10
  //   digamma(x) = digamma(x+1) - 1/x
  while (x < 10.0f) {
    result -= 1.0f / x;
    x += 1.0f;
  }

  // Asymptotic expansion (Bernoulli numbers, Horner form)
  float z = 1.0f / (x * x);
  float y = metal::fma(3.96825396825e-3f, z, -4.16666666667e-3f);
  y = metal::fma(y, z, 7.57575757576e-3f);
  y = metal::fma(y, z, -2.10927960928e-2f);
  y = metal::fma(y, z, 8.33333333333e-2f);

  result += metal::precise::log(x) - 0.5f / x - y * z;
  return result;
}
