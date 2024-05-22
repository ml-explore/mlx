// Copyright © 2023 Apple Inc.

#pragma once
#include <metal_math>

/*
 * Approximation to the error function.
 * Based on code from:
 * https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
 */
float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - metal::exp(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}

float erfinv(float a) {
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float p;
  if (metal::abs(t) > 6.125f) { // maximum ulp error = 2.35793
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = metal::fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = metal::fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = metal::fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = metal::fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = metal::fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = metal::fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = metal::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = metal::fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = metal::fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = metal::fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = metal::fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = metal::fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = metal::fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = metal::fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = metal::fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = metal::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = metal::fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  }
  return a * p;
}
