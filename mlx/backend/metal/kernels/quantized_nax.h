// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 2.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 8.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 32.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 128.0f;
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 1) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 2.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 8.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 32.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 128.0f;
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      accum +=
          (x_thread[8 * i] * (w[i] & 0x01) +
           x_thread[8 * i + 1] * (w[i] & 0x02) +
           x_thread[8 * i + 2] * (w[i] & 0x04) +
           x_thread[8 * i + 3] * (w[i] & 0x08) +
           x_thread[8 * i + 4] * (w[i] & 0x10) +
           x_thread[8 * i + 5] * (w[i] & 0x20) +
           x_thread[8 * i + 6] * (w[i] & 0x40) +
           x_thread[8 * i + 7] * (w[i] & 0x80));
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 1) {
    for (int i = 0; i < (N / 8); i++) {
      accum +=
          (x_thread[8 * i] * (w[i] & 0x01) +
           x_thread[8 * i + 1] * (w[i] & 0x02) +
           x_thread[8 * i + 2] * (w[i] & 0x04) +
           x_thread[8 * i + 3] * (w[i] & 0x08) +
           x_thread[8 * i + 4] * (w[i] & 0x10) +
           x_thread[8 * i + 5] * (w[i] & 0x20) +
           x_thread[8 * i + 6] * (w[i] & 0x40) +
           x_thread[8 * i + 7] * (w[i] & 0x80));
    }
  }

  else if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  if (bits == 1) {
    U s[8] = {
        scale,
        scale / 2.0f,
        scale / 4.0f,
        scale / 8.0f,
        scale / 16.0f,
        scale / 32.0f,
        scale / 64.0f,
        scale / 128.0f};
    for (int i = 0; i < (values_per_thread / 8); i++) {
      result[8 * i] += x * (s[0] * (w[i] & 0x01) + bias);
      result[8 * i + 1] += x * (s[1] * (w[i] & 0x02) + bias);
      result[8 * i + 2] += x * (s[2] * (w[i] & 0x04) + bias);
      result[8 * i + 3] += x * (s[3] * (w[i] & 0x08) + bias);
      result[8 * i + 4] += x * (s[4] * (w[i] & 0x10) + bias);
      result[8 * i + 5] += x * (s[5] * (w[i] & 0x20) + bias);
      result[8 * i + 6] += x * (s[6] * (w[i] & 0x40) + bias);
      result[8 * i + 7] += x * (s[7] * (w[i] & 0x80) + bias);
    }
  }

  else if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <
    typename U,
    int N,
    int bits,
    typename BytePtr = const device uint8_t*,
    typename OutPtr = threadgroup U*>
inline void dequantize(BytePtr w, U scale, U bias, OutPtr w_local) {
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  const float s = float(scale);
  const float b = float(bias);

  if (bits == 1) {
    float sc[8] = {
        s,
        s / 2.0f,
        s / 4.0f,
        s / 8.0f,
        s / 16.0f,
        s / 32.0f,
        s / 64.0f,
        s / 128.0f};
    for (int i = 0; i < (N / 8); i++) {
      w_local[8 * i] = static_cast<U>(sc[0] * (w[i] & 0x01) + b);
      w_local[8 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x02) + b);
      w_local[8 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x04) + b);
      w_local[8 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0x08) + b);
      w_local[8 * i + 4] = static_cast<U>(sc[4] * (w[i] & 0x10) + b);
      w_local[8 * i + 5] = static_cast<U>(sc[5] * (w[i] & 0x20) + b);
      w_local[8 * i + 6] = static_cast<U>(sc[6] * (w[i] & 0x40) + b);
      w_local[8 * i + 7] = static_cast<U>(sc[7] * (w[i] & 0x80) + b);
    }
  }

  else if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = static_cast<U>(sc[0] * (w[i] & 0x03) + b);
      w_local[4 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x0c) + b);
      w_local[4 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x30) + b);
      w_local[4 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0xc0) + b);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = static_cast<U>((w[0] & 0x7) * s + b);
      w_local[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
      w_local[2] =
          static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
      w_local[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
      w_local[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
      w_local[5] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
      w_local[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
      w_local[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
    }
  }

  else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct QuantizedBlockLoader<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    32,
    bits> {
  MLX_MTL_CONST short group_size = 32;

  static_assert(
      BCOLS % group_size == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 1 || bits == 2 || bits == 3 || bits == 4 || bits == 5 ||
          bits == 6 || bits == 8,
      "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        biases(biases_ + bi * src_ld / group_size + group_id) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      // if (group_steps > 1) {
      //   group_step_cnt++;
      //   if (group_step_cnt == group_steps) {
      //     group_step_cnt = 0;
      //     scales++;
      //     biases++;
      //   }
      // } else {
      scales += n_groups;
      biases += n_groups;
      // }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

// QuantizedBlockLoader's thread mapping split into two steps: fetch() reads the
// packed words and the group's scale and bias for one BK step into registers,
// store() dequantizes them into threadgroup memory. Fetching step k + 1 before
// the MMAs of step k hides the device reads behind the matrix work.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderStaged {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_words = n_reads * bytes_per_pack / 4;

  static_assert(
      BCOLS_PACKED * BROWS >= tgp_size &&
          n_words * 4 == n_reads * bytes_per_pack,
      "Every thread loads whole words");
  static_assert(
      n_reads * pack_factor <= group_size && n_reads * pack_factor <= BCOLS,
      "A thread's values must lie in one row and quantization group");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;
  uint32_t raw[n_words];
  T scale;
  T bias;

  QuantizedBlockLoaderStaged(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  // k: first column of the step
  void fetch(const int k) thread {
    auto p = (const device uint32_t*)(src + k * bytes_per_pack / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) {
      raw[i] = p[i];
    }
    const int g = (k + bj * pack_factor) / group_size;
    scale = scales[g];
    bias = biases[g];
  }

  // Dequantize into registers, then write 8 values per store
  void store(threadgroup T* dst) const thread {
    constexpr short n_vals = n_reads * pack_factor;
    static_assert(n_vals % 8 == 0, "8-value stores");
    T v[n_vals];
    // Unpack with shifts (constant indices keep it in registers)
    uint8_t b[n_words * 4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words * 4; i++) {
      b[i] = uint8_t(raw[i / 4] >> (8 * (i % 4)));
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          b + i * bytes_per_pack, scale, bias, v + i * pack_factor);
    }
    dst += bi * dst_ld + bj * pack_factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_vals; i += 8) {
      *(threadgroup vec<T, 8>*)(dst + i) = *(thread vec<T, 8>*)(v + i);
    }
  }
};

// Each simdgroup computes SM rows and prefetches the next activation tile.
// Weights use two threadgroup buffers with one barrier between K steps.
template <
    typename T,
    int group_size,
    int bits,
    typename U = T,
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int SM = 16>
METAL_FUNC void qmm_t_nax_staged_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device U* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int k_end,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK == 64 && BN == 64, "64-column tiles");
  static_assert(SM > 0 && SM % 16 == 0 && BM % SM == 0, "Whole row fragments");
  static_assert(
      sizeof(T) == 2, "Two tiles fit threadgroup memory for 16-bit T");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int tile_size = BN * BK_padded;

  using loader_w_t = QuantizedBlockLoaderStaged<
      T,
      BN,
      BK,
      BK_padded,
      (BM / SM) * SIMD_SIZE,
      group_size,
      bits>;
  using frag_t = typename BaseNAXFrag::template dtype_frag_t<T>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  loader_w_t loader_w(
      (const device uint8_t*)w + y_col * K_w,
      scales + y_col * K_g,
      biases + y_col * K_g,
      K,
      simd_gid,
      simd_lid);

  // This simdgroup's rows and the two rows each thread holds in a fragment
  const int tile_rows = min(BM, M - y_row);
  constexpr int TM = SM / 16;
  const int sm = SM * int(simd_gid);
  const int rows = min(SM, tile_rows - sm);
  const short2 sc = BaseNAXFrag::get_coord();
  const device T* x0[TM];
  const device T* x1[TM];
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < TM; m++) {
    const int r0 = min(sm + m * 16 + int(sc.y), tile_rows - 1);
    const int r1 = min(sm + m * 16 + int(sc.y) + 8, tile_rows - 1);
    x0[m] = x + (y_row + r0) * static_cast<int64_t>(K) + sc.x;
    x1[m] = x + (y_row + r1) * static_cast<int64_t>(K) + sc.x;
  }

  NAXTile<float, TM, BN / 16> Dtile;
  Dtile.clear();
  vec<T, 4> a_cur[BK / 16][TM][2];
  vec<T, 4> a_next[BK / 16][TM][2];

  loader_w.fetch(0);
  loader_w.store(Ws);
  if (rows > 0) {
    STEEL_PRAGMA_UNROLL
    for (short s = 0; s < BK / 16; s++) {
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < TM; m++) {
        a_cur[s][m][0] = *(const device vec<T, 4>*)(x0[m] + s * 16);
        a_cur[s][m][1] = *(const device vec<T, 4>*)(x1[m] + s * 16);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0, it = 0; k < k_end; k += BK, it++) {
    const bool more = k + BK < k_end;
    if (more) {
      loader_w.fetch(k + BK);
    }
    if (rows > 0) {
      const int kn = more ? k + BK : k;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < TM; m++) {
          a_next[s][m][0] = *(const device vec<T, 4>*)(x0[m] + kn + s * 16);
          a_next[s][m][1] = *(const device vec<T, 4>*)(x1[m] + kn + s * 16);
        }
      }
      const threadgroup T* ws =
          Ws + (it & 1) * tile_size + sc.y * BK_padded + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BN / 16; j += 2) {
          const threadgroup T* p0 = ws + (j * 16) * BK_padded + s * 16;
          const threadgroup T* p1 = p0 + 16 * BK_padded;
          frag_t b0;
          frag_t b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<T, 4> u0 =
                *(const threadgroup vec<T, 4>*)(p0 + h * 8 * BK_padded);
            const vec<T, 4> u1 =
                *(const threadgroup vec<T, 4>*)(p1 + h * 8 * BK_padded);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              b0[h * 4 + c] = u0[c];
              b1[h * 4 + c] = u1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short m = 0; m < TM; m++) {
            frag_t a;
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              a[c] = a_cur[s][m][0][c];
              a[4 + c] = a_cur[s][m][1][c];
            }
            BaseNAXFrag::mma(
                Dtile.frag_at(m, j),
                Dtile.frag_at(m, j + 1),
                a,
                metal::bool_constant<false>{},
                b0,
                b1,
                metal::bool_constant<true>{});
          }
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        STEEL_PRAGMA_UNROLL
        for (short m = 0; m < TM; m++) {
          a_cur[s][m][0] = a_next[s][m][0];
          a_cur[s][m][1] = a_next[s][m][1];
        }
      }
    }
    if (more) {
      loader_w.store(Ws + ((it + 1) & 1) * tile_size);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (rows <= 0) {
    return;
  }
  y += (y_row + sm) * static_cast<int64_t>(N) + y_col;
  if (rows == SM) {
    Dtile.store(y, N);
  } else {
    Dtile.store_safe(y, N, short2(BN, rows));
  }
}

template <
    typename T,
    int group_size,
    int bits,
    bool aligned_N,
    typename U = T,
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
METAL_FUNC void qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device U* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int k_end,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < k_end; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_n_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  // Here w is [K, N]: packed and group-quantized along N, with row stride N.
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short sgp_sm = min(int(SM), M - (y_row + tm));

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TK, TN> Btile;

      volatile int compiler_barrier;

      if (sgp_sm == SM) {
        Atile.load(x + kk1, K);
      } else {
        Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
      }

      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm == SM) {
    Dtile.store(y + tm * N + tn, N);
  } else {
    Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm));
  }
}

template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale, // unused
    bool aligned_N,
    bool batched,
    int BM = 64,
    int BK = 32,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void affine_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  // WN == 1 selects the staged impl
  threadgroup T Ws[(WN == 1 ? 2 : 1) * BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  if constexpr (WN == 1) {
    static_assert(aligned_N, "Aligned output columns");
    qmm_t_nax_staged_impl<T, group_size, bits, T, BM, BK, BN, BM / WM>(
        w, scales, biases, x, y, Ws, K, N, M, K, tid, simd_gid, simd_lid);
  } else {
    qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, T, BM, BK, BN, WM, WN>(
        w, scales, biases, x, y, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
  }
}

template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale, // unimplemented
    bool aligned_N,
    bool batched, // unused
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void affine_qmm_t_nax_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_partition_size [[buffer(8)]],
    const constant int& split_k_partition_stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  threadgroup T Ws[(WN == 1 ? 2 : 1) * BN * BK_padded];

  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = (const device uint8_t*)w;
  wl += k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  if constexpr (WN == 1) {
    static_assert(aligned_N, "Aligned output columns");
    qmm_t_nax_staged_impl<T, group_size, bits, float, BM, BK, BN, BM / WM>(
        (const device uint32_t*)wl,
        scales,
        biases,
        x,
        y,
        Ws,
        K,
        N,
        M,
        k_partition_size,
        tid,
        simd_gid,
        simd_lid);
  } else {
    qmm_t_nax_tgp_impl<
        T,
        group_size,
        bits,
        aligned_N,
        float,
        BM,
        BK,
        BN,
        WM,
        WN>(
        (const device uint32_t*)wl,
        scales,
        biases,
        x,
        y,
        Ws,
        K,
        N,
        M,
        k_partition_size,
        tid,
        lid,
        simd_gid,
        simd_lid);
  }
}

template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale, // unused
    bool aligned_N, // unused
    bool batched,
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void affine_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale, // unused
    bool aligned_N,
    bool batched, // unused
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int WM = 2,
    int WN = 2>
[[kernel]] void affine_gather_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, T, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, K, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    bool has_global_scale, // unused
    bool transpose,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void affine_gather_qmm_rhs_nax(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& num_groups [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool rows_in_bounds = y_row + tm + SM <= M;
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  const bool sg_active = sgp_sm > 0;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + tm * K;

  // Prepare threadgroup loading operations
  thread loader_w_t loader_w(
      wl + group * stride_w,
      scales + group * stride_s,
      biases + group * stride_s,
      transpose ? K : N,
      Ws,
      simd_group_id,
      simd_lane_id);

  dispatch_bool(rows_in_bounds, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(
              transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        xn += BK;
        loader_w.next();
      }

      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load_safe(tile_w);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (k_remain - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Store results to device memory
      if constexpr (kAlignedN.value) {
        if (sgp_sm == SM) {
          Dtile.store(y + tm * N + tn, N);
        } else {
          Dtile.store_slice(
              y + tm * N + tn, N, short2(0, 0), short2(SN, sgp_sm));
        }
      } else {
        Dtile.store_slice(
            y + tm * N + tn, N, short2(0, 0), short2(sgp_sn, sgp_sm));
      }
    });
  });
}

// Quantize each group of a row of x as affine_int8_quantize_dequantize does,
// and store the signed 8-bit values. For each group it also writes the scale
// and the sum of the dequantized values. For 4-bit weights, each 8 values are
// stored as the even ones then the odd ones, to match QuantizedBlockLoaderInt.
// Each row of xs holds the float scales of all groups, then the sums as
// T hi and lo parts.
template <typename T, int group_size, int bits>
[[kernel]] void affine_quantize_rows(
    const device T* x [[buffer(0)]],
    device uint2* xq [[buffer(1)]],
    device float* xs [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr short lanes = group_size / 8;
  const int k = tid.x * 8;
  const bool valid = k < K;
  const int64_t offset = tid.y * static_cast<int64_t>(K) + k;

  // A group is either all valid or all past K, so the padding does not mix in
  const vec<T, 8> v =
      valid ? *(const device vec<T, 8>*)(x + offset) : vec<T, 8>(0);
  float amax = 0;
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < 8; j++) {
    amax = max(amax, abs(float(v[j])));
  }
  STEEL_PRAGMA_UNROLL
  for (short o = 1; o < lanes; o *= 2) {
    amax = max(amax, simd_shuffle_xor(amax, ushort(o)));
  }

  const float inv = amax > 0 ? 127.0f / amax : 0.0f;
  char4 lo, hi;
  int qsum = 0;
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < 4; j++) {
    const short l = bits == 4 ? 2 * j : j;
    const short h = bits == 4 ? 2 * j + 1 : 4 + j;
    lo[j] = static_cast<char>(rint(float(v[l]) * inv));
    hi[j] = static_cast<char>(rint(float(v[h]) * inv));
    qsum += lo[j] + hi[j];
  }
  STEEL_PRAGMA_UNROLL
  for (short o = 1; o < lanes; o *= 2) {
    qsum += simd_shuffle_xor(qsum, ushort(o));
  }
  if (!valid) {
    return;
  }

  xq[offset / 8] = uint2(as_type<uint>(lo), as_type<uint>(hi));
  if (simd_lid % lanes == 0) {
    const float scale = amax / 127.0f;
    const int K_g = K / group_size;
    const int g = k / group_size;
    xs += tid.y * 2 * static_cast<int64_t>(K_g);
    xs[g] = scale;
    const float sum = scale * qsum;
    const T sum_hi = static_cast<T>(sum);
    device T* xt = (device T*)(xs + K_g);
    xt[g] = sum_hi;
    xt[K_g + g] = static_cast<T>(sum - float(sum_hi));
  }
}

// Loads weights as QuantizedBlockLoaderStaged, but stores them as signed
// 8-bit values q - offset. Words 0, 2, 4, 6 of each 32 values (in the order
// of x from affine_quantize_rows) go first, then words 1, 3, 5, 7, to match
// the K slots of x in qmm_t_int_nax_staged_impl. Once per group it stores the
// scale of each row r at [group][(r % 16) / 4][r / 16][r % 4], so the 16
// columns of a thread in the 16 x 64 MMA are contiguous.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderInt {
  static_assert(bits == 4 || bits == 6 || bits == 8, "4, 6 and 8 bits");
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_words = n_reads * bytes_per_pack / 4;
  MLX_MTL_CONST short GK = group_size < BCOLS ? group_size : BCOLS;
  MLX_MTL_CONST short GS = BCOLS / GK;
  MLX_MTL_CONST int offset = 1 << (bits - 1);

  static_assert(
      BCOLS_PACKED * BROWS >= tgp_size &&
          n_words * 4 == n_reads * bytes_per_pack &&
          (bits != 6 || n_words % 3 == 0),
      "Every thread loads whole words");
  static_assert(
      n_reads * pack_factor == 32 && 32 <= GK,
      "A thread's values are one 32-value block of one group");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t* src;
  const device T* scales;
  uint32_t raw[n_words];
  T scale;

  QuantizedBlockLoaderInt(
      const device uint8_t* src_,
      const device T* scales_,
      const int src_ld,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size) {}

  // k: first column of the step
  void fetch(const int k) thread {
    auto p = (const device uint32_t*)(src + k * bytes_per_pack / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) {
      raw[i] = p[i];
    }
    scale = scales[(k + bj * pack_factor) / group_size];
  }

  void store(threadgroup int8_t* dst, threadgroup T* sb) const thread {
    dst += bi * dst_ld + bj * pack_factor;
    // The 8 words of the block, in the order of x
    uint q[8];
    if constexpr (bits == 4) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        const uint u = raw[i];
        q[2 * i] = ((u & 0x0f0f0f0f) + 0x78787878) ^ 0x80808080;
        q[2 * i + 1] = (((u >> 4) & 0x0f0f0f0f) + 0x78787878) ^ 0x80808080;
      }
    } else if constexpr (bits == 6) {
      // Three words hold 16 values. Spread each 24 bits to 4 bytes.
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 2; i++) {
        const uint w0 = raw[3 * i];
        const uint w1 = raw[3 * i + 1];
        const uint w2 = raw[3 * i + 2];
        const uint seg[4] = {
            w0 & 0xffffff,
            (w0 >> 24) | ((w1 & 0xffff) << 8),
            (w1 >> 16) | ((w2 & 0xff) << 16),
            w2 >> 8};
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          const uint v = seg[j];
          const uint b = (v & 0x3f) | ((v << 2) & 0x3f00) |
              ((v << 4) & 0x3f0000) | ((v << 6) & 0x3f000000);
          q[4 * i + j] = (b + 0x60606060) ^ 0x80808080;
        }
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 8; i++) {
        q[i] = raw[i] ^ 0x80808080;
      }
    }
    *(threadgroup uint4*)dst = uint4(q[0], q[2], q[4], q[6]);
    *(threadgroup uint4*)(dst + 16) = uint4(q[1], q[3], q[5], q[7]);

    const short col = bj * pack_factor;
    if (col % GK == 0) {
      sb[col / GK * BROWS + (bi % 16) / 4 * 16 + bi / 16 * 4 + bi % 4] = scale;
    }
  }
};

// qmm_t_nax_staged_impl with 8-bit integer MMAs. x comes from
// affine_quantize_rows as x = sx * c, with rows padded to a multiple of BM.
// The weights are w = s * e + f, so when a group ends it adds
// sx * s * sum(c * e). At the end one MMA adds f * sum(x) of all groups.
template <
    typename T,
    int group_size,
    int bits,
    typename U = T,
    int BM = 64,
    int BK = 64,
    int BN = 64,
    int SM = 16>
METAL_FUNC void qmm_t_int_nax_staged_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    device int8_t* x,
    const device float* xs,
    device U* y,
    threadgroup int8_t* Ws,
    threadgroup T* Sb,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const int k_end,
    const int g0,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK == 64 && BN == 64 && SM == 16, "64-column, 16-row tiles");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16;
  constexpr int tile_size = BN * BK_padded;
  constexpr int TK = BK / 32;

  using loader_w_t = QuantizedBlockLoaderInt<
      T,
      BN,
      BK,
      BK_padded,
      (BM / SM) * SIMD_SIZE,
      group_size,
      bits>;
  constexpr int GK = loader_w_t::GK;
  constexpr int GS = loader_w_t::GS;

  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  loader_w_t loader_w(
      (const device uint8_t*)w + y_col * (K * bytes_per_pack / pack_factor),
      scales + y_col * K_g,
      K,
      simd_gid,
      simd_lid);

  // This simdgroup's rows, and the two rows each thread holds
  const int sm = SM * int(simd_gid);
  const int rows = min(SM, M - y_row - sm);
  const short2 sc = BaseNAXFrag::get_coord();
  const short fm = sc.y;
  const short fn = sc.x;
  const device uint2* x0 =
      (const device uint2*)(x + (y_row + sm + fm) * static_cast<int64_t>(K) +
                            2 * fn);
  const device uint2* x1 = x0 + K;
  // g0: first group of the K range
  const device float* xs_sg = xs + (y_row + sm) * 2 * static_cast<int64_t>(K_g);
  xs = xs_sg + 2 * fm * K_g + g0;

  // 16 x 64 x 32 MMAs with x in registers and the weights in threadgroup
  // memory. A thread holds 8 values of K in each row of x: the low word in
  // slots fn to fn + 3 and the high word in 16 + fn to 16 + fn + 3.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      BN,
      32,
      false,
      true,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  using b_extents = extents<int, 32, BN>;
  using b_tensor = tensor<threadgroup int8_t, b_extents, tensor_inline>;
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int>();
  // Integer sums of the current group, as the fragments of Dtile
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      b_tensor,
      int>();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < 32; i++) {
    ct_c[i] = 0;
  }

  NAXTile<float, 1, BN / 16> Dtile;
  Dtile.clear();
  // x of the current 32-column block, rows fm and fm + 8
  uint2 a_cur[2];

  loader_w.fetch(0);
  loader_w.store(Ws, Sb);
  if (rows > 0) {
    a_cur[0] = x0[0];
    a_cur[1] = x1[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0, it = 0; k < k_end; k += BK, it++) {
    const bool more = k + BK < k_end;
    if (more) {
      loader_w.fetch(k + BK);
    }
    if (rows > 0) {
      threadgroup int8_t* ws = Ws + (it & 1) * tile_size;
      const threadgroup T* sb = Sb + (it & 1) * GS * BN;
      // x scale of rows fm and fm + 8, for each group of the step
      float xg[GS][2];
      STEEL_PRAGMA_UNROLL
      for (short gi = 0; gi < GS; gi++) {
        const int g = (k + gi * GK) / group_size;
        xg[gi][0] = xs[g];
        xg[gi][1] = xs[16 * K_g + g];
      }
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < TK; s++) {
        // Load the next block, in this step or the next one
        const int kn = s + 1 < TK ? k + 32 * (s + 1) : (more ? k + BK : k);
        const uint2 a_next[2] = {x0[kn / 8], x1[kn / 8]};
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; r++) {
          const char4 lo = as_type<char4>(a_cur[r].x);
          const char4 hi = as_type<char4>(a_cur[r].y);
          STEEL_PRAGMA_UNROLL
          for (short c = 0; c < 4; c++) {
            ct_a[4 * r + c] = lo[c];
            ct_a[8 + 4 * r + c] = hi[c];
          }
        }
        b_tensor tB(ws + 32 * s, b_extents(), array<int, 2>{1, BK_padded});
        gemm_op.run(ct_a, tB, ct_c);

        // The group ends with this 32-column block
        if ((k + 32 * (s + 1)) % group_size == 0) {
          const short gi = (32 * s) / GK;
          // Scales of columns 16 * n + fn to 16 * n + fn + 3, for each n
          const threadgroup vec<T, 8>* sv =
              (const threadgroup vec<T, 8>*)(sb + gi * BN + 4 * fn);
          const vec<T, 8> sw[2] = {sv[0], sv[1]};
          STEEL_PRAGMA_UNROLL
          for (short n = 0; n < BN / 16; n++) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; e++) {
              const float p =
                  xg[gi][e / 4] * float(sw[n / 2][4 * (n % 2) + e % 4]);
              const float acc = Dtile.frag_at(0, n)[e];
              Dtile.frag_at(0, n)[e] = fma(float(ct_c[8 * n + e]), p, acc);
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 32; i++) {
            ct_c[i] = 0;
          }
        }
        a_cur[0] = a_next[0];
        a_cur[1] = a_next[1];
      }
    }
    if (more) {
      loader_w.store(
          Ws + ((it + 1) & 1) * tile_size, Sb + ((it + 1) & 1) * GS * BN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (rows <= 0) {
    return;
  }

  // f * sum(x) of all groups, as b * sum(x) + offset * s * sum(x), with
  // sum(x) in hi and lo parts. T x T keeps the destination in Dtile order.
  {
    constexpr auto desc_f = mpp::tensor_ops::matmul2d_descriptor(
        16,
        BN,
        static_cast<int>(dynamic_extent),
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc_f, metal::execution_simdgroup> f_op;
    const int groups = k_end / group_size;
    using tt = tensor<device T, dextents<int, 2>, tensor_inline>;
    device T* hi = (device T*)(xs_sg + K_g) + g0;
    tt tHi(hi, dextents<int, 2>(groups, 16), array<int, 2>{1, 4 * K_g});
    tt tLo(hi + K_g, dextents<int, 2>(groups, 16), array<int, 2>{1, 4 * K_g});
    tt tB(
        (device T*)biases + y_col * K_g,
        dextents<int, 2>(groups, BN),
        array<int, 2>{1, K_g});
    tt tS(
        (device T*)scales + y_col * K_g,
        dextents<int, 2>(groups, BN),
        array<int, 2>{1, K_g});
    auto ct_f =
        f_op.template get_destination_cooperative_tensor<tt, tt, float>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; i++) {
      ct_f[i] = Dtile.frag_at(0, i / 8)[i % 8];
    }
    f_op.run(tHi, tB, ct_f);
    f_op.run(tLo, tB, ct_f);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; i++) {
      Dtile.frag_at(0, i / 8)[i % 8] = ct_f[i];
      ct_f[i] = 0;
    }
    f_op.run(tHi, tS, ct_f);
    f_op.run(tLo, tS, ct_f);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 32; i++) {
      const float acc = Dtile.frag_at(0, i / 8)[i % 8];
      Dtile.frag_at(0, i / 8)[i % 8] =
          fma(float(loader_w_t::offset), ct_f[i], acc);
    }
  }
  y += (y_row + sm) * static_cast<int64_t>(N) + y_col;
  if (rows == SM) {
    Dtile.store(y, N);
  } else {
    Dtile.store_safe(y, N, short2(BN, rows));
  }
}

template <typename T, int group_size, int bits, int BM = 64, int WM = 4>
[[kernel]] void affine_qmm_t_int_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device int8_t* x [[buffer(3)]],
    const device float* xs [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& M [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  constexpr int BN = 64;
  constexpr int GS = BK / (group_size < BK ? group_size : BK);
  threadgroup int8_t Ws[2 * BN * (BK + 16)];
  threadgroup T Sb[2 * GS * BN] __attribute__((aligned(16)));
  qmm_t_int_nax_staged_impl<T, group_size, bits, T, BM, BK, BN, BM / WM>(
      w,
      scales,
      biases,
      x,
      xs,
      y,
      Ws,
      Sb,
      K,
      N,
      M,
      K,
      0,
      tid,
      simd_gid,
      simd_lid);
}

template <typename T, int group_size, int bits, int BM = 64, int WM = 4>
[[kernel]] void affine_qmm_t_int_nax_splitk(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    device int8_t* x [[buffer(3)]],
    const device float* xs [[buffer(4)]],
    device float* y [[buffer(5)]],
    const constant int& K [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& M [[buffer(8)]],
    const constant int& k_partition_size [[buffer(9)]],
    const constant int& split_k_partition_stride [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64;
  constexpr int BN = 64;
  constexpr int GS = BK / (group_size < BK ? group_size : BK);
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  threadgroup int8_t Ws[2 * BN * (BK + 16)];
  threadgroup T Sb[2 * GS * BN] __attribute__((aligned(16)));

  const int k_start = tid.z * k_partition_size;
  x += k_start;
  auto wl = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor;
  scales += k_start / group_size;
  biases += k_start / group_size;
  y += tid.z * static_cast<int64_t>(split_k_partition_stride);

  qmm_t_int_nax_staged_impl<T, group_size, bits, float, BM, BK, BN, BM / WM>(
      (const device uint32_t*)wl,
      scales,
      biases,
      x,
      xs,
      y,
      Ws,
      Sb,
      K,
      N,
      M,
      k_partition_size,
      k_start / group_size,
      tid,
      simd_gid,
      simd_lid);
}
