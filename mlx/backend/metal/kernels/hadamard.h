// Copyright © 2024 Apple Inc.
#include <metal_common>
#include <metal_compute>

#include "mlx/backend/metal/kernels/steel/defines.h"

using namespace metal;

// Thread local Hadamard transform for 2^R
template <short R>
METAL_FUNC void radix_func(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < logR; s++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

template <typename T, int N, int max_radix, int read_width>
[[kernel]] void hadamard_n(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const float& scale,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Compute a Hadamard transform of size N = 2^k
  //
  // Equivalent to:
  //    from scipy.linalg import hadamard
  //    y = hadamard(len(x)) @ x

  constexpr short num_threads = N / max_radix;
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = __builtin_ctz(max_radix);
  constexpr short num_steps = logN / logR;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << (logFinal);

  int batch_idx = elem.x * N;
  short i = elem.y;

  threadgroup T buf[N];

  // Read values from device
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix / read_width; j++) {
    short index = j * read_width * num_threads + i * read_width;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      buf[index + r] = in[batch_idx + index + r];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float x[max_radix];
  short h = 1;

  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < num_steps; s++) {
    short k = i & (h - 1);
    short j = ((i - k) << logR) + k;

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      x[r] = buf[j + h * r];
    }

    radix_func<max_radix>(x);

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      buf[j + h * r] = T(x[r]);
    }

    h <<= logR;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Do the final radix
  // e.g. max_radix = 16
  //      N = 1024 = 16 * 16 * 4
  if (final_radix > 1) {
    // Each thread does multiple butterflies
    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < max_radix / final_radix; t++) {
      short index = i + t * num_threads;
      short k = index & (h - 1);
      short j = ((index - k) << logFinal) + k;
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        x[r] = buf[j + h * r];
      }

      radix_func<final_radix>(x);

      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        buf[j + h * r] = T(x[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write values to device
  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix / read_width; j++) {
    short index = j * read_width * num_threads + i * read_width;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      out[batch_idx + index + r] = T(buf[index + r] * scale);
    }
  }
}

template <typename T, int N, int M, int read_width>
[[kernel]] void hadamard_m(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const float& scale,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Compute a Hadamard transform of size M
  // using a naive O(M^2) codelet.
  //
  // This kernel is the second stage in the computation
  // of a Hadamard transform of size M*N where N = 2^k.

  int index = elem.x * grid.y + elem.y;
  short i = index % (N / read_width);
  int batch_idx = index / (N / read_width) * M * N;

  float x[read_width][M];
  STEEL_PRAGMA_UNROLL
  for (short c = 0; c < M; c++) {
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      x[r][c] = in[batch_idx + c * N + i * read_width + r];
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < read_width; r++) {
    // This function is JIT compiled for M
    // using the Hadamard matrix strings in `metal/hadamard.cpp`
    hadamard_radix_m(x[r]);
  }

  // Write back to device
  STEEL_PRAGMA_UNROLL
  for (short c = 0; c < M; c++) {
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      out[batch_idx + c * N + i * read_width + r] = T(x[r][c] * scale);
    }
  }
}
