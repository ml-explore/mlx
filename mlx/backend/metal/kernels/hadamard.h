#include <metal_common>
#include <metal_compute>

#include "mlx/backend/metal/kernels/steel/defines.h"

using namespace metal;

// Plan
// - Optimize all 2^k up to 16384 (fp16)
// - Arbitrary 2^r radix
// - Threadgroup batching
// - Add back up op based approach for arbitrary H
// - Add 2 stage h12/h20/h28/h40 kernel

// template <typename T, int M>
// [[kernel]] void hadamard_arbitrary(
//     const device T* in [[buffer(0)]],
//     device T* out [[buffer(1)]],
//     uint3 elem [[thread_position_in_grid]],
//     uint3 grid [[threads_per_grid]]) {
//   constexpr short radix = 4;

//   int batch_idx = elem.x * M;
//   short i = elem.y;

//   T x[radix][M];
//   STEEL_PRAGMA_UNROLL
//   for (short c = 0; c < M; c++) {
//     STEEL_PRAGMA_UNROLL
//     for (short r = 0; r < radix; r++) {
//       x[r][c] = in[batch_idx + c*N + i*radix + r];
//     }
//   }

//   for (short r = 0; r < radix; r++) {
//     hadamard_m(x[r]);
//   }

//   // Write back to device
//   STEEL_PRAGMA_UNROLL
//   for (short c = 0; c < M; c++) {
//     STEEL_PRAGMA_UNROLL
//     for (short r = 0; r < radix; r++) {
//       out[batch_idx + i*radix + r] = x[r];
//     }
//   }
// }

template <typename T, short R>
METAL_FUNC void radix_func(thread T* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < logR; s++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      T a = x[j];
      T b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}

// This will get us to full bandwidth for large powers of 2
// Then add threadgroup batching for small powers
// Then add the non-power of 2 options as a second stage
// This limits us to half bandwidth but that seems reasonable given the other
// measurements
template <typename T, int N>
[[kernel]] void hadamard(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  constexpr short read_width = 4;
  constexpr short max_radix = 16;
  constexpr short num_threads = N / max_radix;

  int batch_idx = elem.x * N;
  short i = elem.y;

  threadgroup T buf[N];

  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix / read_width; j++) {
    short index = j * read_width * num_threads + i * read_width;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      buf[index + r] = in[batch_idx + index + r];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Perform the hadamard transform
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = __builtin_ctz(max_radix);
  constexpr short num_steps = logN / logR;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << (logFinal);

  T x[max_radix];
  short h = 1;

  STEEL_PRAGMA_UNROLL
  for (short s = 0; s < num_steps; s++) {
    short k = i & (h - 1);
    short j = ((i - k) << logR) + k;

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      x[r] = buf[j + h * r];
    }

    radix_func<T, max_radix>(x);

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < max_radix; r++) {
      buf[j + h * r] = x[r];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    h <<= logR;
  }

  // Do the final radix
  // We need to do multiple per thread
  // e.g. max_radix = 16
  //      N = 1024 = 16 * 16 * 4
  if (final_radix > 1) {
    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < max_radix / final_radix; t++) {
      short index = i + t * num_threads;
      short k = index & (h - 1);
      short j = ((index - k) << logFinal) + k;
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        x[r] = buf[j + h * r];
      }

      radix_func<T, final_radix>(x);

      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < final_radix; r++) {
        buf[j + h * r] = x[r];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < max_radix / read_width; j++) {
    short index = j * read_width * num_threads + i * read_width;
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < read_width; r++) {
      out[batch_idx + index + r] = buf[index + r];
    }
  }
}
