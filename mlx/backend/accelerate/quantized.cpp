// Copyright © 2023 Apple Inc.

#include <cassert>

#include <simd/vector.h>

#include "mlx/primitives.h"

namespace mlx::core {

namespace {

void _qmm_t_4_64(
    float* result,
    const float* x,
    const uint32_t* w,
    const float* scales,
    const float* biases,
    int M,
    int N,
    int K) {
  constexpr int bits = 4;
  constexpr int group_size = 64;
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const int Kg = K / group_size;
  const int Kw = K / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const float* scales_local = scales;
    const float* biases_local = biases;

    for (int n = 0; n < N; n++) {
      const simd_float16* x_local = (simd_float16*)x;
      simd_float16 sum = 0;
      for (int k = 0; k < K; k += group_size) {
        float scale = *scales_local++;
        float bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw += 2) {
          // TODO: vectorize this properly
          simd_uint16 wi;
          for (int e = 0; e < 2; e++) {
            uint32_t wii = *w_local++;
            for (int p = 0; p < 8; p++) {
              wi[e * 8 + p] = wii & bitmask;
              wii >>= bits;
            }
          }
          simd_float16 wf = simd_float(wi);
          wf *= scale;
          wf += bias;

          sum += (*x_local) * wf;
          x_local++;
        }
      }

      *result = simd_reduce_add(sum);
      result++;
    }

    x += K;
  }
}

} // namespace

void QuantizedMatmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& scales = inputs[2];
  auto& biases = inputs[3];

  if (w.strides()[0] != 1) {
    throw std::runtime_error("The quantized weight should be transposed");
  }

  if (!x.flags().row_contiguous || !scales.flags().row_contiguous ||
      !biases.flags().row_contiguous) {
    throw std::runtime_error("x, scales and biases should be row contiguous.");
  }

  if (x.dtype() == float32 && bits_ == 4 && group_size_ == 64) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    int K = x.shape(-1);
    int M = x.size() / K;
    int N = w.shape(1);
    _qmm_t_4_64(
        out.data<float>(),
        x.data<float>(),
        w.data<uint32_t>(),
        scales.data<float>(),
        biases.data<float>(),
        M,
        N,
        K);
  } else {
    eval(inputs, out);
  }
}

} // namespace mlx::core
