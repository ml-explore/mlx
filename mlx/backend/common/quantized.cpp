// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, int width, int groups>
void _qmm_t(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int bitmask = (1 << width) - 1;
  constexpr int pack_factor = 32 / width;
  constexpr int packs_in_group = groups / pack_factor;
  const int Kg = K / groups;
  const int Kw = K / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    for (int n = 0; n < N; n++) {
      const T* x_local = x;
      T sum = 0;
      for (int k = 0; k < K; k += groups) {
        T scale = *scales_local++;
        T bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw++) {
          uint32_t wi = *w_local++;

#pragma clang loop unroll(full)
          for (int p = 0; p < pack_factor; p++) {
            sum += (*x_local++) * (scale * static_cast<T>(wi & bitmask) + bias);
            wi >>= width;
          }
        }
      }
      *result = sum;
      result++;
    }

    x += K;
  }
}

template <typename T>
void _qmm_t_dispatch_typed(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K,
    int width,
    int groups) {
  switch (width) {
    case 2: {
      switch (groups) {
        case 64:
          return _qmm_t<T, 2, 64>(result, x, w, scales, biases, M, N, K);
        case 128:
          return _qmm_t<T, 2, 128>(result, x, w, scales, biases, M, N, K);
      }
    }
    case 4: {
      switch (groups) {
        case 64:
          return _qmm_t<T, 4, 64>(result, x, w, scales, biases, M, N, K);
        case 128:
          return _qmm_t<T, 4, 128>(result, x, w, scales, biases, M, N, K);
      }
    }
    case 8: {
      switch (groups) {
        case 64:
          return _qmm_t<T, 8, 64>(result, x, w, scales, biases, M, N, K);
        case 128:
          return _qmm_t<T, 8, 128>(result, x, w, scales, biases, M, N, K);
      }
    }
  }
  std::ostringstream msg;
  msg << "Quantization type not supported. Provided bit width=" << width
      << " and groups=" << groups << ". The supported options are width in "
      << "{2, 4, 8} and groups in {64, 128}.";
  throw std::invalid_argument(msg.str());
}

void _qmm_t_dispatch(
    array out,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int width,
    int groups) {
  int K = x.shape(-1);
  int M = x.size() / K;
  int N = w.shape(1);

  switch (x.dtype()) {
    case float32:
      _qmm_t_dispatch_typed<float>(
          out.data<float>(),
          x.data<float>(),
          w.data<uint32_t>(),
          scales.data<float>(),
          biases.data<float>(),
          M,
          N,
          K,
          width,
          groups);
      break;
    case float16:
      _qmm_t_dispatch_typed<float16_t>(
          out.data<float16_t>(),
          x.data<float16_t>(),
          w.data<uint32_t>(),
          scales.data<float16_t>(),
          biases.data<float16_t>(),
          M,
          N,
          K,
          width,
          groups);
      break;
    case bfloat16:
      _qmm_t_dispatch_typed<bfloat16_t>(
          out.data<bfloat16_t>(),
          x.data<bfloat16_t>(),
          w.data<uint32_t>(),
          scales.data<bfloat16_t>(),
          biases.data<bfloat16_t>(),
          M,
          N,
          K,
          width,
          groups);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

} // namespace

void QuantizedMatmul::eval(const std::vector<array>& inputs, array& out) {
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

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  _qmm_t_dispatch(out, x, w, scales, biases, width_, groups_);
}

} // namespace mlx::core
