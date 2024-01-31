// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T, int bits, int group_size>
void _qmm(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const int Ng = N / group_size;
  const int Nw = N / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    std::fill(result, result + N, 0);

    for (int k = 0; k < K; k++) {
      T* result_local = result;
      T xi = *x++;

      for (int n = 0; n < N; n += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;
        for (int ng = 0; ng < packs_in_group; ng++) {
          uint32_t wi = *w_local++;

#pragma clang loop unroll(full)
          for (int p = 0; p < pack_factor; p++) {
            (*result_local++) +=
                xi * (scale * static_cast<T>(wi & bitmask) + bias);
            wi >>= bits;
          }
        }
      }
    }

    result += N;
  }
}

template <typename T, int bits, int group_size>
void _qmm_t(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K) {
  constexpr int bitmask = (1 << bits) - 1;
  constexpr int pack_factor = 32 / bits;
  constexpr int packs_in_group = group_size / pack_factor;
  const int Kg = K / group_size;
  const int Kw = K / pack_factor;

  for (int m = 0; m < M; m++) {
    const uint32_t* w_local = w;
    const T* scales_local = scales;
    const T* biases_local = biases;

    for (int n = 0; n < N; n++) {
      const T* x_local = x;
      T sum = 0;
      for (int k = 0; k < K; k += group_size) {
        T scale = *scales_local++;
        T bias = *biases_local++;

        for (int kw = 0; kw < packs_in_group; kw++) {
          uint32_t wi = *w_local++;

#pragma clang loop unroll(full)
          for (int p = 0; p < pack_factor; p++) {
            sum += (*x_local++) * (scale * static_cast<T>(wi & bitmask) + bias);
            wi >>= bits;
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
void _qmm_dispatch_typed(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int M,
    int N,
    int K,
    int group_size,
    int bits,
    bool transposed_w) {
  switch (bits) {
    case 2: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 2, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 2, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 2, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 2, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
    case 4: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 4, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 4, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 4, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 4, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
    case 8: {
      switch (group_size) {
        case 32:
          if (transposed_w) {
            return _qmm_t<T, 8, 32>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 32>(result, x, w, scales, biases, M, N, K);
          }
        case 64:
          if (transposed_w) {
            return _qmm_t<T, 8, 64>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 64>(result, x, w, scales, biases, M, N, K);
          }
        case 128:
          if (transposed_w) {
            return _qmm_t<T, 8, 128>(result, x, w, scales, biases, M, N, K);
          } else {
            return _qmm<T, 8, 128>(result, x, w, scales, biases, M, N, K);
          }
      }
    }
  }
  std::ostringstream msg;
  msg << "Quantization type not supported. Provided bits=" << bits
      << " and group_size=" << group_size
      << ". The supported options are bits in "
      << "{2, 4, 8} and group_size in {64, 128}.";
  throw std::invalid_argument(msg.str());
}

void _qmm_dispatch(
    array out,
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int bits,
    int group_size,
    bool transposed_w) {
  int K = x.shape(-1);
  int M = x.size() / K;
  int N = out.shape(-1);

  switch (x.dtype()) {
    case float32:
      _qmm_dispatch_typed<float>(
          out.data<float>(),
          x.data<float>(),
          w.data<uint32_t>(),
          scales.data<float>(),
          biases.data<float>(),
          M,
          N,
          K,
          bits,
          group_size,
          transposed_w);
      break;
    case float16:
      _qmm_dispatch_typed<float16_t>(
          out.data<float16_t>(),
          x.data<float16_t>(),
          w.data<uint32_t>(),
          scales.data<float16_t>(),
          biases.data<float16_t>(),
          M,
          N,
          K,
          bits,
          group_size,
          transposed_w);
      break;
    case bfloat16:
      _qmm_dispatch_typed<bfloat16_t>(
          out.data<bfloat16_t>(),
          x.data<bfloat16_t>(),
          w.data<uint32_t>(),
          scales.data<bfloat16_t>(),
          biases.data<bfloat16_t>(),
          M,
          N,
          K,
          bits,
          group_size,
          transposed_w);
      break;
    default:
      throw std::invalid_argument(
          "[quantized_matmul] only floating types are supported");
  }
}

} // namespace

void QuantizedMatmul::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  auto& x_pre = inputs[0];
  auto& w_pre = inputs[1];
  auto& scales_pre = inputs[2];
  auto& biases_pre = inputs[3];

  auto ensure_row_contiguous = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return arr;
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      return arr_copy;
    }
  };

  auto x = ensure_row_contiguous(x_pre);
  auto w = ensure_row_contiguous(w_pre);
  auto scales = ensure_row_contiguous(scales_pre);
  auto biases = ensure_row_contiguous(biases_pre);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  _qmm_dispatch(out, x, w, scales, biases, group_size_, bits_, transpose_);
}

} // namespace mlx::core
