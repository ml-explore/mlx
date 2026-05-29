// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/quantized_highway.h"

#include "hwy/highway.h"

namespace mlx::core {

#define MLX_HIGHWAY_CONCAT2(a, b) a##b
#define MLX_HIGHWAY_CONCAT(a, b) MLX_HIGHWAY_CONCAT2(a, b)

using DequantRowFn =
    void (*)(const uint32_t*, const float*, const float*, float*, int, int);
using QuantizeActivationInt8Fn = void (*)(
    const void*,
    QuantizedHighwayDType,
    int,
    int,
    int8_t*,
    float*,
    float*);
using QmmTInt8RowFn = void (*)(
    float*,
    const int8_t*,
    const float*,
    const float*,
    const uint32_t*,
    const void*,
    const void*,
    QuantizedHighwayDType,
    int,
    int,
    int,
    int,
    int);
using QmmTInt8Fn = bool (*)(
    void*,
    const void*,
    const uint32_t*,
    const void*,
    const void*,
    QuantizedHighwayDType,
    int,
    int,
    int,
    int,
    int);
using FpQmmTHighwayRowFn = void (*)(
    void*,
    const void*,
    const uint32_t*,
    const uint8_t*,
    QuantizedHighwayDType,
    int,
    int,
    int,
    int,
    int,
    float,
    const float*,
    const float*);
using FpQmmTHighwayFn = void (*)(
    void*,
    const void*,
    const uint32_t*,
    const uint8_t*,
    QuantizedHighwayDType,
    int,
    int,
    int,
    int,
    int,
    float,
    const float*,
    const float*);
#define MLX_DECLARE_QUANTIZED_TARGET(suffix)                                  \
  void MLX_HIGHWAY_CONCAT(dequant_row_highway_4bit, suffix)(                  \
      const uint32_t*, const float*, const float*, float*, int, int);         \
  void MLX_HIGHWAY_CONCAT(dequant_row_highway_8bit, suffix)(                  \
      const uint32_t*, const float*, const float*, float*, int, int);         \
  void MLX_HIGHWAY_CONCAT(quantize_activation_int8_highway, suffix)(          \
      const void*, QuantizedHighwayDType, int, int, int8_t*, float*, float*); \
  void MLX_HIGHWAY_CONCAT(qmm_t_int8_highway_row, suffix)(                    \
      float*,                                                                 \
      const int8_t*,                                                          \
      const float*,                                                           \
      const float*,                                                           \
      const uint32_t*,                                                        \
      const void*,                                                            \
      const void*,                                                            \
      QuantizedHighwayDType,                                                  \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int);                                                                   \
  bool MLX_HIGHWAY_CONCAT(qmm_t_int8_highway, suffix)(                        \
      void*,                                                                  \
      const void*,                                                            \
      const uint32_t*,                                                        \
      const void*,                                                            \
      const void*,                                                            \
      QuantizedHighwayDType,                                                  \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int);                                                                   \
  void MLX_HIGHWAY_CONCAT(fp_qmm_t_highway_row, suffix)(                      \
      void*,                                                                  \
      const void*,                                                            \
      const uint32_t*,                                                        \
      const uint8_t*,                                                         \
      QuantizedHighwayDType,                                                  \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      const float*,                                                           \
      const float*);                                                          \
  void MLX_HIGHWAY_CONCAT(fp_qmm_t_highway, suffix)(                          \
      void*,                                                                  \
      const void*,                                                            \
      const uint32_t*,                                                        \
      const uint8_t*,                                                         \
      QuantizedHighwayDType,                                                  \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      int,                                                                    \
      float,                                                                  \
      const float*,                                                           \
      const float*)

MLX_DECLARE_QUANTIZED_TARGET(_avx2);
MLX_DECLARE_QUANTIZED_TARGET(_sse4);
MLX_DECLARE_QUANTIZED_TARGET(_ssse3);
MLX_DECLARE_QUANTIZED_TARGET(_sse2);

#undef MLX_DECLARE_QUANTIZED_TARGET

namespace {

struct QuantizedHighwayDispatch {
  DequantRowFn dequant4;
  DequantRowFn dequant8;
  QuantizeActivationInt8Fn quantize_activation_int8;
  QmmTInt8RowFn qmm_t_int8_row;
  QmmTInt8Fn qmm_t_int8;
  FpQmmTHighwayRowFn fp_qmm_t_row;
  FpQmmTHighwayFn fp_qmm_t;
};

#define MLX_QUANTIZED_DISPATCH(suffix)                                \
  QuantizedHighwayDispatch {                                          \
    MLX_HIGHWAY_CONCAT(dequant_row_highway_4bit, suffix),             \
        MLX_HIGHWAY_CONCAT(dequant_row_highway_8bit, suffix),         \
        MLX_HIGHWAY_CONCAT(quantize_activation_int8_highway, suffix), \
        MLX_HIGHWAY_CONCAT(qmm_t_int8_highway_row, suffix),           \
        MLX_HIGHWAY_CONCAT(qmm_t_int8_highway, suffix),               \
        MLX_HIGHWAY_CONCAT(fp_qmm_t_highway_row, suffix),             \
        MLX_HIGHWAY_CONCAT(fp_qmm_t_highway, suffix)                  \
  }

const QuantizedHighwayDispatch& quantized_dispatch() {
  static const QuantizedHighwayDispatch dispatch = [] {
    const int64_t targets = hwy::SupportedTargets();
    if (targets & HWY_AVX2) {
      return MLX_QUANTIZED_DISPATCH(_avx2);
    }
    if (targets & HWY_SSE4) {
      return MLX_QUANTIZED_DISPATCH(_sse4);
    }
    if (targets & HWY_SSSE3) {
      return MLX_QUANTIZED_DISPATCH(_ssse3);
    }
    return MLX_QUANTIZED_DISPATCH(_sse2);
  }();
  return dispatch;
}

#undef MLX_QUANTIZED_DISPATCH
#undef MLX_HIGHWAY_CONCAT
#undef MLX_HIGHWAY_CONCAT2

} // namespace

void dequant_row_highway_4bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  quantized_dispatch().dequant4(
      w_row, scales_row, biases_row, out, group_size, K);
}

void dequant_row_highway_8bit(
    const uint32_t* w_row,
    const float* scales_row,
    const float* biases_row,
    float* out,
    int group_size,
    int K) {
  quantized_dispatch().dequant8(
      w_row, scales_row, biases_row, out, group_size, K);
}

void quantize_activation_int8_highway(
    const void* x,
    QuantizedHighwayDType dtype,
    int K,
    int group_size,
    int8_t* x_q,
    float* x_scales,
    float* x_group_sums) {
  quantized_dispatch().quantize_activation_int8(
      x, dtype, K, group_size, x_q, x_scales, x_group_sums);
}

void qmm_t_int8_highway_row(
    float* result,
    const int8_t* x_q,
    const float* x_scales,
    const float* x_group_sums,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K) {
  quantized_dispatch().qmm_t_int8_row(
      result,
      x_q,
      x_scales,
      x_group_sums,
      w,
      scales,
      biases,
      dtype,
      bits,
      group_size,
      n_start,
      n_end,
      K);
}

bool qmm_t_int8_highway(
    void* result,
    const void* x,
    const uint32_t* w,
    const void* scales,
    const void* biases,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K) {
  return quantized_dispatch().qmm_t_int8(
      result, x, w, scales, biases, dtype, bits, group_size, M, N, K);
}

void fp_qmm_t_highway_row(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int n_start,
    int n_end,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  quantized_dispatch().fp_qmm_t_row(
      result,
      x,
      w,
      scales,
      dtype,
      bits,
      group_size,
      n_start,
      n_end,
      K,
      scale_factor,
      fp4_lut,
      fp8_lut);
}

void fp_qmm_t_highway(
    void* result,
    const void* x,
    const uint32_t* w,
    const uint8_t* scales,
    QuantizedHighwayDType dtype,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    float scale_factor,
    const float* fp4_lut,
    const float* fp8_lut) {
  quantized_dispatch().fp_qmm_t(
      result,
      x,
      w,
      scales,
      dtype,
      bits,
      group_size,
      M,
      N,
      K,
      scale_factor,
      fp4_lut,
      fp8_lut);
}

} // namespace mlx::core
