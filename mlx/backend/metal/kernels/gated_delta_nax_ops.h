// Copyright © 2026 Apple Inc.
#pragma once

#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

typedef mlx::steel::NAXTile<float, 1, 1> _M16x16;
typedef mlx::steel::NAXTile<float, 1, 2> _M16x32;

// NAX MACROS
#define AT_NAX(TILE, IDX) TILE.elems()[IDX]

// out = a - b
template <typename T>
METAL_FUNC T operator-(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) - b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator-(const thread T& a) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = -a.frag_at(0, f);
  }
  return out;
}

// out = a + b
template <typename T>
METAL_FUNC T operator+(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) + b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator+(const thread T& a, float addend) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) + addend;
  }
  return out;
}

#define TRIL_NAX(TILE0, TILE1)                                                 \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) = (_c.x >= _c.y) ? 0.f : AT_NAX(TILE1, _i);            \
    }                                                                          \
  }

// out = a * b, elementwise. Not a matmul.
template <typename T>
METAL_FUNC T operator*(const thread T& a, const thread T& b) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * b.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T operator*(const thread T& a, float s) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * s;
  }
  return out;
}

template <typename T>
METAL_FUNC T fmadd(const thread T& a, const thread T& b, const thread T& c) {
  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = a.frag_at(0, f) * b.frag_at(0, f) + c.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC _M16x16 reduce(const thread T& a) {
  _M16x16 out;
  out.frag_at(0, 0) = a.frag_at(0, 0);
  STEEL_PRAGMA_UNROLL
  for (short f = 1; f < T::kNumFrags; f++) {
    out.frag_at(0, 0) += a.frag_at(0, f);
  }
  return out;
}

template <typename T>
METAL_FUNC T scale_rows(const thread T& tile, const thread float* s) {
  typename T::frag_type sv;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < T::kElemsPerFrag; i++) {
    sv[i] = s[i >> 2];
  }

  T out;
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    out.frag_at(0, f) = tile.frag_at(0, f) * sv;
  }
  return out;
}

template <typename T>
METAL_FUNC void row_sum(thread float* dst, const thread T& tile) {
  STEEL_PRAGMA_UNROLL
  for (short f = 0; f < T::kNumFrags; f++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < T::kElemsPerFrag; i++) {
      dst[i >> 2] += tile.frag_at(0, f)[i];
    }
  }
}

// Scales row i by DEC2[group(i)] == exp(gamma_{C-1} - gamma_i).
#define SCALE2_P(TILE0, DEC2)                                       \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (DEC2)[_w >> 2];                         \
    }                                                               \
  }

#define SUB_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) - AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define ADD_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) + AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define FMA_NAX(TILE0, S, TILE1, TILE2)                                     \
  {                                                                         \
    STEEL_PRAGMA_UNROLL                                                     \
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) { \
      (TILE0)[_i] = (S) * (TILE1)[_i] + (TILE2)[_i];                        \
    }                                                                       \
  }

#define SCALE_NAX(TILE0, S)                                         \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) *= (S);                                     \
    }                                                               \
  }

#define SCALE_ROW_NAX(TILE0, S)                                            \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) *=                                                 \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
  }

#define TRIL_NAX(TILE0, TILE1)                                                 \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) = (_c.x >= _c.y) ? 0.f : AT_NAX(TILE1, _i);            \
    }                                                                          \
  }

#define SCALE_BETA_NAX(TILE0, BETA2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (BETA2)[_w >> 2];                        \
    }                                                               \
  }

#define SCALE2_NAX(TILE0, GAMMA)                                              \
  {                                                                           \
    STEEL_PRAGMA_UNROLL                                                       \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {           \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;           \
      const short _fm = mlx::steel::BaseNAXFrag::get_coord(_w).y;             \
      AT_NAX(TILE0, _i) *= metal::fast::exp((GAMMA)[(C) - 1] - (GAMMA)[_fm]); \
    }                                                                         \
  }

#define SCALE_TRI_NAX(TILE0, GAMMA)                                            \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x > _c.y)                                       \
          ? 0.f                                                                \
          : metal::fast::exp((GAMMA)[_c.y] - (GAMMA)[_c.x]);                   \
    }                                                                          \
  }

#define SCALE_TRIEQ_NAX1(TILE0, BETA)                                          \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x >= _c.y) ? 0.f : (BETA)[_i >> 2];             \
    }                                                                          \
  }

namespace mlx {
namespace steel {

// Split into an fp16 part and a remainder. The relaxed f32 matmul2d
// rounds its inputs to fp16, so multiply the parts separately.
template <typename T>
METAL_FUNC static void split_fp16(
    const thread BaseNAXFrag::dtype_frag_t<T>& x,
    thread BaseNAXFrag::dtype_frag_t<float>& hi,
    thread BaseNAXFrag::dtype_frag_t<float>& lo) {
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    const float h = float(half(x[i]));
    hi[i] = h;
    lo[i] = float(x[i]) - h;
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  // M=16, N=16, K=32: A and B each two K-fragments, single 16x16 C.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, 32, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  if constexpr (is_same_v<AType, float> && is_same_v<BType, float>) {
    using frag_t = BaseNAXFrag::dtype_frag_t<float>;
    frag_t a0_hi, a0_lo, a1_hi, a1_lo, b0_hi, b0_lo, b1_hi, b1_lo;
    split_fp16(A0, a0_hi, a0_lo);
    split_fp16(A1, a1_hi, a1_lo);
    split_fp16(B0, b0_hi, b0_lo);
    split_fp16(B1, b1_hi, b1_lo);

    frag_t acc;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = a0_hi[i];
      ct_a[BaseNAXFrag::kElemsPerFrag + i] = a1_hi[i];
      ct_b[i] = b0_hi[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = b1_hi[i];
      ct_c[i] = C[i];
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] = ct_c[i];
      ct_a[i] = a0_lo[i];
      ct_a[BaseNAXFrag::kElemsPerFrag + i] = a1_lo[i];
      ct_c[i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] += ct_c[i];
      ct_a[i] = a0_hi[i];
      ct_a[BaseNAXFrag::kElemsPerFrag + i] = a1_hi[i];
      ct_b[i] = b0_lo[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = b1_lo[i];
      ct_c[i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] += ct_c[i];
      ct_c[i] = acc[i];
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = A0[i];
      ct_a[BaseNAXFrag::kElemsPerFrag + i] = A1[i];
      ct_b[i] = B0[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = B1[i];
      ct_c[i] = C[i];
    }

    gemm_op.run(ct_a, ct_b, ct_c);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a,
    bool transpose_b,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mma(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B,
    metal::bool_constant<transpose_b>) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  if constexpr (is_same_v<AType, float> && is_same_v<BType, float>) {
    using frag_t = BaseNAXFrag::dtype_frag_t<float>;
    frag_t a_hi, a_lo, b_hi, b_lo;
    split_fp16(A, a_hi, a_lo);
    split_fp16(B, b_hi, b_lo);

    frag_t acc;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = a_hi[i];
      ct_b[i] = b_hi[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
      ct_c[i] = C[i];
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] = ct_c[i];
      ct_a[i] = a_lo[i];
      ct_c[i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] += ct_c[i];
      ct_a[i] = a_hi[i];
      ct_b[i] = b_lo[i];
      ct_c[i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc[i] += ct_c[i];
      ct_c[i] = acc[i];
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = A[i];
      ct_b[i] = B[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
      ct_c[i] = C[i];
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    }

    gemm_op.run(ct_a, ct_b, ct_c);
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C[i] = ct_c[i];
  }
}

template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mman(
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn0,
    thread BaseNAXFrag::dtype_frag_t<CType>& Cn1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& Bn1,
    metal::bool_constant<transpose_b>) {
  // M=16, N=32, K=16: single A (K=16), B and C two N-fragments each.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, transpose_a, transpose_b, true, Mode);

  // Create matmul op
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<
                          AType,
                          BType,
                          CType>());

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c =
      gemm_op
          .template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();

  if constexpr (is_same_v<AType, float> && is_same_v<BType, float>) {
    using frag_t = BaseNAXFrag::dtype_frag_t<float>;
    frag_t a_hi, a_lo, b0_hi, b0_lo, b1_hi, b1_lo;
    split_fp16(A, a_hi, a_lo);
    split_fp16(Bn0, b0_hi, b0_lo);
    split_fp16(Bn1, b1_hi, b1_lo);

    frag_t acc0, acc1;

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = a_hi[i];
      ct_b[i] = b0_hi[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = b1_hi[i];
      ct_c[i] = Cn0[i];
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = Cn1[i];
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc0[i] = ct_c[i];
      acc1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
      ct_a[i] = a_lo[i];
      ct_c[i] = 0.0;
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc0[i] += ct_c[i];
      acc1[i] += ct_c[BaseNAXFrag::kElemsPerFrag + i];
      ct_a[i] = a_hi[i];
      ct_b[i] = b0_lo[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = b1_lo[i];
      ct_c[i] = 0.0;
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      acc0[i] += ct_c[i];
      acc1[i] += ct_c[BaseNAXFrag::kElemsPerFrag + i];
      ct_c[i] = acc0[i];
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = acc1[i];
    }
  } else {
    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
      ct_a[i] = A[i];
      ct_b[i] = Bn0[i];
      ct_b[BaseNAXFrag::kElemsPerFrag + i] = Bn1[i];
      ct_c[i] = Cn0[i];
      ct_c[BaseNAXFrag::kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);
  }

  // Copy out results
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
  }
}

} // namespace steel
} // namespace mlx

#define MM16x16x16(C, CO, A, TA, AO, B, TB, BO)              \
  mlx::steel::mma<                                           \
      typename decltype(C)::elem_type,                       \
      typename decltype(A)::elem_type,                       \
      typename decltype(B)::elem_type,                       \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply>( \
      C.frag_at(0, (CO)),                                    \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      metal::bool_constant<TB>{});

#define MMA16x16x16(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mma<                                                      \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      metal::bool_constant<TB>{});

#define MMA16x16x32(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mma<                                                      \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      A.frag_at(0, (AO)),                                               \
      A.frag_at(0, (AO) + 1),                                           \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});

#define MM16x32x16(C, CO, A, TA, AO, B, TB, BO)              \
  mlx::steel::mman<                                          \
      typename decltype(C)::elem_type,                       \
      typename decltype(A)::elem_type,                       \
      typename decltype(B)::elem_type,                       \
      TA,                                                    \
      TB,                                                    \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply>( \
      C.frag_at(0, (CO)),                                    \
      C.frag_at(0, (CO) + 1),                                \
      A.frag_at(0, (AO)),                                    \
      metal::bool_constant<TA>{},                            \
      B.frag_at(0, (BO)),                                    \
      B.frag_at(0, (BO) + 1),                                \
      metal::bool_constant<TB>{});

#define MMA16x32x16(C, CO, A, TA, AO, B, TB, BO)                        \
  mlx::steel::mman<                                                     \
      typename decltype(C)::elem_type,                                  \
      typename decltype(A)::elem_type,                                  \
      typename decltype(B)::elem_type,                                  \
      TA,                                                               \
      TB,                                                               \
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>( \
      C.frag_at(0, (CO)),                                               \
      C.frag_at(0, (CO) + 1),                                           \
      A.frag_at(0, (AO)),                                               \
      metal::bool_constant<TA>{},                                       \
      B.frag_at(0, (BO)),                                               \
      B.frag_at(0, (BO) + 1),                                           \
      metal::bool_constant<TB>{});

// Row sum of an elementwise product, reduced over the fragment's column axis.
// DST[0] is the fm row group, DST[1] the fm + kElemRowsJump group.
#define ROWSUM_NAX(DST, TILE0, TILE1)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      (DST)[_w >> 2] += AT_NAX(TILE0, _i) * AT_NAX(TILE1, _i);      \
    }                                                               \
  }

// already generic
#define SCALE_BETA_NAX_O(DST, TILE, BETA2)                          \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE)::kElemsPerTile; _i++) {  \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(DST, _i) = AT_NAX(TILE, _i) * (BETA2)[_w >> 2];        \
    }                                                               \
  }

#define ROWSUM1_NAX(DST, TILE0)                                     \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      (DST)[_w >> 2] += AT_NAX(TILE0, _i);                          \
    }                                                               \
  }

#define MUL_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define MULA_NAX(TILE0, TILE1, TILE2)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) += AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);   \
    }                                                               \
  }

#define MULS_NAX(TILE0, TILE1, TILE2)                               \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) -= AT_NAX(TILE1, _i) * AT_NAX(TILE2, _i);   \
    }                                                               \
  }

#define NSCALE_ROW_NAX(TILE0, S)                                           \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) = -AT_NAX(TILE0, _i) *                             \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
  }

#define SCALE_ROW_P(TILE0, ROW2)                                    \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (ROW2)[_w >> 2];                         \
    }                                                               \
  }

// As SCALE_ROW_P but also negates, in one pass.
#define NSCALE_ROW_P(TILE0, ROW2)                                   \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) = -AT_NAX(TILE0, _i) * (ROW2)[_w >> 2];     \
    }                                                               \
  }

#define SCALE2_P(TILE0, DEC2)                                       \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (DEC2)[_w >> 2];                         \
    }                                                               \
  }