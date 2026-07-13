#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

#define AT(TILE, IDX) TILE.thread_elements()[IDX]
#define SUB(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) - AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) - AT(TILE2, 1); \
  }
#define ADD(TILE0, TILE1, TILE2)                \
  {                                             \
    AT(TILE0, 0) = AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = AT(TILE1, 1) + AT(TILE2, 1); \
  }
#define FMA(TILE0, S, TILE1, TILE2)                 \
  {                                                 \
    AT(TILE0, 0) = S * AT(TILE1, 0) + AT(TILE2, 0); \
    AT(TILE0, 1) = S * AT(TILE1, 1) + AT(TILE2, 1); \
  }

#define SCALE(TILE0, S) \
  {                     \
    AT(TILE0, 0) *= S;  \
    AT(TILE0, 1) *= S;  \
  }
#define SCALE2(TILE0, S0, S1) \
  {                           \
    AT(TILE0, 0) *= S0;       \
    AT(TILE0, 1) *= S1;       \
  }
#define SCALE_TRI(TILE0, S0, S1)            \
  {                                         \
    AT(TILE0, 0) *= fn > fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 > fm ? 0.f : S1; \
  }
#define SCALE_TRIEQ(TILE0, S0, S1)           \
  {                                          \
    AT(TILE0, 0) *= fn >= fm ? 0.f : S0;     \
    AT(TILE0, 1) *= fn + 1 >= fm ? 0.f : S1; \
  }

// NAX MACROS I can probably do a nice template instead of doing this

// fm = base_fm + (idx >> 2) * 8;   // idx>>2 = idx/4  -> 0 for idx 0-3, 1 for
// idx 4-7 fn = base_fn + (idx % 4);        // 4 consecutive columns
#define AT_NAX(TILE, IDX) TILE.elems()[IDX]

#define SUB_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) - AT_NAX(TILE2, _i);    \
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

#define SCALE_ROW_NAX(TILE0, S)                                           \
  {                                                                       \
    STEEL_PRAGMA_UNROLL                                                   \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {       \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;       \
      AT_NAX(TILE0, _i) *= (S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]; \
    }                                                                     \
  }

#define SCALE_BETA_NAX(TILE0, BETA2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (BETA2)[_w >> 2];                        \
    }                                                               \
  }

#define SCALE2_NAX(TILE0, GAMMA)                                         \
  {                                                                      \
    STEEL_PRAGMA_UNROLL                                                  \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {      \
      const short _fm = mlx::steel::BaseNAXFrag::get_coord(              \
                            _i % mlx::steel::BaseNAXFrag::kElemsPerFrag) \
                            .y;                                          \
      AT_NAX(TILE0, _i) *= (GAMMA)[(C) - 1] / (GAMMA)[_fm];              \
    }                                                                    \
  }

#define SCALE_TRI_NAX(TILE0, GAMMA)                                            \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x > _c.y) ? 0.f : (1.0f / (GAMMA)[_c.x]);       \
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

#define SCALE_TRIEQ_NAX(TILE0, BETA, GAMMA)                                    \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      const short _fn = _c.x;                                                  \
      const short _fm = _c.y;                                                  \
      const float _s = (BETA)[_i >> 2] * ((GAMMA)[_fm] / (GAMMA)[_fn]);        \
      AT_NAX(TILE0, _i) *= (_fn >= _fm) ? 0.f : _s;                            \
    }                                                                          \
  }

namespace mlx {
namespace steel {
template <
    typename CType,
    typename AType,
    typename BType,
    bool transpose_a = false,
    bool transpose_b = false,
    mpp::tensor_ops::matmul2d_descriptor::mode Mode =
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate>
METAL_FUNC static constexpr void mmak(
    thread BaseNAXFrag::dtype_frag_t<CType>& C,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  // K = 32: two K-fragments per operand, single 16x16 output.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, 32, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A0[i];
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = A1[i];
    ct_b[i] = B0[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = B1[i];
    ct_c[i] = C[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

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
    thread BaseNAXFrag::dtype_frag_t<CType>& C0,
    thread BaseNAXFrag::dtype_frag_t<CType>& C1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B0,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B1,
    metal::bool_constant<transpose_b>) {
  // N = 32: single A fragment, two N-fragments for B and C output.
  // template parameters are M, N, K where
  // Tensor dimensions where M x K tensor A,K x N tensor B, and M x N tensor C.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A[i];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_b[i] = B0[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = B1[i];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_c[i] = C0[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = C1[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  // Copy out both N-fragments
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C0[i] = ct_c[i];
    C1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
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
METAL_FUNC static constexpr void mmam(
    thread BaseNAXFrag::dtype_frag_t<CType>& C0,
    thread BaseNAXFrag::dtype_frag_t<CType>& C1,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A0,
    const thread BaseNAXFrag::dtype_frag_t<AType>& A1,
    metal::bool_constant<transpose_a>,
    const thread BaseNAXFrag::dtype_frag_t<BType>& B,
    metal::bool_constant<transpose_b>) {
  // N = 32: single A fragment, two N-fragments for B and C output.
  // template parameters are M, N, K where
  // Tensor dimensions where M x K tensor A,K x N tensor B, and M x N tensor C.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      32, 16, 16, transpose_a, transpose_b, true, Mode);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  // Create matmul operands in registers
  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();

  // Create matmul output in register
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A0[i];
    ct_a[BaseNAXFrag::kElemsPerFrag + i] = A1[i];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_b[i] = B[i];
  }

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_c[i] = C0[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0;
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  // Copy out both N-fragments
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    C0[i] = ct_c[i];
    C1[i] = ct_c[BaseNAXFrag::kElemsPerFrag + i];
  }
}
} // namespace steel
} // namespace mlx

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_fused_nax(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device StT* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device StT* state_out [[buffer(7)]],
    constant int& T [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

#define OUTPUT_16(T) \
  T.store(y, 16);    \
  return;

#define OUTPUT_S(T, S) \
  T.store(y, S);       \
  return;

// pick the 3rd argument as the macro name
#define OUTPUT_GET(_1, _2, NAME, ...) NAME
#define OUTPUT(...) OUTPUT_GET(__VA_ARGS__, OUTPUT_S, OUTPUT_16)(__VA_ARGS__)

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;

  auto dv_idx = thread_position_in_grid.y * 16;

  // set up pointers
  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  threadgroup float gamma[C];
  float beta_fm[2];

  mlx::steel::NAXTile<StT, 1, Dk / 16> S_tile;
  S_tile.load(i_state, Dk);

  mlx::steel::NAXTile<InT, 1, 2> K_tile, Q_tile;
  mlx::steel::NAXTile<InT, 1, Dk / 16> KP_tile; // panel
  mlx::steel::NAXTile<InT, 1, Dk / 16> W_tile; // panel
  mlx::steel::NAXTile<InT, 1, Dk / 16> XP_tile; // panel

  mlx::steel::NAXTile<InT, 1, 1> X_tile;

  mlx::steel::NAXTile<InT, 1, 1> V_tile;
  mlx::steel::NAXTile<InT, 1, 1> U_tile;
  mlx::steel::NAXTile<InT, 1, 1> WS_tile;
  mlx::steel::NAXTile<InT, 1, 1> delta_tile;
  mlx::steel::NAXTile<InT, 1, 1> tmp_tile;
  mlx::steel::NAXTile<InT, 1, 1> QKt_tile;
  mlx::steel::NAXTile<InT, 1, 1> out_tile;
  mlx::steel::NAXTile<InT, 1, 1> K1_tile, Q1_tile;
  mlx::steel::NAXTile<InT, 1, 2> KD_tile;
  mlx::steel::NAXTile<InT, 1, 1> Tinv_tile;

  mlx::steel::NAXTile<InT, 1, 1> KKtK_tile, KKtV_tile, KKt_tile;

  mlx::steel::NAXTile<InT, 1, 1> P_tile;

  mlx::steel::NAXTile<InT, 1, 1> I_tile; // Declare and init eye
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    const short _fn = _c.x;
    const short _fm = _c.y;
    AT_NAX(I_tile, _i) = (_fn == _fm) ? 1.0f : 0.0f;
  }
  mlx::steel::NAXTile<InT, 1, 1> TMP_tile;

  mlx::steel::NAXTile<float, 1, 1> Z;

  Z.clear();

  for (int t = 0; t < T; t += C) {
    float g_val = (thread_index_in_simdgroup < C)
        ? g_[thread_index_in_simdgroup * Hv + hv_idx]
        : 1.0f;

    float gamma_val = simd_prefix_inclusive_product(g_val);

    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = gamma_val;
    }

    beta_fm[0] = beta_[fm * Hv + hv_idx];
    beta_fm[1] =
        beta_[(fm + mlx::steel::BaseNAXFrag::kElemRowsJump) * Hv + hv_idx];

    // I can only do matmuls with a dimension 32. This one is _easy_
    // i just do two tiles for the reduction
    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) { // two 16-tiles per iter
      K_tile.load(k_ + kk, Dk * Hk);
      mlx::steel::mmak<float, float, float, false, true>(
          KKt_tile.frag_at(0, 0),
          K_tile.frag_at(0, 0),
          K_tile.frag_at(0, 1),
          metal::bool_constant<false>{},
          K_tile.frag_at(0, 0),
          K_tile.frag_at(0, 1),
          metal::bool_constant<true>{});
    }

    // KKt_tile.store(y,16);
    // return;

    KKtK_tile = KKt_tile;
    KKtV_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm)
    SCALE_TRIEQ_NAX(KKtV_tile, beta_fm, gamma)

    // W = (I - diag(b)(KK.T))^-1 diag(b)K
    // diag(b)K
    KP_tile.load(k_, Dk * Hk);
    SCALE_BETA_NAX(KP_tile, beta_fm)

    // (I - diag(b)(KK.T))^-1 = sum T^k
    // Tinv = (I + L_W)^{-1} = sum -L_W_k
    // T0 - T + T2 - T3 + T4 - T5 + T6 - T7 + T8
    // I - T(I - T(I - T(I - T(I - T(I - T(I - T(I - T(I))))))))
    // Update 10.7: compute the above as (1 - T)(1 + T^2)(1 + T^4)(1 + T^8)
    //    => 6 mults instead of 14

    Z.clear();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < decltype(P_tile)::kElemsPerFrag; e++) {
      AT_NAX(P_tile, e) = AT_NAX(KKtK_tile, e);
    }

    // S = I + x = I - KKtV
    SUB_NAX(Tinv_tile, I_tile, KKtK_tile) // Tinv = I - T  (2 terms)
    // P^2
    mlx::steel::mmak<
        float,
        float,
        float,
        false,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply>(
        P_tile.frag_at(0, 0),
        P_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{},
        P_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{});

    STEEL_PRAGMA_UNROLL
    for (int step = 1; (1 << step) < C; step++) {
      // Tinv = S · P2
      mlx::steel::mmam<float, float, float, false, false>(
          Tinv_tile.frag_at(0, 0),
          P_tile.frag_at(0, 0),
          Tinv_tile.frag_at(0, 0),
          P_tile.frag_at(0, 0),
          metal::bool_constant<false>{},
          P_tile.frag_at(0, 0),
          metal::bool_constant<false>{});
    }
    // OUTPUT(Tinv_tile)

    // W = Tinv @ (diag(beta) K)
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Dk / 16; nn += 2) {
      mlx::steel::mman<
          float,
          float,
          float,
          false,
          false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply>(
          W_tile.frag_at(0, nn),
          W_tile.frag_at(0, nn + 1),
          Tinv_tile.frag_at(0, 0), // A = Tinv (reused across N)
          metal::bool_constant<false>{},
          KP_tile.frag_at(0, nn), // B = beta-scaled K
          KP_tile.frag_at(0, nn + 1),
          metal::bool_constant<false>{});
    }
    SCALE_ROW_NAX(W_tile, gamma)

    Z.clear();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < decltype(P_tile)::kElemsPerFrag; e++) {
      AT_NAX(P_tile, e) = AT_NAX(KKtV_tile, e);
    }

    // S = I + x = I - KKtV
    // Tinv = (1 - x)(1 + x^2)(1 + x^4)(1 + x^8)
    SUB_NAX(Tinv_tile, I_tile, KKtV_tile)

    // P^2
    mlx::steel::mmak<
        float,
        float,
        float,
        false,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply>(
        P_tile.frag_at(0, 0),
        P_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{},
        P_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{});

    STEEL_PRAGMA_UNROLL
    for (int step = 1; (1 << step) < C; step++) {
      // Tinv = S · P2
      mlx::steel::mmam<float, float, float, false, false>(
          Tinv_tile.frag_at(0, 0),
          P_tile.frag_at(0, 0),
          Tinv_tile.frag_at(0, 0),
          P_tile.frag_at(0, 0),
          metal::bool_constant<false>{},
          P_tile.frag_at(0, 0),
          metal::bool_constant<false>{});
    }
    V_tile.load(v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm)
    // U = Tinv @ diag(b)V
    mlx::steel::mmak<
        float,
        float,
        float,
        false,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply>(
        U_tile.frag_at(0, 0),
        Tinv_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{},
        V_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{});

    // WS = W @ S.T, easy reduction with stride 2 over K
    WS_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < Dk / 16; kk += 2) {
      mlx::steel::mmak<float, float, float, false, true>(
          WS_tile.frag_at(0, 0),
          W_tile.frag_at(0, kk),
          W_tile.frag_at(0, kk + 1),
          metal::bool_constant<false>{},
          S_tile.frag_at(0, kk),
          S_tile.frag_at(0, kk + 1),
          metal::bool_constant<true>{});
    }
    // OUTPUT(WS_tile)

    SUB_NAX(delta_tile, U_tile, WS_tile)

    tmp_tile.clear();
    QKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      Q_tile.load(q_ + kk, Hk * Dk);
      K_tile.load(k_ + kk, Hk * Dk); // this one is transposed

      SCALE_ROW_NAX(Q_tile, gamma)

      // Q_left @ S^T
      mlx::steel::mmak<float, InT, StT, false, true>(
          tmp_tile.frag_at(0, 0),
          Q_tile.frag_at(0, 0),
          Q_tile.frag_at(0, 1),
          metal::bool_constant<false>{},
          S_tile.frag_at(0, kk / 16),
          S_tile.frag_at(0, kk / 16 + 1),
          metal::bool_constant<true>{});

      // Q @ K^T
      mlx::steel::mmak<float, float, float, false, true>(
          QKt_tile.frag_at(0, 0),
          Q_tile.frag_at(0, 0),
          Q_tile.frag_at(0, 1),
          metal::bool_constant<false>{},
          K_tile.frag_at(0, 0),
          K_tile.frag_at(0, 1),
          metal::bool_constant<true>{});

      SCALE2_NAX(K_tile, gamma)

      // KD = delta.T @ K so that I can sum to S directly
      mlx::steel::mman<
          float,
          InT,
          float,
          true,
          false,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply>(
          KD_tile.frag_at(0, 0),
          KD_tile.frag_at(0, 1),
          delta_tile.frag_at(0, 0),
          metal::bool_constant<true>{},
          K_tile.frag_at(0, 0),
          K_tile.frag_at(0, 1),
          metal::bool_constant<false>{});

      FMA_NAX(
          S_tile.frag_at(0, kk / 16),
          gamma[C - 1],
          S_tile.frag_at(0, kk / 16),
          KD_tile.frag_at(0, 0))
      FMA_NAX(
          S_tile.frag_at(0, kk / 16 + 1),
          gamma[C - 1],
          S_tile.frag_at(0, kk / 16 + 1),
          KD_tile.frag_at(0, 1))
    }
    // OUTPUT(tmp_tile)

    SCALE_TRI_NAX(QKt_tile, gamma)
    // OUTPUT(QKt_tile)

    out_tile = tmp_tile;
    mlx::steel::mmak<float, float, float, false, false>(
        out_tile.frag_at(0, 0),
        QKt_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{},
        delta_tile.frag_at(0, 0),
        Z.frag_at(0, 0),
        metal::bool_constant<false>{});

    out_tile.store(y + dv_idx, Hv * Dv);
    // OUTPUT(out_tile);

    // advance pointers
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  S_tile.store(o_state, Dk);
}

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_fused_chunk(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device StT* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device StT* state_out [[buffer(7)]],
    constant int& T [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  const short qid = thread_index_in_simdgroup / 4;
  const short fm = (qid & 4) +
      ((thread_index_in_simdgroup / 2) % 4); // row coordinate of the held tile
  const short fn = (qid & 2) * 2 +
      (thread_index_in_simdgroup % 2) * 2; // column coordinate of the held tile

  auto dv_idx = thread_position_in_grid.y * 8;

  // set up pointers
  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  simdgroup_float8x8 S_tile[Dk / 8];

  // simdgroup matrices
  simdgroup_float8x8 V_tile, K_tile, KT_tile, Q_tile;
  simdgroup_float8x8 W_tile, U_tile;
  simdgroup_float8x8 WS_tile;
  simdgroup_float8x8 delta_tile;
  simdgroup_float8x8 tmp_tile;
  simdgroup_float8x8 QKt_tile;
  simdgroup_float8x8 out_tile;
  simdgroup_float8x8 KD_tile;

  // tiles for WY form computation
  simdgroup_float8x8 KKtK_tile, KKtV_tile, X_tile, KKt_tile;

  threadgroup float gamma[C];

  simdgroup_float8x8 I_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
  AT(I_tile, 0) = (fm == fn) ? 1.0f : 0.0f;
  AT(I_tile, 1) = (fm == fn + 1) ? 1.0f : 0.0f;

  // load initial state into threadgroup
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_load(S_tile[kk / 8], i_state + kk, Dk, ulong2(0, 0), true);
  }

  for (int t = 0; t < T; t += C) {
    float g_val = (thread_index_in_simdgroup < C)
        ? g_[thread_index_in_simdgroup * Hv + hv_idx]
        : 1.0f;

    float gamma_val = simd_prefix_inclusive_product(g_val);

    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = gamma_val;
    }

    float beta_fm = beta_[fm * Hv + hv_idx];

    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Dk * Hk);
      simdgroup_load(KT_tile, k_ + kk, Dk * Hk, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);
    }
#define OUTPUT(T)        \
  simdgroup_store(T, y); \
  return;

    KKtK_tile = KKt_tile;
    KKtV_tile = KKt_tile;

    // elementwise multiplication by Gamma and beta (for V) and beta (for K)
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)
    SCALE_TRIEQ(
        KKtV_tile,
        beta_fm * (gamma[fm] / gamma[fn]),
        beta_fm * (gamma[fm] / gamma[fn + 1]))

    // Tinv = (I + L_W)^{-1} = sum -L_W_k
    // T0 - T + T2 - T3 + T4 - T5 + T6 - T7 + T8
    // I - T(I - T(I - T(I - T(I - T(I - T(I - T(I - T(I))))))))
    simdgroup_float8x8 Tinv, P; //
    // S = I + x = I - KKtK
    AT(P, 0) = AT(KKtK_tile, 0);
    AT(P, 1) = AT(KKtK_tile, 1);
    SUB(Tinv, I_tile, KKtK_tile)

    for (int step = 1; (1 << step) < C; step++) {
      simdgroup_multiply(P, P, P);
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);
    }
    // OUTPUT(Tinv)

    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Dk * Hk);
      SCALE(K_tile, beta_fm)

      // W = Tinv @ (beta * K)
      simdgroup_multiply(W_tile, Tinv, K_tile);

      SCALE(W_tile, gamma[fm])
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile);
    }
    // OUTPUT(WS_tile)

    AT(P, 0) = AT(KKtV_tile, 0);
    AT(P, 1) = AT(KKtV_tile, 1);
    SUB(Tinv, I_tile, KKtV_tile)
    for (int step = 1; (1 << step) < C; step++) {
      simdgroup_multiply(P, P, P);
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);
    }

    // U = Tinv @ (beta * V)
    simdgroup_load(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE(V_tile, beta_fm)
    simdgroup_multiply(U_tile, Tinv, V_tile);

    // delta = U - WS
    SUB(delta_tile, U_tile, WS_tile)
    // OUTPUT(delta_tile)

    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(Q_tile, q_ + kk, Hk * Dk);
      simdgroup_load(K_tile, k_ + kk, Hk * Dk, ulong2(0, 0), true);

      SCALE(Q_tile, gamma[fm])

      // Q_left @ S^T
      simdgroup_multiply_accumulate(tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);
    }
    // OUTPUT(tmp_tile)

    // (Q @ K) * Gamma, in the paper they use M here but it's probably a typo.
    SCALE_TRI(QKt_tile, (1.0f / gamma[fn]), (1.0f / gamma[fn + 1]))
    // OUTPUT(QKt_tile)

    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);

    y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));
    y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));
    // OUTPUT(out_tile)

    for (int kk = 0; kk < Dk; kk += 8) {
      simdgroup_load(K_tile, k_ + kk, Hk * Dk, ulong2(0, 0), true);

      SCALE2(K_tile, (gamma[C - 1] / gamma[fn]), (gamma[C - 1] / gamma[fn + 1]))

      simdgroup_multiply(KD_tile, K_tile, delta_tile);

      FMA(S_tile[kk / 8], gamma[C - 1], S_tile[kk / 8], KD_tile)
    }

    // advance pointers
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }

  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_store(S_tile[kk / 8], o_state + kk, Dk, ulong2(0, 0), true);
  }
}

/*
        auto grid   = MTL::Size(32, Dv, B * Hv);
    auto threads = MTL::Size(32, 4, 1);
 */
template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_seq(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device InT* g [[buffer(3)]], // [B, T, Hv] or [B, T, Hv, Dk]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device StT* state_in [[buffer(5)]], // [B, Hv, Dv, Dk]
    constant int& T [[buffer(6)]],
    device InT* y [[buffer(7)]], // [B, T, Hv, Dv]
    device StT* state_out [[buffer(8)]], // [B, Hv, Dv, Dk]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  // kernel implementation
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);
  constexpr int n_per_t = Dk / 32;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  y += b_idx * T * Hv * Dv + hv_idx * Dv;

  auto dk_idx = thread_position_in_threadgroup.x;
  auto dv_idx = thread_position_in_grid.y;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(i_state[s_idx]);
  }

  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;
  auto beta_ = beta + b_idx * T * Hv;

  for (int t = 0; t < T; ++t) {
    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] * g_[hv_idx];
      kv_mem += state[i] * k_[s_idx];
    }
    kv_mem = simd_sum(kv_mem);

    auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = state[i] + k_[s_idx] * delta;
      out += state[i] * q_[s_idx];
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
    // Increment data pointers to next time step
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    o_state[s_idx] = static_cast<StT>(state[i]);
  }
}
