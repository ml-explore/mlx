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
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) { \
      AT_NAX(TILE0, _i) = AT_NAX(TILE1, _i) - AT_NAX(TILE2, _i);    \
    }                                                               \
  }

#define ADD_NAX(TILE0, TILE1, TILE2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) { \
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

#define SCALE_ROW_NAX(TILE0, S)                                      \
  {                                                                  \
    STEEL_PRAGMA_UNROLL                                              \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {  \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;  \
      AT_NAX(TILE0, _i) *=                                           \
          metal::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                \
  }

#define SCALE_BETA_NAX(TILE0, BETA2)                                \
  {                                                                 \
    STEEL_PRAGMA_UNROLL                                             \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) { \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag; \
      AT_NAX(TILE0, _i) *= (BETA2)[_w >> 2];                        \
    }                                                               \
  }

#define SCALE2_NAX(TILE0, GAMMA)                                        \
  {                                                                     \
    STEEL_PRAGMA_UNROLL                                                 \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {     \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;     \
      const short _fm = mlx::steel::BaseNAXFrag::get_coord(_w).y;       \
      AT_NAX(TILE0, _i) *= metal::exp((GAMMA)[(C) - 1] - (GAMMA)[_fm]); \
    }                                                                   \
  }

#define SCALE_TRI_NAX(TILE0, GAMMA)                                            \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *=                                                     \
          (_c.x > _c.y) ? 0.f : metal::exp((GAMMA)[_c.y] - (GAMMA)[_c.x]);     \
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
      const float _s =                                                         \
          (BETA)[_i >> 2] * metal::exp((GAMMA)[_fm] - (GAMMA)[_fn]);           \
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

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
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

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
  auto ct_b =
      gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      CType>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemsPerFrag; i++) {
    ct_a[i] = A[i];
    ct_b[i] = B[i];
    ct_b[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
    ct_c[i] = C[i];
    ct_c[BaseNAXFrag::kElemsPerFrag + i] = 0.0;
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
      float,                                                 \
      float,                                                 \
      float,                                                 \
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
      float,                                                            \
      float,                                                            \
      float,                                                            \
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
      float,                                                            \
      float,                                                            \
      float,                                                            \
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
      float,                                                 \
      float,                                                 \
      float,                                                 \
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
      float,                                                            \
      float,                                                            \
      float,                                                            \
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
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

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

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  float beta_fm[2];

  mlx::steel::NAXTile<float, 1, Dk / 16> S_tile;
  S_tile.load(i_state, Dk);

  mlx::steel::NAXTile<float, 1, 2> K_tile, Q_tile;
  mlx::steel::NAXTile<float, 1, Dk / 16> W_tile; // panel

  mlx::steel::NAXTile<float, 1, 1> V_tile;
  mlx::steel::NAXTile<float, 1, 1> U_tile;
  mlx::steel::NAXTile<float, 1, 1> WS_tile;
  mlx::steel::NAXTile<float, 1, 1> delta_tile;
  mlx::steel::NAXTile<float, 1, 1> tmp_tile;
  mlx::steel::NAXTile<float, 1, 1> QKt_tile;
  mlx::steel::NAXTile<float, 1, 1> out_tile;
  mlx::steel::NAXTile<float, 1, 1> Tinv_tile, P;

  mlx::steel::NAXTile<float, 1, 1> KKtK_tile, KKtV_tile, KKt_tile;

  mlx::steel::NAXTile<float, 1, 1> I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    const short _fn = _c.x;
    const short _fm = _c.y;
    AT_NAX(I_tile, _i) = (_fn == _fm) ? 1.0f : 0.0f;
  }
  mlx::steel::NAXTile<float, 1, 1> TMP_tile;

  auto process_chunk = [&](const short valid_rows,
                           auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;

    auto load_seq = [&](thread auto& tile, auto src, int ld) {
      if constexpr (B) {
        tile.load_rows(src, ld, valid_rows);
      } else {
        tile.load(src, ld);
      }
    };

    auto g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::log(
              metal::clamp(
                  g_[thread_index_in_simdgroup * Hv + hv_idx], 1e-6f, 1.0f))
        : 0.0f;
    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      MMA16x16x32(KKt_tile, 0, K_tile, false, 0, K_tile, true, 0);
    }

    KKtK_tile = KKt_tile;
    KKtV_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);
    MM16x16x16(P, 0, KKtK_tile, false, 0, KKtK_tile, false, 0);

    ADD_NAX(Tinv_tile, I_tile, P);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 6; step++) {
      MM16x16x16(TMP_tile, 0, P, false, 0, Tinv_tile, false, 0);
      ADD_NAX(Tinv_tile, I_tile, TMP_tile);
    }

    MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, Tinv_tile, false, 0);
    SUB_NAX(Tinv_tile, Tinv_tile, TMP_tile);

    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Dk / 16; nn += 2) {
      load_seq(K_tile, k_ + nn * 16, Dk * Hk);
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_tile, nn, Tinv_tile, false, 0, K_tile, false, 0);
    }
    SCALE_ROW_NAX(W_tile, gamma)

    SCALE_TRI_NAX(Tinv_tile, gamma)
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(U_tile, 0, Tinv_tile, false, 0, V_tile, false, 0)

        WS_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < Dk / 16; kk += 2) {
      MMA16x16x32(WS_tile, 0, W_tile, false, kk, S_tile, true, kk)
    }

    SUB_NAX(delta_tile, U_tile, WS_tile)

    tmp_tile.clear();
    QKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Hk * Dk);

      MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, K_tile, true, 0);

      SCALE_ROW_NAX(Q_tile, gamma);
      MMA16x16x32(tmp_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16);
    }

    SCALE_TRI_NAX(QKt_tile, gamma)

    out_tile = tmp_tile;
    MMA16x16x16(out_tile, 0, QKt_tile, false, 0, delta_tile, false, 0);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(out_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      const short _fn = _c.x;
      const short _fm = _c.y;
      if (_fm < valid_rows) {
        y[_fm * Hv * Dv + dv_idx + _fn] =
            static_cast<InT>(AT_NAX(out_tile, _i));
      }
    }

    SCALE_NAX(S_tile, metal::exp(gamma[C - 1]));

    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Hk * Dk);
      SCALE2_NAX(K_tile, gamma);
      MMA16x32x16(S_tile, kk / 16, delta_tile, true, 0, K_tile, false, 0);
    }
  };

  int t = 0;
  for (; t + C <= T; t += C) {
    process_chunk(C, metal::false_type{});
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    process_chunk(short(T - t), metal::true_type{});
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
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
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
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

#define OUTPUT(T)          \
  if (true) {              \
    simdgroup_store(T, y); \
    return;                \
  }
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
  simdgroup_float8x8 KKtK_tile, KKtV_tile, KKt_tile;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  simdgroup_float8x8 I_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
  AT(I_tile, 0) = (fm == fn) ? 1.0f : 0.0f;
  AT(I_tile, 1) = (fm == fn + 1) ? 1.0f : 0.0f;

  // load initial state into registers
  for (int kk = 0; kk < Dk; kk += 8) {
    simdgroup_load(S_tile[kk / 8], i_state + kk, Dk, ulong2(0, 0), true);
  }

  auto process_chunk = [&](thread simdgroup_float8x8* S_tile,
                           const short valid_rows,
                           auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;

    // non-transposed load
    auto load_M = [&](thread simdgroup_float8x8& M, auto src, int ld) {
      if constexpr (B) {
        AT(M, 0) = (fm < valid_rows) ? float(src[fm * ld + fn]) : 0.f;
        AT(M, 1) = (fm < valid_rows) ? float(src[fm * ld + fn + 1]) : 0.f;
      } else {
        simdgroup_load(M, src, ld);
      }
    };

    // transposed load
    auto load_MT = [&](thread simdgroup_float8x8& M, auto src, int ld) {
      if constexpr (B) {
        AT(M, 0) = (fn < valid_rows) ? float(src[fn * ld + fm]) : 0.f;
        AT(M, 1) = (fn + 1 < valid_rows) ? float(src[(fn + 1) * ld + fm]) : 0.f;
      } else {
        simdgroup_load(M, src, ld, ulong2(0, 0), true);
      }
    };

    float g_val = (thread_index_in_simdgroup < C)
        ? g_[thread_index_in_simdgroup * Hv + hv_idx]
        : 1.0f;

    float gamma_val = simd_prefix_inclusive_product(g_val);

    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = gamma_val;
    }

    gamma[C - 1] = metal::max(gamma[C - 1], 1e-9f);

    float beta_fm = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;

    KKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    STEEL_PRAGMA_UNROLL
    for (int kk = 0; kk < Dk; kk += 8) {
      load_M(K_tile, k_ + kk, Dk * Hk);
      load_MT(KT_tile, k_ + kk, Dk * Hk);
      simdgroup_multiply_accumulate(KKt_tile, K_tile, KT_tile, KKt_tile);
    }

    KKtK_tile = KKt_tile;
    SCALE_TRIEQ(KKtK_tile, beta_fm, beta_fm)

    simdgroup_float8x8 Tinv, P;
    AT(P, 0) = AT(KKtK_tile, 0);
    AT(P, 1) = AT(KKtK_tile, 1);
    SUB(Tinv, I_tile, KKtK_tile)

    STEEL_PRAGMA_UNROLL
    for (int step = 1; (1 << step) < C; step++) {
      simdgroup_multiply(P, P, P);
      simdgroup_multiply_accumulate(Tinv, Tinv, P, Tinv);
    }

    WS_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    STEEL_PRAGMA_UNROLL
    for (int kk = 0; kk < Dk; kk += 8) {
      load_M(K_tile, k_ + kk, Dk * Hk);
      SCALE(K_tile, beta_fm)
      simdgroup_multiply(W_tile, Tinv, K_tile);
      SCALE(W_tile, gamma[fm])
      simdgroup_multiply_accumulate(WS_tile, W_tile, S_tile[kk / 8], WS_tile);
    }

    SCALE_TRI(Tinv, (gamma[fm] / gamma[fn]), (gamma[fm] / gamma[fn + 1]))

    load_M(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE(V_tile, beta_fm)
    simdgroup_multiply(U_tile, Tinv, V_tile);
    SUB(delta_tile, U_tile, WS_tile)

    tmp_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    QKt_tile = make_filled_simdgroup_matrix<float, 8>(0.f);
    STEEL_PRAGMA_UNROLL
    for (int kk = 0; kk < Dk; kk += 8) {
      load_M(Q_tile, q_ + kk, Hk * Dk);
      load_MT(K_tile, k_ + kk, Hk * Dk);
      SCALE(Q_tile, gamma[fm])
      simdgroup_multiply_accumulate(QKt_tile, Q_tile, K_tile, QKt_tile);
      simdgroup_multiply_accumulate(tmp_tile, Q_tile, S_tile[kk / 8], tmp_tile);
    }

    SCALE_TRI(QKt_tile, (1.0f / gamma[fn]), (1.0f / gamma[fn + 1]))

    simdgroup_multiply_accumulate(out_tile, QKt_tile, delta_tile, tmp_tile);

    if (fm < valid_rows) {
      y[fm * Hv * Dv + dv_idx + fn] = static_cast<InT>(AT(out_tile, 0));
      y[fm * Hv * Dv + dv_idx + fn + 1] = static_cast<InT>(AT(out_tile, 1));
    }

    STEEL_PRAGMA_UNROLL
    for (int kk = 0; kk < Dk; kk += 8) {
      load_MT(K_tile, k_ + kk, Hk * Dk);
      SCALE2(K_tile, (gamma[C - 1] / gamma[fn]), (gamma[C - 1] / gamma[fn + 1]))

      simdgroup_multiply(KD_tile, K_tile, delta_tile);
      FMA(S_tile[kk / 8], gamma[C - 1], S_tile[kk / 8], KD_tile)
    }
  };

  int t = 0;
  STEEL_PRAGMA_UNROLL
  for (; t + C <= T; t += C) {
    process_chunk(S_tile, C, metal::false_type{});
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    process_chunk(S_tile, short(T - t), metal::true_type{});
  }

  STEEL_PRAGMA_UNROLL
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