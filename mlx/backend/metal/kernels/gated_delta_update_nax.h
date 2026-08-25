#pragma once

#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

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

#define SCALE_ROW_NAX(TILE0, S)                                            \
  {                                                                        \
    STEEL_PRAGMA_UNROLL                                                    \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerTile; _i++) {        \
      const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;        \
      AT_NAX(TILE0, _i) *=                                                 \
          metal::fast::exp((S)[mlx::steel::BaseNAXFrag::get_coord(_w).y]); \
    }                                                                      \
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

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_fused_nax(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* state_in [[buffer(3)]],
    const device InT* g [[buffer(4)]],
    const device InT* beta [[buffer(5)]],
    device InT* y [[buffer(6)]],
    device float* state_out [[buffer(7)]],
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

  mlx::steel::NAXTile<float, 1, 1> KKtK_tile, KKt_tile;

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

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(
                  static_cast<float>(
                      g_[thread_index_in_simdgroup * Hv + hv_idx]),
                  1e-6f))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      MMA16x16x32(KKt_tile, 0, K_tile, false, 0, K_tile, true, 0);
    }

    KKtK_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);
    SUB_NAX(Tinv_tile, I_tile, KKtK_tile);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, Tinv_tile, false, 0);
      SUB_NAX(Tinv_tile, I_tile, TMP_tile);
    }

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

    SCALE_NAX(S_tile, metal::fast::exp(gamma[C - 1]));

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