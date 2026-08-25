#pragma once

#include <metal_stdlib>

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>

#include <metal_atomic>
#include "mlx/backend/metal/kernels/atomic.h"

#include "mlx/backend/metal/kernels/gated_delta_update_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

typedef mlx::steel::NAXTile<float, 1, 1> _M16x16;
typedef mlx::steel::NAXTile<float, 1, 2> _M16x32;

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

#define SCALE_TRI_T_NAX(TILE0, GAMMA)                                          \
  {                                                                            \
    STEEL_PRAGMA_UNROLL                                                        \
    for (short _i = 0; _i < decltype(TILE0)::kElemsPerFrag; _i++) {            \
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */ \
      AT_NAX(TILE0, _i) *= (_c.x < _c.y)                                       \
          ? 0.f                                                                \
          : metal::fast::exp((GAMMA)[_c.x] - (GAMMA)[_c.y]);                   \
    }                                                                          \
  }

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_vjp_fused_nax(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_chunks, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<InT>* dq [[buffer(9)]], // [B, T, Hk, Dk]
    device mlx_atomic<InT>* dk [[buffer(10)]], // [B, T, Hk, Dk]
    device InT* dv [[buffer(11)]], // [B, T, Hv, Dv]
    device mlx_atomic<InT>* dg [[buffer(12)]], // [B, T, Hv] holds dL/dgamma
    device mlx_atomic<InT>* db [[buffer(13)]], // [B, T, Hv]
    device float* dh [[buffer(14)]], // [B, Hv, Dv, Dk]
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  using _M16xDk = mlx::steel::NAXTile<float, 1, Dk / 16>;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  const int n_chunks = (T + C - 1) / C;
  const int t_last = (n_chunks - 1) * C;

  // Pointers positioned at the final chunk
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dq_ = dq + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;
  auto dk_ = dk + b_idx * T * Hk * Dk + hk_idx * Dk + t_last * Hk * Dk;

  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto dv_ = dv + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;
  auto co_ = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv + t_last * Hv * Dv;

  auto g_ = g + b_idx * T * Hv + t_last * Hv;
  auto beta_ = beta + b_idx * T * Hv + t_last * Hv;
  auto dg_ = dg + b_idx * T * Hv + t_last * Hv;
  auto db_ = db + b_idx * T * Hv + t_last * Hv;

  auto c_state = state_cache + (n * n_chunks * Dv + dv_idx) * Dk +
      (n_chunks - 1) * Dv * Dk;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  float beta_fm[2];

  // Carried state gradient, dL/dS. Same [dv, dk] orientation as S_tile.
  _M16xDk dS_tile;
  dS_tile.load(i_cot_h, Dk);

  // Forward state at chunk entry, reloaded from the checkpoint each chunk
  _M16xDk S_tile;

  // Recomputed forward tiles
  _M16x32 K_tile, Q_tile;
  _M16xDk W_tile;
  _M16x16 K16_tile, Q16_tile;
  _M16x16 V_tile;
  _M16x16 U_tile;
  _M16x16 WS_tile;
  _M16x16 delta_tile;
  _M16x16 QKt_tile, QKt_raw;
  _M16x16 KKtK_tile, KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  // Backward tiles
  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 NI_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(NI_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(NI_tile, _i) = (_c.x == _c.y) ? -1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

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

    // reload the checkpoint
    S_tile.load(c_state, Dk);

    // recompute the forward tiles for this chunk. Same code as forward kernel.
    // Is there a better way of doing this?
    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(g_[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const float gamma_last = gamma[C - 1];

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // KKt = K K^T
    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      MMA16x16x32(KKt_tile, 0, K_tile, false, 0, K_tile, true, 0);
    }

    // A = tril_(diag(beta) K K^T)
    KKtK_tile = KKt_tile;
    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);

    SUB_NAX(TWinv_tile, I_tile, KKtK_tile);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, TWinv_tile, false, 0);
      SUB_NAX(TWinv_tile, I_tile, TMP_tile);
    }

    // W = diag(gamma) (TWinv diag(beta) K), kept per Dk block since the
    // backward needs W^T d_delta for every block.
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Dk / 16; nn += 2) {
      load_seq(K_tile, k_ + nn * 16, Dk * Hk);
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_tile, nn, TWinv_tile, false, 0, K_tile, false, 0);
    }
    SCALE_ROW_NAX(W_tile, gamma)

    TUinv_tile = TWinv_tile;
    SCALE_TRI_NAX(TUinv_tile, gamma)

    // U = T_U diag(beta) V
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(U_tile, 0, TUinv_tile, false, 0, V_tile, false, 0)

        WS_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < Dk / 16; kk += 2) {
      MMA16x16x32(WS_tile, 0, W_tile, false, kk, S_tile, true, kk)
    }
    SUB_NAX(delta_tile, U_tile, WS_tile)

    QKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Hk * Dk);
      MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, K_tile, true, 0);
    }
    QKt_raw = QKt_tile;
    SCALE_TRI_NAX(QKt_tile, gamma)

    // From here do the backward through the chunk
    // dgamma accumulators:
    //   dgam_row  : per-row
    //   dgam_pair : row-minus-col
    //   dgam_last : the gamma_{C-1}
    _M16x16 dgam_row, dgam_pair;
    dgam_row.clear();
    dgam_pair.clear();
    float dgam_last = 0.0f;
    float dbeta_acc[2] = {0.0f, 0.0f};

    load_seq(dout_tile, co_ + dv_idx, Hv * Dv);

    // dODT = (dO @ delta.T * D), i.e. dM already carrying the decay + mask
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0)
        SCALE_TRI_NAX(dODT, gamma)

        // cannot fuse this because ddelta is used by the next loop
        MM16x16x16(
            ddelta_tile,
            0,
            QKt_tile,
            true,
            0,
            dout_tile,
            false,
            0) for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      SCALE2_NAX(K_tile, gamma);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16)
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      _M16x32 Kb_tile;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16)
          SCALE_ROW_NAX(dW_raw, gamma) SCALE_NAX(dW_raw, -1.0f)

              SCALE_BETA_NAX_O(Kb_tile, K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0)

          MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16)
              SCALE_ROW_NAX(dq_acc, gamma)
                  MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0)

          // dq reduces over Dv
          STEEL_PRAGMA_UNROLL for (short _i = 0;
                                   _i < decltype(dq_acc)::kElemsPerTile;
                                   _i++) {
        const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
        const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
        const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
        const short _fn = _c.x + _f * 16;
        const short _fm = _c.y;
        if (_fm < valid_rows) {
          mlx_atomic_fetch_add_explicit(
              dq_,
              static_cast<InT>(AT_NAX(dq_acc, _i)),
              _fm * Hk * Dk + kk + _fn);
        }
      }
    }

    // dv = B(Tu.T @ ddelta)
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0)
        _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    SCALE_BETA_NAX(dVb_tile, beta_fm);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(dVb_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      if (_c.y < valid_rows) {
        dv_[_c.y * Hv * Dv + dv_idx + _c.x] =
            static_cast<InT>(AT_NAX(dVb_tile, _i));
      }
    }

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0)
        SCALE_TRI_NAX(dTU_tile, gamma) // dTinv = dT_W + (dT_U * D).

        ADD_NAX(dTinv_tile, dTW_tile, dTU_tile)

        // dA = -T.T @ dTinv @ T.T
        MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0)
            MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0)
                SCALE_NAX(dA_tile, -1.0f)

        // G = tril_(dA) * beta, and GGt = G + G.T
        G_tile = dA_tile;
    SCALE_TRIEQ_NAX1(G_tile, beta_fm);
    _M16x16 GGt_tile;
    MM16x16x16(GGt_tile, 0, G_tile, false, 0, I_tile, false, 0)
        MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0)

        // dgamma
        _M16x16 P_tile;
    _M16x16 R_tile;

    MUL_NAX(P_tile, QKt_raw, dODT)
    MUL_NAX(R_tile, TWinv_tile, dTU_tile)
    ADD_NAX(dgam_pair, P_tile, R_tile)

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, dKb_tile, t1, KdKb_tmp;
      _M16x32 QgdOS, TbK, gdDS, dKC, KdKC_blk;

      load_seq(Q_tile, q_ + kk, Hk * Dk); // [16 x 32]
      load_seq(K_tile, k_ + kk, Dk * Hk); // [16 x 32]

      // gamma: + rowsum(Q * gamma*(dO @ S))
      MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16)
          SCALE_ROW_NAX(QgdOS, gamma) MULA_NAX(dgam_row32, QgdOS, Q_tile)

          // dODT.T @ Q
          // (G + G.T) @ K
          MM16x32x16(
              dk_acc, 0, dODT, true, 0, Q_tile, false, 0) // [16 x 32] = [16 x
                                                          // 16] @ [16 x 32]
          MMA16x32x16(
              dk_acc,
              0,
              GGt_tile,
              false,
              0,
              K_tile,
              false,
              0) // [16 x 32] = [16 x 16] @ [16 x 32]

          // B (T.T @ dW_raw),  dW_raw = -gamma * (ddelta @ S)
          MM16x32x16(
              dW_raw,
              0,
              ddelta_tile,
              false,
              0,
              S_tile,
              false,
              kk / 16) // [16 x 32] = [16 x 16] @ [16 x 32]
          SCALE_ROW_NAX(dW_raw, gamma) SCALE_NAX(dW_raw, -1.0f) MM16x32x16(
              dKb_tile,
              0,
              TWinv_tile,
              true,
              0,
              dW_raw,
              false,
              0) // [16 x 32] = [16 x 16] @ [16 x 32]

          // dbeta term uses dK_b before the beta scaling, with raw K
          MUL_NAX(KdKb_tmp, dKb_tile, K_tile) ADD_NAX(KdKb, KdKb, KdKb_tmp)

              SCALE_BETA_NAX(dKb_tile, beta_fm);
      ADD_NAX(dk_acc, dk_acc, dKb_tile)

      // D_C * (delta @ dS)
      MM16x32x16(
          t1,
          0,
          delta_tile,
          false,
          0,
          dS_tile,
          false,
          kk / 16) // [16 x 32] = [16 x 16] @ [16 x 32]
          SCALE2_NAX(t1, gamma) ADD_NAX(dk_acc, dk_acc, t1)

          // gamma stuff
          MUL_NAX(KdKC_blk, K_tile, t1)
              SUB_NAX(dgam_row32, dgam_row32, KdKC_blk)
                  STEEL_PRAGMA_UNROLL for (short _i = 0; _i <
                                           decltype(KdKC_blk)::kElemsPerTile;
                                           _i++) {
        dgam_last += AT_NAX(KdKC_blk, _i);
      }

      // gamma stuff: gamma_C * sum(S * dS)
      STEEL_PRAGMA_UNROLL
      for (short _f = 0; _f < 2; _f++) {
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
          dgam_last += metal::fast::exp(gamma_last) *
              S_tile.frag_at(0, kk / 16 + _f)[_i] *
              dS_tile.frag_at(0, kk / 16 + _f)[_i];
        }
      }

      // gamma stuff
      SCALE_BETA_NAX(K_tile, beta_fm)
      SCALE_NAX(dW_raw, -1.0f)
      MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0)
          MULS_NAX(dgam_row32, TbK, dW_raw)

          // dK reduces over Dv (and over hv under GQA) -> atomic
          STEEL_PRAGMA_UNROLL for (short _i = 0;
                                   _i < decltype(dk_acc)::kElemsPerTile;
                                   _i++) {
        const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
        const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
        const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
        if (_c.y < valid_rows) {
          mlx_atomic_fetch_add_explicit(
              dk_,
              static_cast<InT>(AT_NAX(dk_acc, _i)),
              _c.y * Hk * Dk + kk + _c.x + _f * 16);
        }
      }
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)
    _M16x16 VdVb;
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    MUL_NAX(VdVb, V_tile, TUddelta)

    _M16x16 AKKt;
    TRIL_NAX(AKKt, dA_tile)
    MUL_NAX(AKKt, AKKt, KKt_tile);

    // ADD_NAX(VdVb, VdVb, KdKb)
    ADD_NAX(VdVb, VdVb, AKKt)
    ROWSUM1_NAX(dbeta_acc, VdVb);
    ROWSUM1_NAX(dbeta_acc, KdKb);

    STEEL_PRAGMA_UNROLL
    for (short _r = 0; _r < 2; _r++) {
      float _b = dbeta_acc[_r];
      _b += simd_shuffle_xor(_b, ushort(1));
      _b += simd_shuffle_xor(_b, ushort(8));

      const short _row = fm + _r * mlx::steel::BaseNAXFrag::kElemRowsJump;
      if ((simd_lane_id & 9) == 0 && _row < valid_rows) {
        mlx_atomic_fetch_add_explicit(
            db_, static_cast<InT>(_b), _row * Hv + hv_idx);
      }
    }

    // Fold the two fragments: only the row sum of dgam_row is used downstream,
    // and row sums are additive across fragments.
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
      AT_NAX(dgam_row, _i) = AT_NAX(dgam_row32, _i) +
          AT_NAX(dgam_row32, mlx::steel::BaseNAXFrag::kElemsPerFrag + _i);
    }

    // reduce and store dgamma
    _M16x16 dgam_tile, rs_tile, cs_tile;
    MM16x16x16(dgam_tile, 0, dgam_row, false, 0, Ones_tile, false, 0)
        MM16x16x16(rs_tile, 0, dgam_pair, false, 0, Ones_tile, false, 0)
            MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0)
                ADD_NAX(dgam_tile, dgam_tile, rs_tile) SUB_NAX(
                    dgam_tile, dgam_tile, cs_tile)

                    STEEL_PRAGMA_UNROLL for (short _i = 0; _i <
                                             decltype(dgam_tile)::kElemsPerFrag;
                                             _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      if (_c.x == 0 && _c.y < valid_rows) {
        mlx_atomic_fetch_add_explicit(
            dg_, static_cast<InT>(AT_NAX(dgam_tile, _i)), _c.y * Hv + hv_idx);
      }
    }

    const float dgam_last_red = simd_sum(dgam_last);
    if (simd_lane_id == 0 && valid_rows > 0) {
      mlx_atomic_fetch_add_explicit(
          dg_, static_cast<InT>(dgam_last_red), (valid_rows - 1) * Hv + hv_idx);
    }

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, metal::fast::exp(gamma_last));
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      SCALE_ROW_NAX(Q_tile, gamma)
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0)

          // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
          SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0)
          SCALE_ROW_NAX(W_raw, gamma) SCALE_NAX(W_raw, -1.0f) MMA16x32x16(
              term, 0, ddelta_tile, true, 0, W_raw, false, 0)

              STEEL_PRAGMA_UNROLL for (short _i = 0; _i <
                                       mlx::steel::BaseNAXFrag::kElemsPerFrag;
                                       _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, mlx::steel::BaseNAXFrag::kElemsPerFrag + _i);
      }
    }
  };

  // Walk chunks in reverse. The tail chunk comes first.
  int c = n_chunks - 1;
  const short tail = short(T - c * C);
  if (tail != C) {
    process_chunk(tail, metal::true_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
    --c;
  }
  for (; c >= 0; --c) {
    process_chunk(C, metal::false_type{});
    q_ -= C * Hk * Dk;
    k_ -= C * Hk * Dk;
    v_ -= C * Hv * Dv;
    co_ -= C * Hv * Dv;
    g_ -= C * Hv;
    beta_ -= C * Hv;
    dq_ -= C * Hk * Dk;
    dk_ -= C * Hk * Dk;
    dv_ -= C * Hv * Dv;
    dg_ -= C * Hv;
    db_ -= C * Hv;
    c_state -= Dv * Dk;
  }

  dS_tile.store(o_dh, Dk);
}

// Postprocessing dg. Similar idea to what is done in the triton kernel.
// The kernel computes the gradient with respect to the log.
template <typename InT, int C>
[[kernel]] void gated_delta_dgamma_to_dg(
    const device InT* g [[buffer(0)]], // [B, T, Hv]
    device InT* dg [[buffer(1)]], // [B, T, Hv], in: dL/dgamma, out: dL/dg
    constant int& T [[buffer(2)]],
    constant int& Hv [[buffer(3)]],
    constant int& n_total [[buffer(4)]], // B * Hv
    uint2 pos [[thread_position_in_grid]]) {
  const int n = int(pos.x); // b * Hv + hv
  const int c = int(pos.y); // chunk index

  if (n >= n_total) {
    return;
  }

  const int t0 = c * C;
  if (t0 >= T) {
    return;
  }
  const int len = min(C, T - t0);

  const int b_idx = n / Hv;
  const int hv_idx = n % Hv;
  const int base = b_idx * T * Hv + hv_idx;

  float acc = 0.0f;
  for (int j = len - 1; j >= 0; --j) {
    const int idx = base + (t0 + j) * Hv;
    acc += static_cast<float>(dg[idx]);
    float gv = metal::max(static_cast<float>(g[idx]), 1e-6f);
    dg[idx] = static_cast<InT>(acc / gv);
  }
}
