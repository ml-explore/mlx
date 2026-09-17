#pragma once

#include "mlx/backend/metal/kernels/gated_delta_nax_ops.h"

#include <metal_atomic>
#include "mlx/backend/metal/kernels/atomic.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

template <int kNSG, bool kGQA, typename TileT>
METAL_FUNC void reduce_tile_tg(
    thread TileT& acc,
    threadgroup float* scratch,
    device mlx_atomic<float>* out,
    int row_stride,
    int col_off,
    short valid_rows,
    const short sg_id,
    const ushort simd_lane_id) {
  constexpr short kE = TileT::kElemsPerTile;

  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < kE; _i++) {
    scratch[(sg_id * 32 + simd_lane_id) * kE + _i] = AT_NAX(acc, _i);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg_id == 0) {
    STEEL_PRAGMA_UNROLL
    for (short _s = 1; _s < kNSG; _s++) {
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        AT_NAX(acc, _i) += scratch[(_s * 32 + simd_lane_id) * kE + _i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Simdgroup 0 alone holds the total.
  if (sg_id != 0) {
    return;
  }

  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < kE; _i++) {
    const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
    const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
    if (_c.y < valid_rows) {
      const int idx = _c.y * row_stride + col_off + _c.x + _f * 16;
      if (kGQA) {
        mlx_atomic_fetch_add_explicit(out, AT_NAX(acc, _i), idx);
      } else {
        mlx_atomic_store_explicit(out, AT_NAX(acc, _i), idx);
      }
    }
  }
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C, int Ckpt>
[[kernel]] void gated_delta_vjp_fused_nax(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_ckpt, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
    device float* seg_states [[buffer(17)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  using _M16xDk = mlx::steel::NAXTile<float, 1, Dk / 16>;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
  const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;

  const int n_chunks = (T + C - 1) / C;
  // Ceiling: the floor form gives 0 when n_chunks < Ckpt, which would leave the
  // allocation empty and fault on the first read.
  const int n_ckpt = (n_chunks + Ckpt - 1) / Ckpt;
  const short tail = short(T - (n_chunks - 1) * C);

  // Bases, indexed by chunk rather than walked. With two nested loops the
  // incremental decrements are no longer expressible.
  auto q_base = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_base = k + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto dq_base = dq + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto dk_base = dk + b_idx * T * Hk * Dk + hk_idx * Dk;

  auto v_base = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto dv_base = dv + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto co_base = cot_o + b_idx * T * Hv * Dv + hv_idx * Dv;

  auto g_base = g + b_idx * T * Hv;
  auto beta_base = beta + b_idx * T * Hv;
  auto dg_base = dg + b_idx * T * Hv;
  auto db_base = db + b_idx * T * Hv;

  auto cm_base = chunk_mats + n * n_chunks * 3 * 256;
  auto cd_base = chunk_delta + n * n_chunks * C * Dv + dv_idx;
  auto ck_base = state_cache + (n * n_ckpt * Dv + dv_idx) * Dk;
  // Each simdgroup owns a distinct dv slice, so these regions are disjoint and
  // the replay needs no cross-simdgroup synchronisation.
  auto seg_base = seg_states + (n * Ckpt * Dv + dv_idx) * Dk;

  // Per-chunk pointers, assigned from the bases before each process_chunk call.
  auto q_ = q_base;
  auto k_ = k_base;
  auto dq_ = dq_base;
  auto dk_ = dk_base;
  auto v_ = v_base;
  auto dv_ = dv_base;
  auto co_ = co_base;
  auto g_ = g_base;
  auto beta_ = beta_base;
  auto dg_ = dg_base;
  auto db_ = db_base;
  auto i_chunk_mats = cm_base;
  auto i_chunk_delta = cd_base;
  auto i_seg = seg_base;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  threadgroup float red_scratch[(kNSG / 1) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

  float beta_fm[2];

  // Carried state gradient, dL/dS. Same [dv, dk] orientation as S_tile.
  _M16xDk dS_tile;
  dS_tile.load(i_cot_h, Dk);

  // Forward state at chunk entry, taken from the replayed segment.
  _M16xDk S_tile;

  _M16x32 K_tile, Q_tile;
  _M16x16 V_tile;
  _M16x16 delta_tile;
  _M16x16 QKt_tile, QKt_raw;
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

  _M16x16 tril_mask;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(tril_mask)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(tril_mask, _i) = (_c.x >= _c.y) ? 0.0f : 1.0f;
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

    // From the replayed segment rather than a per-chunk checkpoint.
    S_tile.load(i_seg, Dk);

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
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw * tri_decay;
    TUinv_tile = TWinv_tile * tri_decay;

    delta_tile.load(i_chunk_delta, Dv);

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

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    dODT = dODT * tri_decay;

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      K_tile = scale_rows(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);
      _M16x32 Kb_tile = scale_rows(K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      dq_acc = scale_rows(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is entirely inside this threadgroup.
      reduce_tile_tg<kNSG, kGQA>(
          dq_acc,
          red_scratch,
          dq_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    dVb_tile = scale_rows(dVb_tile, beta_fm);

    dVb_tile.store_rows(dv_ + dv_idx, Hv * Dv, valid_rows);

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    V_tile = scale_rows(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    dTU_tile = dTU_tile * tri_decay;

    dTinv_tile = dTW_tile + dTU_tile;

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    dA_tile = dA_tile * -1.0f;

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = scale_rows(dA_tile, beta_fm) * tril_mask;
    _M16x16 GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile = QKt_raw * dODT;
    _M16x16 R_tile = TWinv_tile * dTU_tile;
    dgam_pair = P_tile + R_tile;

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, t1;

      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      {
        _M16x32 QgdOS;
        MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
        dgam_row32 = dgam_row32 + scale_rows(QgdOS, row_exp) * Q_tile;
      }

      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);

      {
        _M16x32 dKb_tile;
        MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);
        KdKb = KdKb + dKb_tile * K_tile;
        dk_acc = dk_acc + scale_rows(dKb_tile, beta_fm);
      }

      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      t1 = scale_rows(t1, dec_exp);
      dk_acc = dk_acc + t1;

      {
        const _M16x32 KdKC = K_tile * t1;
        dgam_row32 = dgam_row32 - KdKC;
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(KdKC)::kElemsPerTile; _i++) {
          dgam_last += AT_NAX(KdKC, _i);
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp *
            (S_tile.frag_at(0, kk / 16)[_i] * dS_tile.frag_at(0, kk / 16)[_i] +
             S_tile.frag_at(0, kk / 16 + 1)[_i] *
                 dS_tile.frag_at(0, kk / 16 + 1)[_i]);
      }

      {
        _M16x32 TbK;
        K_tile = scale_rows(K_tile, beta_fm);
        MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
        dgam_row32 = fmadd(TbK, dW_raw, dgam_row32);
      }

      reduce_tile_tg<kNSG, kGQA>(
          dk_acc,
          red_scratch,
          dk_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);

    _M16x16 AKKt = dA_tile * KKt_tile;
    AKKt = AKKt * tril_mask;
    _M16x16 VdVb = fmadd(V_tile, TUddelta, AKKt);
    row_sum(dbeta_acc, VdVb);
    row_sum(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      float _b0 = dbeta_acc[0];
      float _b1 = dbeta_acc[1];
      _b0 += simd_shuffle_xor(_b0, ushort(1));
      _b0 += simd_shuffle_xor(_b0, ushort(8));
      _b1 += simd_shuffle_xor(_b1, ushort(1));
      _b1 += simd_shuffle_xor(_b1, ushort(8));

      if (fm < valid_rows) {
        db_st[fm] = _b0;
      }
      if (fm1 < valid_rows) {
        db_st[fm1] = _b1;
      }
    }

    dgam_row = reduce(dgam_row32);

    float dgam_acc[2] = {0.0f, 0.0f};
    dgam_row = dgam_row + dgam_pair;
    row_sum(dgam_acc, dgam_row);

    _M16x16 cs_tile;
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);

    const float dgam_last_red = simd_sum(dgam_last);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      float _d0 = dgam_acc[0];
      float _d1 = dgam_acc[1];
      _d0 += simd_shuffle_xor(_d0, ushort(1));
      _d0 += simd_shuffle_xor(_d0, ushort(8));
      _d1 += simd_shuffle_xor(_d1, ushort(1));
      _d1 += simd_shuffle_xor(_d1, ushort(8));

      _d0 -= AT_NAX(cs_tile, 0);
      _d1 -= AT_NAX(cs_tile, mlx::steel::BaseNAXFrag::kElemCols);

      if (fm == valid_rows - 1) {
        _d0 += dgam_last_red;
      }
      if (fm < valid_rows) {
        dg_st[fm] = _d0;
      }
      if (fm1 == valid_rows - 1) {
        _d1 += dgam_last_red;
      }
      if (fm1 < valid_rows) {
        dg_st[fm1] = _d1;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      Q_tile = scale_rows(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      K_tile = scale_rows(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      W_raw = scale_rows(-W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
      }
    }
  };

  // Walk segments in reverse. Each is replayed forward from its checkpoint to
  // recover the entry states, then walked backwards.
  // Backward over one segment: replay its entry states from the checkpoint,
  // then walk it in reverse. HasTail is a compile-time tag so valid_rows is the
  // constant C on every chunk of an interior segment, which folds away the
  // row guards in the dq/dk stores and the load_rows switch in the replay.
  auto process_segment = [&](const int seg,
                             auto tail_tag) __attribute__((always_inline)) {
    constexpr bool HasTail = decltype(tail_tag)::value;

    const int seg_start = seg * Ckpt;
    const int rem = n_chunks - seg_start;
    const int n_in_seg = (rem < Ckpt) ? rem : Ckpt;
    const int n_full = HasTail ? n_in_seg - 1 : n_in_seg;

    // delta is cached per chunk, so the replay is the state recurrence alone:
    //   S <- exp(gamma_C) S + K_dec^T delta
    // No T^-1, no W, no U -- roughly a tenth of a full chunk's work, which is
    // what makes trading Ckpt-fold memory for it worthwhile.
    //
    // Scoped so S is dead before process_chunk runs: it is a _M16xDk, and
    // having it live alongside S_tile and dS_tile would put three of them in
    // flight.
    {
      _M16xDk S;
      S.load(ck_base + seg * Dv * Dk, Dk);

      auto replay_one = [&](const int j,
                            const short valid_rows,
                            auto bounded_tag) __attribute__((always_inline)) {
        constexpr bool B = decltype(bounded_tag)::value;

        S.store(seg_base + j * Dv * Dk, Dk);

        const int c = seg_start + j;
        auto k_c = k_base + c * C * Hk * Dk;
        auto g_c = g_base + c * C * Hv;
        auto cd_c = cd_base + c * C * Dv;

        float gv = (thread_index_in_simdgroup < (uint)valid_rows)
            ? metal::fast::log(
                  metal::max(
                      g_c[thread_index_in_simdgroup * Hv + hv_idx], 1e-6))
            : 0.0f;
        auto gs = simd_prefix_inclusive_sum(gv);
        if (thread_index_in_simdgroup < C) {
          gamma[thread_index_in_simdgroup] = static_cast<float>(gs);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float g_last = gamma[C - 1];
        const float dec[2] = {
            metal::fast::exp(g_last - gamma[fm]),
            metal::fast::exp(g_last - gamma[fm1])};

        _M16x16 delta_tile;
        delta_tile.load(cd_c, Dv);

        S = S * metal::fast::exp(g_last);
        for (int kk = 0; kk < Dk; kk += 32) {
          _M16x32 Kd;
          if constexpr (B) {
            Kd.load_rows(k_c + kk, Dk * Hk, valid_rows);
          } else {
            Kd.load(k_c + kk, Dk * Hk);
          }
          Kd = scale_rows(Kd, dec);
          MMA16x32x16(S, kk / 16, delta_tile, true, 0, Kd, false, 0);
        }
      };

      for (int j = 0; j < n_full; j++) {
        replay_one(j, C, metal::false_type{});
      }
      if constexpr (HasTail) {
        replay_one(n_in_seg - 1, tail, metal::true_type{});
      }
    }

    auto set_chunk_ptrs = [&](const int j) __attribute__((always_inline)) {
      const int c = seg_start + j;
      q_ = q_base + c * C * Hk * Dk;
      k_ = k_base + c * C * Hk * Dk;
      dq_ = dq_base + c * C * Hk * Dk;
      dk_ = dk_base + c * C * Hk * Dk;
      v_ = v_base + c * C * Hv * Dv;
      dv_ = dv_base + c * C * Hv * Dv;
      co_ = co_base + c * C * Hv * Dv;
      g_ = g_base + c * C * Hv;
      beta_ = beta_base + c * C * Hv;
      dg_ = dg_base + c * C * Hv;
      db_ = db_base + c * C * Hv;
      i_chunk_mats = cm_base + c * 3 * 256;
      i_chunk_delta = cd_base + c * C * Dv;
      i_seg = seg_base + j * Dv * Dk;
    };

    if constexpr (HasTail) {
      set_chunk_ptrs(n_in_seg - 1);
      process_chunk(tail, metal::true_type{});
    }
    for (int j = n_full - 1; j >= 0; --j) {
      set_chunk_ptrs(j);
      process_chunk(C, metal::false_type{});
    }
  };

  // Only the last segment can hold a short chunk
  int seg = n_ckpt - 1;
  if (tail != C) {
    process_segment(seg, metal::true_type{});
    --seg;
  }
  for (; seg >= 0; --seg) {
    process_segment(seg, metal::false_type{});
  }

  dS_tile.store(o_dh, Dk);
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_vjp_fused_nax1(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_chunks, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    // Cached, chunk-local forward intermediates produced by the forward-save
    // pass (see gated_delta_fused_nax): avoids redoing the K@K^T contraction,
    // the 15-step Neumann inversion, and the W/U/S contraction for delta.
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
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

  // Cached forward intermediates, walked in reverse alongside c_state.
  auto i_chunk_mats =
      chunk_mats + n * n_chunks * 3 * 256 + (n_chunks - 1) * 3 * 256;
  auto i_chunk_delta =
      chunk_delta + n * n_chunks * C * Dv + dv_idx + (n_chunks - 1) * C * Dv;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) now covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  // (kNSG/2) destinations x 32 lanes x kElemsPerTile(_M16x32) floats.
  threadgroup float red_scratch[(kNSG / 1) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

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
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  // Backward tiles
  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

  _M16x16 I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    AT_NAX(I_tile, _i) = (_c.x == _c.y) ? 1.0f : 0.0f;
  }

  _M16x16 Ones_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(Ones_tile)::kElemsPerFrag; _i++) {
    AT_NAX(Ones_tile, _i) = 1.0f;
  }

  _M16x16 tril_mask;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(tril_mask)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i);
    AT_NAX(tril_mask, _i) = (_c.x >= _c.y) ? 0.0f : 1.0f;
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
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw * tri_decay;
    TUinv_tile = TWinv_tile * tri_decay;

    delta_tile.load(i_chunk_delta, Dv);

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

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    dODT = dODT * tri_decay;

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      K_tile = scale_rows(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);
      _M16x32 Kb_tile = scale_rows(K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      dq_acc = scale_rows(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is now entirely inside this threadgroup.
      reduce_tile_tg<kNSG, kGQA>(
          dq_acc,
          red_scratch,
          dq_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    dVb_tile = scale_rows(dVb_tile, beta_fm);

    dVb_tile.store_rows(dv_ + dv_idx, Hv * Dv, valid_rows);

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    V_tile = scale_rows(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    dTU_tile = dTU_tile * tri_decay;

    dTinv_tile = dTW_tile + dTU_tile;

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    dA_tile = dA_tile * -1.0f;

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = scale_rows(dA_tile, beta_fm) * tril_mask;
    _M16x16 GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile = QKt_raw * dODT;
    _M16x16 R_tile = TWinv_tile * dTU_tile;
    dgam_pair = P_tile + R_tile;

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, t1;

      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      {
        _M16x32 QgdOS;
        MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
        dgam_row32 = dgam_row32 + scale_rows(QgdOS, row_exp) * Q_tile;
      }

      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      dW_raw = scale_rows(-dW_raw, row_exp);

      {
        _M16x32 dKb_tile;
        MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);
        KdKb = KdKb + dKb_tile * K_tile; // KdKb_tmp folded away
        dk_acc = dk_acc + scale_rows(dKb_tile, beta_fm);
      }

      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      t1 = scale_rows(t1, dec_exp);
      dk_acc = dk_acc + t1;

      {
        const _M16x32 KdKC = K_tile * t1;
        dgam_row32 = dgam_row32 - KdKC;
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(KdKC)::kElemsPerTile; _i++) {
          dgam_last += AT_NAX(KdKC, _i);
        }
      }

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp *
            (S_tile.frag_at(0, kk / 16)[_i] * dS_tile.frag_at(0, kk / 16)[_i] +
             S_tile.frag_at(0, kk / 16 + 1)[_i] *
                 dS_tile.frag_at(0, kk / 16 + 1)[_i]);
      }

      {
        _M16x32 TbK;
        K_tile = scale_rows(K_tile, beta_fm);
        MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
        dgam_row32 = fmadd(TbK, dW_raw, dgam_row32);
      }

      reduce_tile_tg<kNSG, kGQA>(
          dk_acc,
          red_scratch,
          dk_,
          Hk * Dk,
          kk,
          valid_rows,
          sg_id,
          simd_lane_id);
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)

    load_seq(V_tile, v_ + dv_idx, Dv * Hv);

    _M16x16 AKKt = dA_tile * KKt_tile;
    AKKt = AKKt * tril_mask;
    _M16x16 VdVb = fmadd(V_tile, TUddelta, AKKt);
    row_sum(dbeta_acc, VdVb);
    row_sum(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      float _b0 = dbeta_acc[0];
      float _b1 = dbeta_acc[1];
      _b0 += simd_shuffle_xor(_b0, ushort(1));
      _b0 += simd_shuffle_xor(_b0, ushort(8));
      _b1 += simd_shuffle_xor(_b1, ushort(1));
      _b1 += simd_shuffle_xor(_b1, ushort(8));

      const short _r0 = fm;
      const short _r1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
      if (_r0 < valid_rows) {
        db_st[_r0] = _b0;
      }
      if (_r1 < valid_rows) {
        db_st[_r1] = _b1;
      }
    }

    dgam_row = reduce(dgam_row32);

    float dgam_acc[2] = {0.0f, 0.0f};
    dgam_row = dgam_row + dgam_pair;
    row_sum(dgam_acc, dgam_row);

    _M16x16 cs_tile;
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);

    const float dgam_last_red = simd_sum(dgam_last);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      float _d0 = dgam_acc[0];
      float _d1 = dgam_acc[1];
      _d0 += simd_shuffle_xor(_d0, ushort(1));
      _d0 += simd_shuffle_xor(_d0, ushort(8));
      _d1 += simd_shuffle_xor(_d1, ushort(1));
      _d1 += simd_shuffle_xor(_d1, ushort(8));

      const short _r0 = fm;
      const short _r1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;

      _d0 -= AT_NAX(cs_tile, 0);
      _d1 -= AT_NAX(cs_tile, mlx::steel::BaseNAXFrag::kElemCols);

      if (_r0 == valid_rows - 1) {
        _d0 += dgam_last_red;
      }
      if (_r0 < valid_rows) {
        dg_st[_r0] = _d0;
      }
      if (_r1 == valid_rows - 1) {
        _d1 += dgam_last_red;
      }
      if (_r1 < valid_rows) {
        dg_st[_r1] = _d1;
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      Q_tile = scale_rows(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      K_tile = scale_rows(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      W_raw = scale_rows(-W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
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
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
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
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
  }

  dS_tile.store(o_dh, Dk);
}

template <int kNSG, typename TileT>
METAL_FUNC void reduce_tile_tg0(
    thread TileT& acc,
    threadgroup float* scratch,
    const short sg_id,
    const ushort simd_lane_id) {
  constexpr short kE = TileT::kElemsPerTile;

  STEEL_PRAGMA_UNROLL
  for (short lo = kNSG / 2; lo > 0; lo >>= 1) {
    if (sg_id >= lo && sg_id < 2 * lo) {
      const short dst = sg_id - lo;
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        scratch[(dst * 32 + simd_lane_id) * kE + _i] = AT_NAX(acc, _i);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id < lo) {
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < kE; _i++) {
        AT_NAX(acc, _i) += scratch[(sg_id * 32 + simd_lane_id) * kE + _i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C>
[[kernel]] void gated_delta_vjp_fused_nax0(
    const device InT* q [[buffer(0)]], // [B, T, Hk, Dk]
    const device InT* k [[buffer(1)]], // [B, T, Hk, Dk]
    const device InT* v [[buffer(2)]], // [B, T, Hv, Dv]
    const device InT* g [[buffer(3)]], // [B, T, Hv]
    const device InT* beta [[buffer(4)]], // [B, T, Hv]
    const device InT* cot_o [[buffer(5)]], // [B, T, Hv, Dv]
    const device float* cot_h [[buffer(6)]], // [B, Hv, Dv, Dk]
    const device float* state_cache [[buffer(7)]], // [B, Hv, n_chunks, Dv, Dk]
    constant int& T [[buffer(8)]],
    device mlx_atomic<float>* dq [[buffer(9)]],
    device mlx_atomic<float>* dk [[buffer(10)]],
    device float* dv [[buffer(11)]],
    device float* dg [[buffer(12)]],
    device float* db [[buffer(13)]],
    device float* dh [[buffer(14)]],
    // Cached, chunk-local forward intermediates produced by the forward-save
    // pass (see gated_delta_fused_nax): avoids redoing the K@K^T contraction,
    // the 15-step Neumann inversion, and the W/U/S contraction for delta.
    const device float* chunk_mats [[buffer(15)]],
    const device float* chunk_delta [[buffer(16)]], // [B, Hv, n_chunks, C, Dv]
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

  // Cached forward intermediates, walked in reverse alongside c_state.
  auto i_chunk_mats =
      chunk_mats + n * n_chunks * 3 * 256 + (n_chunks - 1) * 3 * 256;
  auto i_chunk_delta =
      chunk_delta + n * n_chunks * C * Dv + dv_idx + (n_chunks - 1) * C * Dv;

  auto i_cot_h = cot_h + (n * Dv + dv_idx) * Dk;
  auto o_dh = dh + (n * Dv + dv_idx) * Dk;

  // One threadgroup per (b, hv) now covers every Dv slice, so the Dv reduction
  // never leaves threadgroup memory.
  constexpr int kNSG = Dv / 16;
  // Under GQA several threadgroups still share an hk, so dq/dk keep an atomic
  // for that axis -- but with kNSG writers folded away first.
  constexpr bool kGQA = (Hv != Hk);

  threadgroup float gamma_all[C * kNSG];
  threadgroup float* gamma = gamma_all + sg_id * C;

  // (kNSG/2) destinations x 32 lanes x kElemsPerTile(_M16x32) floats.
  threadgroup float red_scratch[(kNSG / 2) * 32 * 16];
  threadgroup float db_stage[kNSG * C];
  threadgroup float dg_stage[kNSG * C];

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
  _M16x16 KKt_tile;
  _M16x16 TWinv_tile, TUinv_tile;
  _M16x16 TMP_tile;

  // Backward tiles
  _M16x16 dout_tile;
  _M16x16 ddelta_tile;
  _M16x16 dTU_tile, dTW_tile;
  _M16x16 dTinv_tile;
  _M16x16 dA_tile, G_tile;
  _M16x16 tri_decay;

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
    const float gamma_last_exp = metal::fast::exp(gamma_last);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    // Per-lane decay factors. gamma[j] == gamma[valid_rows-1] for
    // j >= valid_rows (the prefix sum of zeros), so the tail chunk needs no
    // special case here.
    const float row_exp[2] = {
        metal::fast::exp(gamma[fm]), metal::fast::exp(gamma[fm1])};
    const float dec_exp[2] = {
        metal::fast::exp(gamma_last - gamma[fm]),
        metal::fast::exp(gamma_last - gamma[fm1])};

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(tri_decay)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      AT_NAX(tri_decay, _i) =
          (_c.x > _c.y) ? 0.f : metal::fast::exp(gamma[_c.y] - gamma[_c.x]);
    }

    KKt_tile.load(i_chunk_mats, 16);
    TWinv_tile.load(i_chunk_mats + 256, 16);
    QKt_raw.load(i_chunk_mats + 512, 16);

    QKt_tile = QKt_raw;
    MUL_NAX(QKt_tile, QKt_tile, tri_decay);

    TUinv_tile = TWinv_tile;
    MUL_NAX(TUinv_tile, TUinv_tile, tri_decay);

    delta_tile.load(i_chunk_delta, Dv);

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

    // dODT = (dO @ delta.T * D)
    _M16x16 dODT;
    MM16x16x16(dODT, 0, dout_tile, false, 0, delta_tile, true, 0);
    MUL_NAX(dODT, dODT, tri_decay);

    // cannot fuse this because ddelta is used by the next loop
    MM16x16x16(ddelta_tile, 0, QKt_tile, true, 0, dout_tile, false, 0);
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      SCALE2_P(K_tile, dec_exp);
      MMA16x16x32(ddelta_tile, 0, K_tile, false, 0, dS_tile, true, kk / 16);
    }

    // ddelta = M^T @ dO + K_dec @ dS^T
    // dq = gamma * (dO @ S) + dODT @ K
    dTW_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dW_raw;
      _M16x32 dq_acc;
      _M16x32 Kb_tile;
      load_seq(K_tile, k_ + kk, Dk * Hk);

      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      NSCALE_ROW_P(dW_raw, row_exp);

      SCALE_BETA_NAX_O(Kb_tile, K_tile, beta_fm);
      MMA16x16x32(dTW_tile, 0, dW_raw, false, 0, Kb_tile, true, 0);

      MM16x32x16(dq_acc, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      SCALE_ROW_P(dq_acc, row_exp);
      MMA16x32x16(dq_acc, 0, dODT, false, 0, K_tile, false, 0);

      // dq reduces over Dv, which is now entirely inside this threadgroup.
      reduce_tile_tg<kNSG>(dq_acc, red_scratch, sg_id, simd_lane_id);

      if (sg_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(dq_acc)::kElemsPerTile; _i++) {
          const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
          const short _fn = _c.x + _f * 16;
          const short _fm = _c.y;
          if (_fm < valid_rows) {
            const int idx = _fm * Hk * Dk + kk + _fn;
            if (kGQA) {
              mlx_atomic_fetch_add_explicit(dq_, AT_NAX(dq_acc, _i), idx);
            } else {
              // Dv reduction done in threadgroup memory: single writer.
              mlx_atomic_store_explicit(dq_, AT_NAX(dq_acc, _i), idx);
            }
          }
        }
      }
    }

    // dv = B(Tu.T @ ddelta).  Indexed by dv_idx, so no reduction.
    _M16x16 dVb_tile;
    MM16x16x16(dVb_tile, 0, TUinv_tile, true, 0, ddelta_tile, false, 0);
    _M16x16 TUddelta = dVb_tile; // pre-beta, reused by dbeta
    SCALE_BETA_NAX(dVb_tile, beta_fm);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(dVb_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      if (_c.y < valid_rows) {
        dv_[_c.y * Hv * Dv + dv_idx + _c.x] = AT_NAX(dVb_tile, _i);
      }
    }

    // dT_U = ddelta @ (beta * V).T
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(dTU_tile, 0, ddelta_tile, false, 0, V_tile, true, 0);
    MUL_NAX(dTU_tile, dTU_tile, tri_decay); // dTinv = dT_W + (dT_U * D).

    ADD_NAX(dTinv_tile, dTW_tile, dTU_tile);

    // dA = -T.T @ dTinv @ T.T
    MM16x16x16(TMP_tile, 0, TWinv_tile, true, 0, dTinv_tile, false, 0);
    MM16x16x16(dA_tile, 0, TMP_tile, false, 0, TWinv_tile, true, 0);
    SCALE_NAX(dA_tile, -1.0f);

    // G = tril_(dA) * beta, and GGt = G + G.T
    G_tile = dA_tile;
    SCALE_TRIEQ_NAX1(G_tile, beta_fm);
    _M16x16 GGt_tile;
    GGt_tile = G_tile;
    MMA16x16x16(GGt_tile, 0, I_tile, true, 0, G_tile, true, 0);

    // dgamma pair sites
    _M16x16 P_tile;
    _M16x16 R_tile;

    MUL_NAX(P_tile, QKt_raw, dODT);
    MUL_NAX(R_tile, TWinv_tile, dTU_tile);
    ADD_NAX(dgam_pair, P_tile, R_tile);

    _M16x32 KdKb; // rowsum(K * dK_b) source, needed for dbeta
    KdKb.clear();
    _M16x32 dgam_row32;
    dgam_row32.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 dk_acc, dW_raw, dKb_tile, t1, KdKb_tmp;
      _M16x32 QgdOS, TbK, KdKC_blk;

      load_seq(Q_tile, q_ + kk, Hk * Dk); // [16 x 32]
      load_seq(K_tile, k_ + kk, Dk * Hk); // [16 x 32]

      // gamma: + rowsum(Q * gamma*(dO @ S))
      MM16x32x16(QgdOS, 0, dout_tile, false, 0, S_tile, false, kk / 16);
      SCALE_ROW_P(QgdOS, row_exp);
      MULA_NAX(dgam_row32, QgdOS, Q_tile);

      // dODT.T @ Q  and  (G + G.T) @ K
      MM16x32x16(dk_acc, 0, dODT, true, 0, Q_tile, false, 0);
      MMA16x32x16(dk_acc, 0, GGt_tile, false, 0, K_tile, false, 0);

      // B (T.T @ dW_raw),  dW_raw = -gamma * (ddelta @ S)
      MM16x32x16(dW_raw, 0, ddelta_tile, false, 0, S_tile, false, kk / 16);
      NSCALE_ROW_P(dW_raw, row_exp);
      MM16x32x16(dKb_tile, 0, TWinv_tile, true, 0, dW_raw, false, 0);

      // dbeta term uses dK_b before the beta scaling, with raw K
      MUL_NAX(KdKb_tmp, dKb_tile, K_tile);
      ADD_NAX(KdKb, KdKb, KdKb_tmp);

      SCALE_BETA_NAX(dKb_tile, beta_fm);
      ADD_NAX(dk_acc, dk_acc, dKb_tile);

      // D_C * (delta @ dS)
      MM16x32x16(t1, 0, delta_tile, false, 0, dS_tile, false, kk / 16);
      SCALE2_P(t1, dec_exp);
      ADD_NAX(dk_acc, dk_acc, t1);

      // gamma stuff
      MUL_NAX(KdKC_blk, K_tile, t1);
      SUB_NAX(dgam_row32, dgam_row32, KdKC_blk);
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(KdKC_blk)::kElemsPerTile; _i++) {
        dgam_last += AT_NAX(KdKC_blk, _i);
      }

      // gamma stuff: gamma_C * sum(S * dS)
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
        dgam_last += gamma_last_exp * S_tile.frag_at(0, kk / 16)[_i] *
            dS_tile.frag_at(0, kk / 16)[_i];
        dgam_last += gamma_last_exp * S_tile.frag_at(0, kk / 16 + 1)[_i] *
            dS_tile.frag_at(0, kk / 16 + 1)[_i];
      }

      // gamma stuff
      SCALE_BETA_NAX(K_tile, beta_fm);
      SCALE_NAX(dW_raw, -1.0f);
      MM16x32x16(TbK, 0, TWinv_tile, false, 0, K_tile, false, 0);
      MULS_NAX(dgam_row32, TbK, dW_raw);

      // dk reduces over Dv the same way dq does.
      reduce_tile_tg<kNSG>(dk_acc, red_scratch, sg_id, simd_lane_id);

      if (sg_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short _i = 0; _i < decltype(dk_acc)::kElemsPerTile; _i++) {
          const short _f = _i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short _w = _i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_w); // {fn, fm}
          if (_c.y < valid_rows) {
            const int idx = _c.y * Hk * Dk + kk + _c.x + _f * 16;
            if (kGQA) {
              mlx_atomic_fetch_add_explicit(dk_, AT_NAX(dk_acc, _i), idx);
            } else {
              mlx_atomic_store_explicit(dk_, AT_NAX(dk_acc, _i), idx);
            }
          }
        }
      }
    }

    // dbeta = rowsum(V * dV_b) + rowsum(K * dK_b) + rowsum(tril_(dA) * KKt)
    _M16x16 VdVb;
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    MUL_NAX(VdVb, V_tile, TUddelta);

    _M16x16 AKKt;
    TRIL_NAX(AKKt, dA_tile);
    MUL_NAX(AKKt, AKKt, KKt_tile);

    ADD_NAX(VdVb, VdVb, AKKt);
    ROWSUM1_NAX(dbeta_acc, VdVb);
    ROWSUM1_NAX(dbeta_acc, KdKb);

    {
      threadgroup float* db_st = db_stage + sg_id * C;
      STEEL_PRAGMA_UNROLL
      for (short _r = 0; _r < 2; _r++) {
        float _b = dbeta_acc[_r];
        _b += simd_shuffle_xor(_b, ushort(1));
        _b += simd_shuffle_xor(_b, ushort(8));

        const short _row = fm + _r * mlx::steel::BaseNAXFrag::kElemRowsJump;
        if (_row < valid_rows) {
          db_st[_row] = _b;
        }
      }
    }

    // Fold the two fragments: only the row sum of dgam_row is used downstream,
    // and row sums are additive across fragments.
    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < mlx::steel::BaseNAXFrag::kElemsPerFrag; _i++) {
      AT_NAX(dgam_row, _i) = AT_NAX(dgam_row32, _i) +
          AT_NAX(dgam_row32, mlx::steel::BaseNAXFrag::kElemsPerFrag + _i);
    }

    // reduce and store dgamma: rowsum(row + pair) - colsum(pair)
    ADD_NAX(dgam_row, dgam_row, dgam_pair);
    _M16x16 dgam_tile, cs_tile;
    MM16x16x16(dgam_tile, 0, dgam_row, false, 0, Ones_tile, false, 0);
    MM16x16x16(cs_tile, 0, dgam_pair, true, 0, Ones_tile, false, 0);
    SUB_NAX(dgam_tile, dgam_tile, cs_tile);

    {
      threadgroup float* dg_st = dg_stage + sg_id * C;
      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(dgam_tile)::kElemsPerFrag; _i++) {
        const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
        if (_c.x == 0) {
          dg_st[_c.y] = AT_NAX(dgam_tile, _i);
        }
      }
      // The gamma_{C-1}-only term lands on the same slot the loop just wrote,
      // so the lanes must be ordered before lane 0 accumulates into it.
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const float dgam_last_red = simd_sum(dgam_last);
      if (simd_lane_id == 0 && valid_rows > 0) {
        dg_st[valid_rows - 1] += dgam_last_red;
      }
    }

    // One barrier serves both stages; simdgroup 0 commits.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0 && thread_index_in_simdgroup < (uint)valid_rows) {
      float b_sum = 0.0f;
      float g_sum = 0.0f;
      STEEL_PRAGMA_UNROLL
      for (short _s = 0; _s < kNSG; _s++) {
        b_sum += db_stage[_s * C + thread_index_in_simdgroup];
        g_sum += dg_stage[_s * C + thread_index_in_simdgroup];
      }
      const int idx = thread_index_in_simdgroup * Hv + hv_idx;
      db_[idx] = b_sum;
      dg_[idx] = g_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dS update
    // dS = gamma_C * dS + dO.T @ (gamma * Q) - ddelta.T @ W
    SCALE_NAX(dS_tile, gamma_last_exp);
    for (int kk = 0; kk < Dk; kk += 32) {
      _M16x32 term, W_raw;
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Dk * Hk);

      // dO.T @ (gamma * Q)  -> [dv, dk]
      SCALE_ROW_P(Q_tile, row_exp);
      MM16x32x16(term, 0, dout_tile, true, 0, Q_tile, false, 0);

      // - ddelta.T @ W,  with W = gamma * (TWinv @ beta*K)
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_raw, 0, TWinv_tile, false, 0, K_tile, false, 0);
      NSCALE_ROW_P(W_raw, row_exp);
      MMA16x32x16(term, 0, ddelta_tile, true, 0, W_raw, false, 0);

      STEEL_PRAGMA_UNROLL
      for (short _i = 0; _i < decltype(term)::kElemsPerFrag; _i++) {
        dS_tile.frag_at(0, kk / 16)[_i] += AT_NAX(term, _i);
        dS_tile.frag_at(0, kk / 16 + 1)[_i] +=
            AT_NAX(term, decltype(term)::kElemsPerFrag + _i);
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
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
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
    i_chunk_mats -= 3 * 256;
    i_chunk_delta -= C * Dv;
  }

  dS_tile.store(o_dh, Dk);
}

// Postprocessing dg. Similar idea to what is done in the triton kernel.
// The kernel computes the gradient with respect to the log.
template <typename InT, int C>
[[kernel]] void gated_delta_dgamma_to_dg(
    const device InT* g [[buffer(0)]],
    device float* dg [[buffer(1)]], // in: dL/dgamma, out: dL/dg
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
    dg[idx] = acc / gv;
  }
}
