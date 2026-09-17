#pragma once

#include "mlx/backend/metal/kernels/gated_delta_nax_ops.h"

using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

///////////////////////////////////////////////////////////////////////////////
// Function constants
///////////////////////////////////////////////////////////////////////////////

constant bool save_state_cache [[function_constant(200)]];

template <typename InT, int Dk, int Dv, int Hk, int Hv, int C, int Ckpt>
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
    device float* state_cache [[buffer(9)]], // [B, Hv, n_ckpt, Dv, Dk]
    device float* chunk_mats [[buffer(10)]], // [B, Hv, n_chunks, 3, 16, 16]
    device float* chunk_delta [[buffer(11)]], // [B, Hv, n_chunks, C, Dv]
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

  const int n_chunks = (T + C - 1) / C;
  const int n_ckpt = (n_chunks + Ckpt - 1) / Ckpt;
  auto c_state = state_cache + (n * n_ckpt * Dv + dv_idx) * Dk;

  auto o_chunk_mats = chunk_mats + n * n_chunks * 3 * 256;
  auto o_chunk_delta = chunk_delta + n * n_chunks * C * Dv + dv_idx;

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

  // Which chunk process_chunk is on, so it knows whether this is a segment
  // boundary.
  int chunk_idx = 0;

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

    // Checkpoint the state entering this chunk before anything mutates it, on
    // segment boundaries only.
    if (save_state_cache && (chunk_idx % Ckpt) == 0) {
      S_tile.store(c_state, Dk);
      c_state += Dv * Dk;
    }

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(float(g_[thread_index_in_simdgroup * Hv + hv_idx]), 1e-6))
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

    // KKt_raw is identical for every Dv tile of this chunk; only the first
    // tile needs to write it out for the vjp kernel to reuse.
    if (save_state_cache && dv_idx == 0) {
      KKt_tile.store(o_chunk_mats, 16);
    }

    KKtK_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);
    SUB_NAX(Tinv_tile, I_tile, KKtK_tile);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, Tinv_tile, false, 0);
      SUB_NAX(Tinv_tile, I_tile, TMP_tile);
    }

    // Same as KKt_raw: cache the pre-decay inverse before it is overwritten
    // below by the gamma-scaled (TUinv) version.
    if (save_state_cache && dv_idx == 0) {
      Tinv_tile.store(o_chunk_mats + 256, 16);
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

    QKt_tile.clear();
    tmp_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Hk * Dk);
      MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, K_tile, true, 0);
      SCALE_ROW_NAX(Q_tile, gamma);
      MMA16x16x32(tmp_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16);
    }

    if (save_state_cache) {
      if (dv_idx == 0) {
        QKt_tile.store(o_chunk_mats + 512, 16);
      }
      // Every Dv tile stores its own slice.
      delta_tile.store(o_chunk_delta, Dv);
      o_chunk_mats += 3 * 256;
      o_chunk_delta += C * Dv;
    }

    // Output path runs unconditionally: the forward serves both inference and
    // the training save, so y and state_out are always needed.
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

    chunk_idx++;
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
