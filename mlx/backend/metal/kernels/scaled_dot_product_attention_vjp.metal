// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/gated_delta_nax_ops.h"

using namespace metal;

// Declared so the host may bind it; the causal path is not implemented.
constant bool Causal [[function_constant(200)]];


#define NAX_LOG2E 1.44269504088896340736f

///////////////////////////////////////////////////////////////////////////////
// OdO[b, h, i] = <dO_i, O_i>
///////////////////////////////////////////////////////////////////////////////

template <typename InT, int Dv>
[[kernel]] void sdpa_vjp_odo(
    const device InT* o [[buffer(0)]], // [B, H, qL, Dv]
    const device InT* cot_o [[buffer(1)]], // [B, H, qL, Dv]
    device float* odo [[buffer(2)]], // [B, H, qL]
    constant int& qL [[buffer(3)]],
    uint3 tpg [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  const uint bh = tpg.z;
  const int i = int(tpg.y);

  if (i >= qL) {
    return;
  }

  const size_t row = (size_t)bh * qL + i;
  auto o_i = o + row * Dv;
  auto co_i = cot_o + row * Dv;

  float acc = 0.0f;
  for (int d = int(simd_lane_id); d < Dv; d += 32) {
    acc += float(o_i[d]) * float(co_i[d]);
  }
  acc = simd_sum(acc);

  if (simd_lane_id == 0) {
    odo[row] = acc;
  }
}


template <typename InT, int D, int Dv, int H, int Hk, int BQ, int BK, int WM, int WN, bool Causal>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
sdpa_vjp_dv(
    const device InT* q,      // [B, H,  qL, D]
    const device InT* k,      // [B, Hk, kL, D]
    const device float* lse,  // [B, H, qL]  from the forward, natural log
    const device InT* cot_o,  // [B, H,  qL, Dv]  dO
    constant int& qL,
    constant int& kL,
    constant float& scale,
    device InT* dv,           // [B, Hk, kL, Dv]
    uint3 tpg [[thread_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  constexpr int kNWarps = WM * WN;

  constexpr int TK = BK / (kNWarps * 16);
  constexpr int TQ = BQ / 16;
  constexpr int Rep = H / Hk;
  static_assert(TK == 1, "BK must equal kNWarps * 16");
  static_assert(BQ % 16 == 0, "BQ must be a multiple of 16");
  static_assert(D % 32 == 0 && Dv % 32 == 0, "head dims must pair for K=32");
  static_assert(H % Hk == 0, "H must be a multiple of Hk");

  constexpr int kFragRowsD = 16 * D;
  constexpr int kFragRowsDv = 16 * Dv;

  const uint bh = tpg.z;
  const uint b_idx = bh / Hk;
  const uint kh_idx = bh % Hk;

  const int kl_idx = BK * tpg.y;
  const int tm = 16 * TK * int(simd_group_id);

  auto k_j = k + ((b_idx * Hk + kh_idx) * kL + kl_idx + tm) * D;
  auto dv_ = dv + ((b_idx * Hk + kh_idx) * kL + kl_idx + tm) * Dv;

  const float scale2 = scale * 1.44269504088896340736f;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  constexpr short kRowsPT = TQ * 2;

  mlx::steel::NAXTile<float, 1, TQ> S_tile;
  mlx::steel::NAXTile<float, 1, 2> Q_tile, K_tile, dO_tile;
  mlx::steel::NAXTile<float, 1, Dv / 16> dV_tile;
  dV_tile.clear();

  const int k_first = kl_idx + tm;

  int qb0 = 0;
  int k_last = 0;
  int c_off = 0;
  if constexpr (Causal) {
    c_off = kL - qL;
    k_last = k_first + 16 * TK - 1;
    const int r = k_first - c_off;
    qb0 = (r > 0) ? ((r / BQ) * BQ) : 0;
  }

  for (int hv = 0; hv < Rep; hv++) {
    const uint qh_idx = kh_idx * Rep + uint(hv);

    auto q_i = q + ((b_idx * H + qh_idx) * qL + qb0) * D;
    auto co_i = cot_o + ((b_idx * H + qh_idx) * qL + qb0) * Dv;
    auto lse_i = lse + (b_idx * H + qh_idx) * qL + qb0;

    for (int qb = qb0; qb < qL; qb += BQ) {
      bool straddle = false;
      if constexpr (Causal) {
        straddle = (k_last > qb + c_off);
      }

      float lse_local[kRowsPT];
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short g = 0; g < 2; g++) {
          const int r = iq * 16 + g * mlx::steel::BaseNAXFrag::kElemRowsJump + fm;
          lse_local[iq * 2 + g] = lse_i[r] * 1.44269504088896340736f;
        }
      }

      S_tile.clear();
      // STEEL_PRAGMA_UNROLL
      for (short id = 0; id < D; id += 32) {
        K_tile.load(k_j + id, D);
        auto q_p = q_i + id;
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          Q_tile.load(q_p, D);
          MMA16x16x32(S_tile, iq, Q_tile, false, 0, K_tile, true, 0);
          q_p += kFragRowsD;
        }
      }

      if constexpr (Causal) {
        for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
          const short f = i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short w = i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 c = mlx::steel::BaseNAXFrag::get_coord(w);

          float p = metal::fast::exp2(
              AT_NAX(S_tile, i) * scale2 - lse_local[f * 2 + (w >> 2)]);

          if (straddle) {
            p = ((k_first + c.x) > (qb + f * 16 + c.y + c_off)) ? 0.0f : p;
          }

          AT_NAX(S_tile, i) = p;
        }
      } else {
        for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
          const short f = i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short w = i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          AT_NAX(S_tile, i) = metal::fast::exp2(
              AT_NAX(S_tile, i) * scale2 - lse_local[f * 2 + (w >> 2)]);
        }
      }

      for (short id = 0; id < Dv; id += 32) {
        auto co_p = co_i + id;
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          dO_tile.load(co_p, Dv);
          MMA16x32x16(dV_tile, id / 16, S_tile, true, iq, dO_tile, false, 0);
          co_p += kFragRowsDv;
        }
      }

      q_i += BQ * D;
      co_i += BQ * Dv;
      lse_i += BQ;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < Dv / 16; id++) {
    const thread auto& fg = dV_tile.frag_at(0, id);
    auto dv_p = dv_ + id * 16;
    for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(e);
      dv_p[c.y * Dv + c.x] = InT(fg[e]);
    }
  }
}

template <typename InT, int D, int Dv, int H, int Hk, int BQ, int BK, int WM, int WN, bool Causal>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
sdpa_vjp_dk(
    const device InT* q,      // [B, H,  qL, D]
    const device InT* k,      // [B, Hk, kL, D]
    const device InT* v,      // [B, Hk, kL, Dv]
    const device float* odo,  // [B, H, qL]
    const device float* lse,  // [B, H, qL]  from the forward, natural log
    const device InT* cot_o,  // [B, H,  qL, Dv]  dO
    constant int& qL,
    constant int& kL,
    constant float& scale,
    device InT* dk,           // [B, Hk, kL, D]
    uint3 tpg [[thread_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  constexpr int kNWarps = WM * WN;

  constexpr int TK = BK / (kNWarps * 16);
  constexpr int TQ = BQ / 16;
  constexpr int Rep = H / Hk;
  static_assert(TK == 1, "BK must equal kNWarps * 16");
  static_assert(BQ % 16 == 0, "BQ must be a multiple of 16");
  static_assert(D % 32 == 0 && Dv % 32 == 0, "head dims must pair for K=32");
  static_assert(H % Hk == 0, "H must be a multiple of Hk");

  constexpr int kFragRowsD = 16 * D;
  constexpr int kFragRowsDv = 16 * Dv;

  const uint bh = tpg.z;
  const uint b_idx = bh / Hk;
  const uint kh_idx = bh % Hk;

  const int kl_idx = BK * tpg.y;
  const int tm = 16 * TK * int(simd_group_id);

  auto k_j = k + ((b_idx * Hk + kh_idx) * kL + kl_idx + tm) * D;
  auto v_j = v + ((b_idx * Hk + kh_idx) * kL + kl_idx + tm) * Dv;
  auto dk_ = dk + ((b_idx * Hk + kh_idx) * kL + kl_idx + tm) * D;

  const float scale2 = scale * 1.44269504088896340736f;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  constexpr short kRowsPT = TQ * 2;

  mlx::steel::NAXTile<float, 1, TQ> S_tile;
  mlx::steel::NAXTile<float, 1, TQ> dP_tile;
  mlx::steel::NAXTile<float, 1, 2> Q_tile, K_tile, dO_tile;
  mlx::steel::NAXTile<float, 1, D / 16> dK_tile;
  dK_tile.clear();

  const int k_first = kl_idx + tm;

  int qb0 = 0;
  int k_last = 0;
  int c_off = 0;
  if constexpr (Causal) {
    c_off = kL - qL;
    k_last = k_first + 16 * TK - 1;
    const int r = k_first - c_off;
    qb0 = (r > 0) ? ((r / BQ) * BQ) : 0;
  }

  for (int hv = 0; hv < Rep; hv++) {
    const uint qh_idx = kh_idx * Rep + uint(hv);

    auto q_i = q + ((b_idx * H + qh_idx) * qL + qb0) * D;
    auto co_i = cot_o + ((b_idx * H + qh_idx) * qL + qb0) * Dv;
    auto odo_i = odo + (b_idx * H + qh_idx) * qL + qb0;
    auto lse_i = lse + (b_idx * H + qh_idx) * qL + qb0;

    for (int qb = qb0; qb < qL; qb += BQ) {
      bool straddle = false;
      if constexpr (Causal) {
        straddle = (k_last > qb + c_off);
      }

      float lse_local[kRowsPT];
      float odo_local[kRowsPT];
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        for (short g = 0; g < 2; g++) {
          const int r =
              iq * 16 + g * mlx::steel::BaseNAXFrag::kElemRowsJump + fm;
          lse_local[iq * 2 + g] = lse_i[r] * 1.44269504088896340736f;
          odo_local[iq * 2 + g] = odo_i[r];
        }
      }

      S_tile.clear();
      for (short id = 0; id < D; id += 32) {
        K_tile.load(k_j + id, D);
        auto q_p = q_i + id;
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          Q_tile.load(q_p, D);
          MMA16x16x32(S_tile, iq, Q_tile, false, 0, K_tile, true, 0);
          q_p += kFragRowsD;
        }
      }

      if constexpr (Causal) {
        for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
          const short f = i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short w = i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short2 c = mlx::steel::BaseNAXFrag::get_coord(w);

          float p = metal::fast::exp2(
              AT_NAX(S_tile, i) * scale2 - lse_local[f * 2 + (w >> 2)]);

          if (straddle) {
            p = ((k_first + c.x) > (qb + f * 16 + c.y + c_off)) ? 0.0f : p;
          }

          AT_NAX(S_tile, i) = p;
        }
      } else {
        for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
          const short f = i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
          const short w = i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
          AT_NAX(S_tile, i) = metal::fast::exp2(
              AT_NAX(S_tile, i) * scale2 - lse_local[f * 2 + (w >> 2)]);
        }
      }

      dP_tile.clear();
      for (short id = 0; id < Dv; id += 32) {
        mlx::steel::NAXTile<float, 1, 2> V_tile;
        V_tile.load(v_j + id, Dv);
        auto co_p = co_i + id;
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          dO_tile.load(co_p, Dv);
          MMA16x16x32(dP_tile, iq, dO_tile, false, 0, V_tile, true, 0);
          co_p += kFragRowsDv;
        }
      }

      for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
        const short f = i / mlx::steel::BaseNAXFrag::kElemsPerFrag;
        const short w = i % mlx::steel::BaseNAXFrag::kElemsPerFrag;
        AT_NAX(S_tile, i) *=
            scale * (AT_NAX(dP_tile, i) - odo_local[f * 2 + (w >> 2)]);
      }

      for (short id = 0; id < D; id += 32) {
        auto q_p = q_i + id;
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          Q_tile.load(q_p, D);
          MMA16x32x16(dK_tile, id / 16, S_tile, true, iq, Q_tile, false, 0);
          q_p += kFragRowsD;
        }
      }

      q_i += BQ * D;
      co_i += BQ * Dv;
      odo_i += BQ;
      lse_i += BQ;
    }
  }

  for (short id = 0; id < D / 16; id++) {
    const thread auto& fg = dK_tile.frag_at(0, id);
    auto dk_p = dk_ + id * 16;
    for (short e = 0; e < mlx::steel::BaseNAXFrag::kElemsPerFrag; e++) {
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(e);
      dk_p[c.y * D + c.x] = InT(fg[e]);
    }
  }
}
template <typename InT, int D, int Dv, int H, int Hk, int BQ, int BK, int WM, int WN, bool Causal>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
sdpa_vjp_dq(
    const device InT* q,      // [B, H,  qL, D]
    const device InT* k,      // [B, Hk, kL, D]
    const device InT* v,      // [B, Hk, kL, Dv]
    const device float* odo,  // [B, H, qL]
    const device float* lse,  // [B, H, qL]  from the forward, natural log
    const device InT* cot_o,  // [B, H,  qL, Dv]  dO
    constant int& qL,
    constant int& kL,
    constant float& scale,
    device InT* dq,           // [B, H,  qL, D]
    uint3 tpg [[thread_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  constexpr int kNWarps = WM * WN;

  constexpr int TQ = BQ / (kNWarps * 16);
  constexpr int TK = BK / 16;
  static_assert(TQ == 1, "BQ must equal kNWarps * 16");
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(D % 32 == 0 && Dv % 32 == 0, "head dims must pair for K=32");

  constexpr short kEPF = mlx::steel::BaseNAXFrag::kElemsPerFrag;

  constexpr int kFragRowsD = 16 * D;
  constexpr int kFragRowsDv = 16 * Dv;

  const uint bh = tpg.z;
  const uint b_idx = bh / H;
  const uint qh_idx = bh % H;
  const uint kh_idx = qh_idx / (H / Hk);

  const int ql_idx = BQ * tpg.y;

  const int tm = 16 * TQ * int(simd_group_id);
  const int q_row0 = ql_idx + tm;

  const short q_rows = short(metal::min(16, qL - q_row0));
  if (q_rows <= 0) {
    return;
  }

  auto q_ = q + ((b_idx * H + qh_idx) * qL + q_row0) * D;
  auto co_ = cot_o + ((b_idx * H + qh_idx) * qL + q_row0) * Dv;
  auto dq_ = dq + ((b_idx * H + qh_idx) * qL + q_row0) * D;

  auto odo_ = odo + (b_idx * H + qh_idx) * qL + q_row0;
  auto lse_ = lse + (b_idx * H + qh_idx) * qL + q_row0;

  auto k_j = k + (b_idx * Hk + kh_idx) * kL * D;
  auto v_j = v + (b_idx * Hk + kh_idx) * kL * Dv;

  const float scale2 = scale * NAX_LOG2E;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  constexpr short kRowsPT = TQ * 2;

  float lse_local[kRowsPT];
  float odo_local[kRowsPT];
  for (short g = 0; g < kRowsPT; g++) {
    const short r = g * mlx::steel::BaseNAXFrag::kElemRowsJump + fm;
    lse_local[g] = (r < q_rows) ? lse_[r] * NAX_LOG2E : INFINITY;
    odo_local[g] = (r < q_rows) ? odo_[r] : 0.0f;
  }

  mlx::steel::NAXTile<float, 1, TK> S_tile;
  mlx::steel::NAXTile<float, 1, TK> dP_tile;
  _M16x32 Q_tile, K_tile;
  _M16x32 dO_tile, V_tile;

  mlx::steel::NAXTile<float, 1, D / 16> dQ_tile;
  dQ_tile.clear();

  int q_last = 0;
  int c_off = 0;
  if constexpr (Causal) {
    q_last = q_row0 + 16 * TQ - 1;
    c_off = kL - qL;
  }

  auto load_q = [&](thread auto& tile, auto src, int ld) {
    tile.load_rows(src, ld, q_rows);
  };

  auto do_kblock = [&](auto SafeTag, int kb) {
    constexpr bool KSafe = decltype(SafeTag)::value;

    auto k_valid = [&](short ik) -> short {
      if constexpr (KSafe) {
        return short(metal::clamp(kL - (kb + ik * 16), 0, 16));
      } else {
        return 16;
      }
    };

    auto load_kv = [&](thread auto& tile, auto src, int ld, short rows) {
      if constexpr (KSafe) {
        tile.load_rows(src, ld, rows);
      } else {
        tile.load(src, ld);
      }
    };

    bool straddle = false;
    if constexpr (Causal) {
      straddle = (kb + BK - 1 > q_row0 + c_off);
    }

    S_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      const short kr = k_valid(ik);
      if (kr <= 0) {
        continue;
      }
      auto k_r = k_j + ik * kFragRowsD;
      auto q_p = q_;
      for (short id = 0; id < D; id += 32) {
        load_q(Q_tile, q_p, D);
        load_kv(K_tile, k_r, D, kr);
        MMA16x16x32(S_tile, ik, Q_tile, false, 0, K_tile, true, 0);
        q_p += 32;
        k_r += 32;
      }
    }

    
    dP_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      const short kr = k_valid(ik);
      if (kr <= 0) {
        continue;
      }
      auto v_r = v_j + ik * kFragRowsDv;
      auto co_p = co_;
      for (short id = 0; id < Dv; id += 32) {
        load_q(dO_tile, co_p, Dv);
        load_kv(V_tile, v_r, Dv, kr);
        MMA16x16x32(dP_tile, ik, dO_tile, false, 0, V_tile, true, 0);
        co_p += 32;
        v_r += 32;
      }
    }

    mlx::steel::NAXTile<float, 1, TK> dS_tile;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
      const short f = i / kEPF;
      const short w = i % kEPF;
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(w); 
      const int gj = kb + f * 16 + c.x;

      bool keep = true;
      if constexpr (KSafe) {
        keep = gj < kL;
      }
      if constexpr (Causal) {
        if (straddle) {
          keep = keep && (gj <= q_row0 + c.y + c_off);
        }
      }

      if (!keep) {
        AT_NAX(dS_tile, i) = 0.0f;
        continue;
      }

      const float p =
          metal::fast::exp2(AT_NAX(S_tile, i) * scale2 - lse_local[w >> 2]);
      AT_NAX(dS_tile, i) =
          p * scale * (AT_NAX(dP_tile, i) - odo_local[w >> 2]);
    }

    for (short id = 0; id < D; id += 32) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short kr = k_valid(ik);
        if (kr <= 0) {
          continue;
        }
        load_kv(K_tile, k_j + id + ik * kFragRowsD, D, kr);
        MMA16x32x16(dQ_tile, id / 16, dS_tile, false, ik, K_tile, false, 0);
      }
    }
  };

  for (int kb = 0; kb < kL; kb += BK) {
    if constexpr (Causal) {
      if (kb > q_last + c_off) {
        break;
      }
    }
    if (kb + BK <= kL) {
      do_kblock(metal::false_type{}, kb);
    } else {
      do_kblock(metal::true_type{}, kb);
    }

    k_j += BK * D;
    v_j += BK * Dv;
  }

  for (short id = 0; id < D / 16; id++) {
    const thread auto& fg = dQ_tile.frag_at(0, id);
    auto dq_p = dq_ + id * 16;
    for (short e = 0; e < kEPF; e++) {
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(e);
      if (c.y < q_rows) {
        dq_p[c.y * D + c.x] = InT(fg[e]);
      }
    }
  }
}

template <typename InT, int D, int Dv, int H, int Hk, int BQ, int BK, int WM, int WN, bool Causal>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
sdpa_vjp_dkv(
    const device InT* q,      // [B, H,  qL, D]
    const device InT* k,      // [B, Hk, kL, D]
    const device InT* v,      // [B, Hk, kL, Dv]
    const device float* odo,  // [B, H, qL]
    const device float* lse,  // [B, H, qL]  from the forward, natural log
    const device InT* cot_o,  // [B, H,  qL, Dv]  dO
    constant int& qL,
    constant int& kL,
    constant float& scale,
    device InT* dk,           // [B, Hk, kL, D]
    device InT* dv,           // [B, Hk, kL, Dv]
    uint3 tpg [[thread_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  constexpr int kNWarps = WM * WN;

  constexpr int TK = BK / (kNWarps * 16);
  constexpr int TQ = BQ / 16;
  constexpr int Rep = H / Hk;
  static_assert(TK == 1, "BK must equal kNWarps * 16");
  static_assert(BQ % 16 == 0, "BQ must be a multiple of 16");
  static_assert(D % 32 == 0 && Dv % 32 == 0, "head dims must pair for K=32");
  static_assert(H % Hk == 0, "H must be a multiple of Hk");

  constexpr short kEPF = mlx::steel::BaseNAXFrag::kElemsPerFrag;

  constexpr int kFragRowsD = 16 * D;
  constexpr int kFragRowsDv = 16 * Dv;

  const uint bh = tpg.z;
  const uint b_idx = bh / Hk;
  const uint kh_idx = bh % Hk;

  const int kl_idx = BK * tpg.y;
  const int tm = 16 * TK * int(simd_group_id);
  const int k_first = kl_idx + tm;

  const short k_rows = short(metal::min(16, kL - k_first));
  if (k_rows <= 0) {
    return;
  }
  const bool k_partial = (k_rows < 16);

  auto k_j = k + ((b_idx * Hk + kh_idx) * kL + k_first) * D;
  auto v_j = v + ((b_idx * Hk + kh_idx) * kL + k_first) * Dv;
  auto dk_ = dk + ((b_idx * Hk + kh_idx) * kL + k_first) * D;
  auto dv_ = dv + ((b_idx * Hk + kh_idx) * kL + k_first) * Dv;

  const float scale2 = scale * NAX_LOG2E;

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  constexpr short kRowsPT = TQ * 2;

  mlx::steel::NAXTile<float, 1, TQ> S_tile;
  mlx::steel::NAXTile<float, 1, TQ> dP_tile;
  mlx::steel::NAXTile<float, 1, 2> Q_tile, K_tile, dO_tile;
  mlx::steel::NAXTile<float, 1, D / 16> dK_tile;
  mlx::steel::NAXTile<float, 1, Dv / 16> dV_tile;
  dK_tile.clear();
  dV_tile.clear();

  int qb0 = 0;
  int k_last = 0;
  int c_off = 0;
  if constexpr (Causal) {
    c_off = kL - qL;
    k_last = k_first + 16 * TK - 1;
    const int r = k_first - c_off;
    qb0 = (r > 0) ? ((r / BQ) * BQ) : 0;
  }

  auto load_kv = [&](thread auto& tile, auto src, int ld) {
    tile.load_rows(src, ld, k_rows);
  };

  const device InT* q_i = nullptr;
  const device InT* co_i = nullptr;
  const device float* odo_i = nullptr;
  const device float* lse_i = nullptr;

  auto do_qblock = [&](auto SafeTag, int qb) {
    constexpr bool QSafe = decltype(SafeTag)::value;

    auto q_valid = [&](short iq) -> short {
      if constexpr (QSafe) {
        return short(metal::clamp(qL - (qb + iq * 16), 0, 16));
      } else {
        return 16;
      }
    };

    auto load_q = [&](thread auto& tile, auto src, int ld, short rows) {
      if constexpr (QSafe) {
        tile.load_rows(src, ld, rows);
      } else {
        tile.load(src, ld);
      }
    };

    bool straddle = false;
    if constexpr (Causal) {
      straddle = (k_last > qb + c_off);
    }

    float lse_local[kRowsPT];
    float odo_local[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      const short rows = q_valid(iq);
      for (short g = 0; g < 2; g++) {
        const short r = g * mlx::steel::BaseNAXFrag::kElemRowsJump + fm;
        const bool ok = (r < rows);
        lse_local[iq * 2 + g] =
            ok ? lse_i[iq * 16 + r] * NAX_LOG2E : INFINITY;
        odo_local[iq * 2 + g] = ok ? odo_i[iq * 16 + r] : 0.0f;
      }
    }

    S_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < D; id += 32) {
      load_kv(K_tile, k_j + id, D);
      auto q_p = q_i + id;
      for (short iq = 0; iq < TQ; iq++) {
        const short rows = q_valid(iq);
        if (rows > 0) {
          load_q(Q_tile, q_p, D, rows);
          MMA16x16x32(S_tile, iq, Q_tile, false, 0, K_tile, true, 0);
        }
        q_p += kFragRowsD;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
      const short f = i / kEPF;
      const short w = i % kEPF;
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(w); 

      bool keep = true;
      if constexpr (QSafe) {
        keep = (qb + f * 16 + c.y) < qL;
      }
      if (k_partial && (c.x >= k_rows)) {
        keep = false;
      }
      if constexpr (Causal) {
        if (straddle) {
          keep = keep && ((k_first + c.x) <= (qb + f * 16 + c.y + c_off));
        }
      }

      AT_NAX(S_tile, i) = keep
          ? metal::fast::exp2(
                AT_NAX(S_tile, i) * scale2 - lse_local[f * 2 + (w >> 2)])
          : 0.0f;
    }

    dP_tile.clear();
    for (short id = 0; id < Dv; id += 32) {
      mlx::steel::NAXTile<float, 1, 2> V_tile;
      load_kv(V_tile, v_j + id, Dv);
      auto co_p = co_i + id;
      for (short iq = 0; iq < TQ; iq++) {
        const short rows = q_valid(iq);
        if (rows > 0) {
          load_q(dO_tile, co_p, Dv, rows);
          MMA16x32x16(dV_tile, id / 16, S_tile, true, iq, dO_tile, false, 0);
          MMA16x16x32(dP_tile, iq, dO_tile, false, 0, V_tile, true, 0);
        }
        co_p += kFragRowsDv;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(S_tile)::kElemsPerTile; i++) {
      const short f = i / kEPF;
      const short w = i % kEPF;
      AT_NAX(S_tile, i) *=
          scale * (AT_NAX(dP_tile, i) - odo_local[f * 2 + (w >> 2)]);
    }

    for (short id = 0; id < D; id += 32) {
      auto q_p = q_i + id;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        const short rows = q_valid(iq);
        if (rows > 0) {
          load_q(Q_tile, q_p, D, rows);
          MMA16x32x16(dK_tile, id / 16, S_tile, true, iq, Q_tile, false, 0);
        }
        q_p += kFragRowsD;
      }
    }
  };

  for (int hv = 0; hv < Rep; hv++) {
    const uint qh_idx = kh_idx * Rep + uint(hv);

    q_i = q + ((b_idx * H + qh_idx) * qL + qb0) * D;
    co_i = cot_o + ((b_idx * H + qh_idx) * qL + qb0) * Dv;
    odo_i = odo + (b_idx * H + qh_idx) * qL + qb0;
    lse_i = lse + (b_idx * H + qh_idx) * qL + qb0;

    for (int qb = qb0; qb < qL; qb += BQ) {
      if (qb + BQ <= qL) {
        do_qblock(metal::false_type{}, qb);
      } else {
        do_qblock(metal::true_type{}, qb);
      }

      q_i += BQ * D;
      co_i += BQ * Dv;
      odo_i += BQ;
      lse_i += BQ;
    }
  }

  for (short id = 0; id < D / 16; id++) {
    const thread auto& fg = dK_tile.frag_at(0, id);
    auto dk_p = dk_ + id * 16;
    for (short e = 0; e < kEPF; e++) {
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(e);
      if (c.y < k_rows) {
        dk_p[c.y * D + c.x] = InT(fg[e]);
      }
    }
  }

  for (short id = 0; id < Dv / 16; id++) {
    const thread auto& fg = dV_tile.frag_at(0, id);
    auto dv_p = dv_ + id * 16;
    for (short e = 0; e < kEPF; e++) {
      const short2 c = mlx::steel::BaseNAXFrag::get_coord(e);
      if (c.y < k_rows) {
        dv_p[c.y * Dv + c.x] = InT(fg[e]);
      }
    }
  }
}



#define instantiate_odo(in_type, dv) \
  instantiate_kernel(                \
      "sdpa_vjp_odo_" #in_type "_" #dv, sdpa_vjp_odo, in_type, dv)

// #define instantiate_attn(in_type, d, dv, h, hk, bq, bk, wm, wn)                \
//   instantiate_kernel(                                                          \
//       "sdpa_vjp_nax_dq_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" #bk    \
//       "_" #wm,                                                         \
//       sdpa_vjp_dq,                                                             \
//       in_type, d, dv, h, hk, bq, bk, wm, wn, false)                            \
//   instantiate_kernel(                                                          \
//       "sdpa_vjp_nax_dq_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" #bk    \
//       "_" #wm "_causal",                                                       \
//       sdpa_vjp_dq,                                                             \
//       in_type, d, dv, h, hk, bq, bk, wm, wn, true)                             \
//   instantiate_kernel(                                                        \
//       "sdpa_vjp_nax_dkv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
//       #bk "_" #wm,                                                           \
//       sdpa_vjp_dkv,                                                          \
//       in_type, d, dv, h, hk, bq, bk, wm, wn,false)                           \
//       instantiate_kernel(                                                        \
//       "sdpa_vjp_nax_dkv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
//       #bk "_" #wm "_causal",                                                           \
//       sdpa_vjp_dkv,                                                          \
//       in_type, d, dv, h, hk, bq, bk, wm, wn, true)

#define instantiate_attn(in_type, d, dv, h, hk, bq, bk, wm, wn)             \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dk_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm,                                                          \
      sdpa_vjp_dk,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, false)                         \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dk_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm "_causal",                                                \
      sdpa_vjp_dk,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, true)                          \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm,                                                          \
      sdpa_vjp_dv,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, false)                         \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm "_causal",                                                \
      sdpa_vjp_dv,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, true)                          \
instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dkv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm,                                                          \
      sdpa_vjp_dkv,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, false)                         \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dkv_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm "_causal",                                                \
      sdpa_vjp_dkv,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, true)                          \

// dq only. dkv cannot take BK = 16 at 4 warps, since TK would be 0.
#define instantiate_attn_dq(in_type, d, dv, h, hk, bq, bk, wm, wn)          \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dq_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm,                                                          \
      sdpa_vjp_dq,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, false)                         \
  instantiate_kernel(                                                       \
      "sdpa_vjp_nax_dq_" #in_type "_" #d "_" #dv "_" #h "_" #hk "_" #bq "_" \
      #bk "_" #wm "_causal",                                                \
      sdpa_vjp_dq,                                                          \
      in_type, d, dv, h, hk, bq, bk, wm, wn, true)

#define instantiate_attn_shapes(in_type)                          \
    instantiate_odo(in_type, 64);                                 \
    instantiate_odo(in_type, 128);                                \
    instantiate_odo(in_type, 256);                                \
    instantiate_attn(in_type, 256, 256, 16, 16, 64, 64, 4, 1);    \
    instantiate_attn(in_type, 256, 256, 16,  4, 64, 64, 4, 1);    \
    instantiate_attn(in_type, 128, 128, 16, 16, 64, 64, 4, 1);    \
    instantiate_attn(in_type, 128, 128, 16,  4, 64, 64, 4, 1);    \
    instantiate_attn(in_type,  64,  64, 16, 16, 64, 64, 4, 1);    \
    instantiate_attn(in_type,  64,  64,  8,  8, 64, 64, 4, 1);    \
    instantiate_attn(in_type,  64,  64, 32,  8, 64, 64, 4, 1);    \
    instantiate_attn_dq(in_type,  64,  64, 32,  8, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type, 256, 256, 16, 16, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type, 256, 256, 16,  4, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type, 128, 128, 16, 16, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type, 128, 128, 16,  4, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type,  64,  64, 16, 16, 64, 16, 4, 1); \
    instantiate_attn_dq(in_type,  64,  64,  8,  8, 64, 16, 4, 1);

instantiate_attn_shapes(bfloat16_t);
instantiate_attn_shapes(float16_t);
instantiate_attn_shapes(float);
