// Copyright © 2024-25 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// STEEL VJP dQ Kernel for Scaled Dot-Product Attention
//
// Part of the two-kernel backward pass optimization that eliminates atomic
// operations. This kernel computes ONLY dQ gradients.
//
// Grid: [NQ, H, B] - one threadgroup per (query_block, head, batch)
// Loop: Over KV blocks to accumulate dQ
//
// Algorithm (MFA-aligned, log2 domain):
//   S = Q @ K^T (unscaled)
//   S *= scale_log2           (post-scale in float32)
//   P = exp2(S - LSE)
//   dP = dO @ V^T
//   dS = scale * P * (dP - delta)   (scale baked into dS)
//   dQ += dS @ K              (no write-back scale needed)
//
// Architecture (MFA-aligned):
//   - Aliased KV_smem: single buffer for K^T, V^T, K row-major (3 phases)
//   - Q/dO hoisted to register tiles after initial smem load
//   - 3-phase per KV iteration: (1) K_t→S, (2) V_t→dP, (3) K_r→dQ
//   - LSE/delta read from device memory
//   - Per-fragment MMA for S and dP
//
// See companion kernel steel_attention_vjp_dkv.h for dK/dV computation.
///////////////////////////////////////////////////////////////////////////////

// JIT-baked constants: when compiled via JIT, these are #defined as literals
// before this header is included. For metallib builds, they fall back to
// params-> reads at runtime (values) or function constants (booleans).
#ifndef VJP_GQA_FACTOR
  #define VJP_GQA_FACTOR (params->gqa_factor)
  #define VJP_SCALE (params->scale)
  #define VJP_SCALE_LOG2 (params->scale_log2)
  #define VJP_UNDEF_DEFINES
#endif

using namespace mlx::steel;

// When JIT-compiled, align/causal flags are baked as constexpr booleans,
// enabling full dead-code elimination. Metallib builds use function constants.
#ifdef VJP_BAKED_FC
constexpr constant bool align_Q_vjp_dq = VJP_ALIGN_Q;
constexpr constant bool align_K_vjp_dq = VJP_ALIGN_K;
constexpr constant bool do_causal_vjp_dq = VJP_DO_CAUSAL;
constexpr constant bool has_block_mask_vjp_dq = VJP_HAS_BLOCK_MASK;
#else
constant bool align_Q_vjp_dq [[function_constant(200)]];
constant bool align_K_vjp_dq [[function_constant(201)]];
constant bool do_causal_vjp_dq [[function_constant(301)]];
constant bool has_block_mask_vjp_dq [[function_constant(302)]];
#endif

///////////////////////////////////////////////////////////////////////////////
// STEEL Attention VJP dQ Kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    int BQ,           // Query block size (32)
    int BK,           // KV block size (16)
    int BD,           // Head dimension (64, 96, 128)
    int WM,           // Warps in M dimension (4)
    int WN,           // Warps in N dimension (1)
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void attention_vjp_dq(
    // Forward inputs
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* delta [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    const device float* LSE [[buffer(5)]],
    // Gradient output (dQ only)
    device T* dQ [[buffer(6)]],
    // Parameters
    const constant AttnVJPParams* params [[buffer(7)]],
    // Sparse block mask (optional, gated by has_block_mask function constant)
    const device uint8_t* block_mask [[buffer(8)]],
    // Thread info
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // clang-format on

  (void)lid;

  ulong3 tidl{tid.x, tid.y, tid.z};

  // Input pointer setup
  const device T* Q_block = Q +
      tidl.z * params->Q_strides[0] +
      tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / VJP_GQA_FACTOR;
  const device T* K_base = K +
      tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1];

  const device T* V_base = V +
      tidl.z * params->V_strides[0] +
      kv_head_idx * params->V_strides[1];

  const device T* dO_block = dO +
      tidl.z * params->dO_strides[0] +
      tidl.y * params->dO_strides[1] +
      tidl.x * BQ * params->dO_strides[2];

  device T* dQ_block = dQ +
      tidl.z * params->dQ_strides[0] +
      tidl.y * params->dQ_strides[1] +
      tidl.x * BQ * params->dQ_strides[2];

  // =========================================================================
  // Threadgroup memory setup (MFA-aligned)
  // KV_smem aliased for K^T, V^T, K row-major (3 separate load phases)
  // =========================================================================
  constexpr short pad = 16 / sizeof(T);
  constexpr short LDQ = BD + pad;       // Q/dO row stride in smem
  constexpr short LDKt = BK + pad;      // K^T/V^T row stride (transposed)
  constexpr short LDKr = BD + pad;      // K row-major row stride

  // KV_smem aliased for K^T, V^T, K row-major
  constexpr int kv_s0 = BD * LDKt;      // K^T or V^T size
  constexpr int kv_s1 = BK * LDKr;      // K row-major size
  constexpr int kv_s = kv_s0 > kv_s1 ? kv_s0 : kv_s1;

  threadgroup T Q_smem[BQ * LDQ];       // staging only
  threadgroup T dO_smem[BQ * LDQ];      // staging only
  threadgroup T KV_smem[kv_s];          // aliased buffer

  // Block loaders
  using QBlockLoader  = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, WM * WN * 32>;
  using KtBlockLoader = BlockLoaderT<T, BK, BD, 1, LDKt, 0, WM * WN * 32>;
  using KrBlockLoader = BlockLoaderT<T, BK, BD, LDKr, 1, 0, WM * WN * 32>;

  QBlockLoader loader_q(Q_block, params->Q_strides[2], Q_smem, simd_group_id, simd_lane_id);
  QBlockLoader loader_do(dO_block, params->dO_strides[2], dO_smem, simd_group_id, simd_lane_id);
  KtBlockLoader loader_kt(K_base, params->K_strides[2], KV_smem, simd_group_id, simd_lane_id);
  KtBlockLoader loader_vt(V_base, params->V_strides[2], KV_smem, simd_group_id, simd_lane_id);
  KrBlockLoader loader_kr(K_base, params->K_strides[2], KV_smem, simd_group_id, simd_lane_id);

  // MMA setup
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "TQ must be 1");

  // Register tiles
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Qtile;     // Q hoisted from smem
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> dOtile;    // dO hoisted from smem
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> dQtile;    // accumulator
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;     // S / P / dS
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> dPtile;    // dP
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;      // K fragment per dd
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> KRtile;      // K row-major fragment

  // Coordinates
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  // Smem offsets
  const short Qs_off = (tm + sm) * LDQ + sn;       // Q/dO read offset
  const short Kts_off = sm * LDKt + sn;             // K^T/V^T read offset
  const short KRs_off = sm * LDKr + sn;             // K row-major read offset

  // =========================================================================
  // Pre-loop: Load Q/dO → hoist to registers → read LSE/delta
  // =========================================================================
  if (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned) {
    loader_q.load_safe(short2(BD, params->qL_rem));
    loader_do.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
    loader_do.load_unsafe();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Hoist Q and dO to register tiles
  Qtile.template load<T, 1, 1, LDQ, 1>(&Q_smem[Qs_off]);
  dOtile.template load<T, 1, 1, LDQ, 1>(&dO_smem[Qs_off]);

  dQtile.clear();

  // Read LSE and delta from device memory (scalar per row)
  const long lse_base = (long)(tidl.z * params->H + tidl.y) * params->LSE_strides[0];
  const long delta_base = (long)(tidl.z * params->H + tidl.y) * params->delta_strides[0];
  const long q_row_idx = (long)tid.x * BQ + tm + sm;
  const AccumType L_val = (q_row_idx < params->qL)
      ? LSE[lse_base + q_row_idx * params->LSE_strides[1]] : AccumType(0);
  const AccumType delta_val = (q_row_idx < params->qL)
      ? delta[delta_base + q_row_idx * params->delta_strides[1]] : AccumType(0);

  // KV loop bounds
  int kb_lim = params->NK;
  if (do_causal_vjp_dq) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  // =========================================================================
  // Main loop over KV blocks (3-phase per iteration)
  // =========================================================================
  const int qb = tid.x; // Q-block index for block_mask lookup

  for (int kb = 0; kb < kb_lim; kb++) {

    // Block-sparse: skip K-tiles where block_mask[qb][kb] == 0.
    // All threads in a threadgroup share tid.x and kb, so this is a
    // uniform branch — no warp divergence, just skips the barriers and math.
    if (has_block_mask_vjp_dq && !block_mask[qb * params->NK_tiles + kb]) {
      loader_kt.next();
      loader_vt.next();
      loader_kr.next();
      continue;
    }

    // =====================================================================
    // Phase 1: K^T → S = Q @ K^T
    // =====================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_kt.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_kt.load_unsafe();
    }

    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      Ktile.template load<T, 1, 1, LDKt, 1>(&KV_smem[Kts_off + dd * kFragSize * LDKt]);

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_acc_t::mma(
            Stile.frag_at(0, ik),
            Qtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    // Post-scale S to log2 domain: S *= scale_log2 (in float32)
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      Stile.frag_at(0, j)[0] *= VJP_SCALE_LOG2;
      Stile.frag_at(0, j)[1] *= VJP_SCALE_LOG2;
    }

    // Apply sequence length mask
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      using stile_t = decltype(Stile);
      constexpr AccumType neg_inf = -INFINITY;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Apply causal mask
    if (do_causal_vjp_dq && kb >= (kb_lim - ((BQ + BK - 1) / BK) - int(!align_K_vjp_dq))) {
      using stile_t = decltype(Stile);
      constexpr AccumType neg_inf = -INFINITY;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // =====================================================================
    // Phase 2: V^T → dP = dO @ V^T
    // =====================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_vt.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_vt.load_unsafe();
    }

    dPtile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      Ktile.template load<T, 1, 1, LDKt, 1>(&KV_smem[Kts_off + dd * kFragSize * LDKt]);

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_acc_t::mma(
            dPtile.frag_at(0, ik),
            dOtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            dPtile.frag_at(0, ik));
      }
    }

    // =====================================================================
    // Softmax + dS (no barrier needed — purely register operations)
    // =====================================================================

    // P = exp2(S - LSE)
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      Stile.frag_at(0, j)[0] = fast::exp2(Stile.frag_at(0, j)[0] - L_val);
      Stile.frag_at(0, j)[1] = fast::exp2(Stile.frag_at(0, j)[1] - L_val);
    }

    // dS = scale * P * (dP - delta)
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      Stile.frag_at(0, j)[0] = VJP_SCALE * Stile.frag_at(0, j)[0] * (dPtile.frag_at(0, j)[0] - delta_val);
      Stile.frag_at(0, j)[1] = VJP_SCALE * Stile.frag_at(0, j)[1] * (dPtile.frag_at(0, j)[1] - delta_val);
    }
    // Stile now holds dS

    // =====================================================================
    // Phase 3: K row-major → dQ += dS @ K
    // =====================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_kr.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_kr.load_unsafe();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          KRtile.template load<T, 1, 1, LDKr, 1>(
              &KV_smem[KRs_off + ik * kFragSize * LDKr + id * kFragSize]);
          MMAFrag_acc_t::mma(
              dQtile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              KRtile.frag_at(0, 0),
              dQtile.frag_at(iq, id));
        }
      }
    }

    loader_kt.next();
    loader_vt.next();
    loader_kr.next();
  }

  // =========================================================================
  // Write dQ output — no scale needed (scale already baked into dS)
  // =========================================================================
  threadgroup_barrier(mem_flags::mem_none);

  dQ_block += (tm + sm) * params->dQ_strides[2] + sn;

  if (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    dQtile.template store_safe<T, 1, 1>(dQ_block, params->dQ_strides[2], dst_tile_dims);
  } else {
    dQtile.template store<T, 1, 1>(dQ_block, params->dQ_strides[2]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Template instantiation macros
///////////////////////////////////////////////////////////////////////////////

// tname is the string name used in kernel lookup (e.g., "float32", "float16")
// dtype is the actual C++ type (e.g., float, half, bfloat16_t)
#define instantiate_attention_vjp_dq_kernel(tname, dtype, bq, bk, bd, wm, wn) \
  template [[host_name("attention_vjp_dq_" #tname "_" #bq "_" #bk "_" #bd)]]  \
  [[kernel]] void attention_vjp_dq<dtype, bq, bk, bd, wm, wn>(                \
      const device dtype*,                                                      \
      const device dtype*,                                                      \
      const device dtype*,                                                      \
      const device float*,                                                      \
      const device dtype*,                                                      \
      const device float*,                                                      \
      device dtype*,                                                            \
      const constant AttnVJPParams*,                                            \
      const device uint8_t*,                                                    \
      uint,                                                                     \
      uint,                                                                     \
      uint3,                                                                    \
      uint3);

// Common configurations (2 bytes per half in threadgroup memory):
// D=64: BK=32, Q+K+V+dO = ~19KB
// D=96: BK=16, Q+K+V+dO = ~22KB
// D=128: BK=16, Q+KV(aliased)+dO = ~24KB
#define instantiate_attention_vjp_dq_bd64(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 32, 64, 4, 1)

#define instantiate_attention_vjp_dq_bd96(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 16, 96, 4, 1)

// BK=32 for D=96 on M3+ (halves KV iterations, fits in ~22KB smem)
#define instantiate_attention_vjp_dq_bd96_bk32(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 32, 96, 4, 1)

#define instantiate_attention_vjp_dq_bd128(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 16, 128, 4, 1)

// BK=32 for D=128 on M3+ (halves KV iterations, fits in ~27KB smem)
#define instantiate_attention_vjp_dq_bd128_bk32(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 32, 128, 4, 1)

#define instantiate_attention_vjp_dq_all(tname, dtype) \
  instantiate_attention_vjp_dq_bd64(tname, dtype)      \
  instantiate_attention_vjp_dq_bd96(tname, dtype)      \
  instantiate_attention_vjp_dq_bd96_bk32(tname, dtype) \
  instantiate_attention_vjp_dq_bd128(tname, dtype)     \
  instantiate_attention_vjp_dq_bd128_bk32(tname, dtype)

#ifdef VJP_UNDEF_DEFINES
  #undef VJP_GQA_FACTOR
  #undef VJP_SCALE
  #undef VJP_SCALE_LOG2
  #undef VJP_UNDEF_DEFINES
#endif
