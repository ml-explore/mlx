// Copyright © 2024-25 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// STEEL VJP dKV Kernel for Scaled Dot-Product Attention
//
// Supports WM=1 (32 threads, single simdgroup) and WM=2 (64 threads, two
// simdgroups). WM=2 halves per-thread register pressure (~216 vs ~364 regs
// for D=128) at the cost of a threadgroup reduction at the end.
// dO is loaded on-the-fly from smem (not hoisted to registers) to further
// reduce live register count by TQ*TD*2 floats (~64 regs for D=128 WM=2).
//
// Grid: [NK, num_kv_heads, B] - one threadgroup per (kv_block, kv_head, batch)
// Loop: Over GQA query heads, then over Q blocks to accumulate dK/dV
//
// Algorithm (log2 domain):
//   S = Q @ K^T (unscaled)
//   S *= scale_log2           (post-scale in float32)
//   P = exp2(S - LSE)
//   dV += P^T @ dO            (via scatter-to-smem transpose)
//   dP = dO @ V^T
//   dS = scale * P * (dP - delta)   (scale baked into dS)
//   dK += dS^T @ Q            (via scatter-to-smem transpose)
//
// See companion kernel steel_attention_vjp_dq.h for dQ computation.
///////////////////////////////////////////////////////////////////////////////

// JIT-baked constants: when compiled via JIT, these are #defined as literals
// before this header is included. For metallib builds, they fall back to
// params-> reads at runtime.
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
constexpr constant bool align_Q_vjp_dkv = VJP_ALIGN_Q;
constexpr constant bool align_K_vjp_dkv = VJP_ALIGN_K;
constexpr constant bool do_causal_vjp_dkv = VJP_DO_CAUSAL;
constexpr constant bool has_block_mask_vjp_dkv = VJP_HAS_BLOCK_MASK;
#else
constant bool align_Q_vjp_dkv [[function_constant(200)]];
constant bool align_K_vjp_dkv [[function_constant(201)]];
constant bool do_causal_vjp_dkv [[function_constant(301)]];
constant bool has_block_mask_vjp_dkv [[function_constant(302)]];
#endif

///////////////////////////////////////////////////////////////////////////////
// STEEL Attention VJP dKV Kernel
///////////////////////////////////////////////////////////////////////////////

// clang-format off
template <
    typename T,
    int BQ,           // Query block size (32)
    int BK,           // KV block size (16 or 32)
    int BD,           // Head dimension (64, 96, 128)
    int WM,           // Warps in M dimension (1 or 2)
    int WN,           // Warps in N dimension (1)
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void attention_vjp_dkv(
    // Forward inputs
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device float* delta [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    const device float* LSE [[buffer(5)]],
    // Gradient outputs (dK and dV)
    device T* dK [[buffer(6)]],
    device T* dV [[buffer(7)]],
    // Parameters
    const constant AttnVJPParams* params [[buffer(8)]],
    // Sparse block mask (optional, gated by has_block_mask function constant)
    const device uint8_t* block_mask [[buffer(9)]],
    // Thread info
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // clang-format on

  (void)lid;

  // =========================================================================
  // Constants
  // =========================================================================
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TGP_SIZE = kNWarps * 32;
  constexpr int TQ = BQ / (kNWarps * kFragSize); // 4 for WM=1, 2 for WM=2
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  // =========================================================================
  // Simd coordinates
  // WM=1: tm=0 always. WM=2: tm=0 (sg0) or TQ*8 (sg1).
  // =========================================================================
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  // =========================================================================
  // Thread/block IDs
  // =========================================================================
  int kb = tid.x;
  ulong3 tidl{tid.x, tid.y, tid.z};
  ulong kv_head_idx = int(tid.y);

  // =========================================================================
  // Shared memory layout
  // =========================================================================
  constexpr short pad = 16 / sizeof(T);
  constexpr short LDQ = BD + pad; // Q/dO row stride
  constexpr short LDKt = BK + pad; // K^T/V^T row stride (transposed)
  constexpr short LDT = BQ + pad; // P^T/dS^T row stride (scatter)

  // KV_smem: aliased for K^T, V^T, P^T scatter, dS^T scatter
  constexpr int kv_s0 = BD * LDKt; // K^T/V^T (BD rows x LDKt cols)
  constexpr int kv_s1 = BK * LDT; // P^T/dS^T (BK rows x LDT cols)
  constexpr int kv_s = kv_s0 > kv_s1 ? kv_s0 : kv_s1;

  // Q_smem and dO_smem are used only during the iteration phase (Q-block loop).
  // red_smem is used only during the post-loop reduction phase (WM>1).
  // Since they are temporally disjoint, alias red_smem over the Q+dO region
  // to reduce threadgroup memory (e.g. D=128: 23 KB → 15 KB, enabling 2 TGs/core).
  constexpr int kQdO_elems = 2 * BQ * LDQ;

  constexpr int kRedTK = (TK <= 2) ? TK : TK / 2;
  constexpr int kRedRows = kRedTK * kFragSize;
  constexpr int kRedSize = (kNWarps > 1) ? kRedRows * BD : 1;

  static_assert(
      kNWarps == 1 || kQdO_elems * sizeof(T) >= kRedSize * sizeof(AccumType),
      "QdO smem region too small to alias with red_smem");

  threadgroup T QdO_smem[kQdO_elems];
  threadgroup T* Q_smem = QdO_smem;
  threadgroup T* dO_smem = QdO_smem + BQ * LDQ;

  threadgroup T KV_smem[kv_s];

  // red_smem aliases over QdO_smem (safe: temporally disjoint with Q/dO usage).
  // For WM=1, red_smem is never accessed (compiler eliminates the reduction block).
  threadgroup AccumType* red_smem = (threadgroup AccumType*)QdO_smem;

  // Smem offsets for fragment reads (each simdgroup reads its own Q rows)
  const short Qs_off = (tm + sm) * LDQ + sn;
  const short Kts_off = sm * LDKt + sn;

  // =========================================================================
  // K, V, dK, dV pointers (fixed for all Q iterations)
  // =========================================================================
  const device T* K_block = K + tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1] + kb * BK * params->K_strides[2];

  const device T* V_block = V + tidl.z * params->V_strides[0] +
      kv_head_idx * params->V_strides[1] + kb * BK * params->V_strides[2];

  device T* dK_block = dK + tidl.z * params->dK_strides[0] +
      kv_head_idx * params->dK_strides[1] + kb * BK * params->dK_strides[2];

  device T* dV_block = dV + tidl.z * params->dV_strides[0] +
      kv_head_idx * params->dV_strides[1] + kb * BK * params->dV_strides[2];

  // =========================================================================
  // Block loader types
  // =========================================================================
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, 1, TGP_SIZE>;
  using KtBlockLoader = BlockLoaderT<T, BK, BD, 1, LDKt, 0, TGP_SIZE>;

  // =========================================================================
  // dK, dV accumulators — full [TK, TD] tiles (no D-column distribution)
  // =========================================================================
  MMATile<AccumType, TK, TD, MMAFrag_acc_t> dKtile;
  MMATile<AccumType, TK, TD, MMAFrag_acc_t> dVtile;
  dKtile.clear();
  dVtile.clear();

  // =========================================================================
  // Q block loop bounds (causal: skip Q-tiles fully below this K-tile)
  // =========================================================================
  int qb_start = 0;
  if (do_causal_vjp_dkv) {
    int k_start = kb * BK;
    qb_start = max(0, (k_start - params->qL_off) / BQ);
  }

  // =========================================================================
  // Main loop: iterate over GQA heads, then Q blocks
  // =========================================================================
  const ulong q_head_start = kv_head_idx * VJP_GQA_FACTOR;
  STEEL_PRAGMA_UNROLL
  for (int gqa_idx = 0; gqa_idx < VJP_GQA_FACTOR; gqa_idx++) {
    ulong q_head_idx = q_head_start + gqa_idx;
    for (int qb = qb_start; qb < params->NQ; qb++) {

      // Block-sparse: skip Q-tiles where block_mask[qb][kb] == 0.
      // All threads in a threadgroup share kb and qb, so this is a
      // uniform branch — no warp divergence, just skips the iteration.
      if (has_block_mask_vjp_dkv && !block_mask[qb * params->NK_tiles + kb]) {
        continue;
      }

      // Per-head pointers
      const device T* Q_ptr = Q + tidl.z * params->Q_strides[0] +
          q_head_idx * params->Q_strides[1] +
          qb * BQ * params->Q_strides[2];

      const device T* dO_ptr = dO + tidl.z * params->dO_strides[0] +
          q_head_idx * params->dO_strides[1] +
          qb * BQ * params->dO_strides[2];

      // =======================================================================
      // Load Q, dO, K^T into shared memory
      // =======================================================================
      QBlockLoader loader_q(
          Q_ptr, params->Q_strides[2], Q_smem, simd_group_id, simd_lane_id);
      QBlockLoader loader_do(
          dO_ptr, params->dO_strides[2], dO_smem, simd_group_id, simd_lane_id);
      KtBlockLoader loader_kt(
          K_block, params->K_strides[2], KV_smem, simd_group_id, simd_lane_id);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (!align_Q_vjp_dkv && qb == params->NQ_aligned) {
        loader_q.load_safe(short2(BD, params->qL_rem));
        loader_do.load_safe(short2(BD, params->qL_rem));
      } else {
        loader_q.load_unsafe();
        loader_do.load_unsafe();
      }

      if (!align_K_vjp_dkv && kb == params->NK_aligned) {
        loader_kt.load_safe(short2(BD, params->kL_rem));
      } else {
        loader_kt.load_unsafe();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // =======================================================================
      // Hoist Q into register tile (each simdgroup loads its own rows).
      // dO is loaded on-the-fly from smem to reduce register pressure.
      // For D=128 WM=2, this saves ~64 registers/thread (TQ*TD*2 floats).
      // =======================================================================
      MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Qreg;
      Qreg.template load<T, 1, 1, LDQ, 1>(&Q_smem[Qs_off]);

      // =======================================================================
      // S = Q @ K^T (unscaled)
      // =======================================================================
      MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
      Stile.clear();

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        Ktile.template load<T, 1, 1, LDKt, 1>(
            &KV_smem[Kts_off + dd * kFragSize * LDKt]);

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            MMAFrag_acc_t::mma(
                Stile.frag_at(iq, ik),
                Qreg.frag_at(iq, dd),
                Ktile.frag_at(0, ik),
                Stile.frag_at(iq, ik));
          }
        }
      }

      // =======================================================================
      // Post-scale S *= scale_log2 (in float32)
      // =======================================================================
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < TQ * TK * 2; ii++) {
        Stile.elems()[ii] *= VJP_SCALE_LOG2;
      }

      // =======================================================================
      // K boundary mask (last K block)
      // =======================================================================
      if (!align_K_vjp_dkv && kb == params->NK_aligned) {
        constexpr AccumType neg_inf = -INFINITY;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            short col = sn + j * kFragSize;
            if (col >= params->kL_rem)
              Stile.frag_at(iq, j)[0] = neg_inf;
            if ((col + 1) >= params->kL_rem)
              Stile.frag_at(iq, j)[1] = neg_inf;
          }
        }
      }

      // =======================================================================
      // Q boundary mask (last Q block — ensures exp2 gives exact zeros)
      // =======================================================================
      if (!align_Q_vjp_dkv && qb == params->NQ_aligned) {
        constexpr AccumType neg_inf = -INFINITY;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          if ((tm + iq * kFragSize + sm) >= params->qL_rem) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < TK; j++) {
              Stile.frag_at(iq, j)[0] = neg_inf;
              Stile.frag_at(iq, j)[1] = neg_inf;
            }
          }
        }
      }

      // =======================================================================
      // Causal mask
      // =======================================================================
      if (do_causal_vjp_dkv) {
        constexpr AccumType neg_inf = -INFINITY;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          int q_row = qb * BQ + params->qL_off + tm + iq * kFragSize + sm;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TK; j++) {
            int k_col = kb * BK + sn + j * kFragSize;
            if (q_row < k_col)
              Stile.frag_at(iq, j)[0] = neg_inf;
            if (q_row < (k_col + 1))
              Stile.frag_at(iq, j)[1] = neg_inf;
          }
        }
      }

      // =======================================================================
      // Read LSE and delta from device memory (no shared memory needed)
      // =======================================================================
      const long lse_base =
          (long)(tidl.z * params->H + q_head_idx) * params->LSE_strides[0];
      const long delta_base =
          (long)(tidl.z * params->H + q_head_idx) * params->delta_strides[0];
      AccumType L_vals[TQ];
      AccumType delta_vals[TQ];

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        long q_row_idx = (long)qb * BQ + tm + iq * kFragSize + sm;
        L_vals[iq] = (q_row_idx < params->qL)
            ? LSE[lse_base + q_row_idx * params->LSE_strides[1]]
            : AccumType(0);
        delta_vals[iq] = (q_row_idx < params->qL)
            ? delta[delta_base + q_row_idx * params->delta_strides[1]]
            : AccumType(0);
      }

      // =======================================================================
      // P = exp2(S - LSE)
      // =======================================================================
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          Stile.frag_at(iq, j)[0] =
              fast::exp2(Stile.frag_at(iq, j)[0] - L_vals[iq]);
          Stile.frag_at(iq, j)[1] =
              fast::exp2(Stile.frag_at(iq, j)[1] - L_vals[iq]);
        }
      }
      // Stile now holds P

      // =======================================================================
      // dV += P^T @ dO
      // Step 1: Scatter P^T -> KV_smem[BK x BQ+pad] as type T
      // =======================================================================
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          KV_smem[(j * kFragSize + sn) * LDT + tm + iq * kFragSize + sm] =
              static_cast<T>(Stile.frag_at(iq, j)[0]);
          KV_smem[(j * kFragSize + sn + 1) * LDT + tm + iq * kFragSize + sm] =
              static_cast<T>(Stile.frag_at(iq, j)[1]);
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Step 2: Load Pt_tile[TK, TQ] from KV_smem
      MMATile<AccumType, TK, TQ, MMAFrag_acc_t> Pt_tile;
      Pt_tile.template load<T, 1, 1, LDT, 1>(
          &KV_smem[sm * LDT + tm + sn]);

      // Step 3: dV[TK, TD] += Pt[TK, TQ] @ dO[TQ, TD]
      // dO loaded on-the-fly from smem (saves TQ*TD*2 = 64 regs for D=128).
      // Reduction over TQ (=2 for WM=2), so only 1 dO fragment live at a time.
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          typename MMAFrag_acc_t::frag_type dO_frag;
          MMAFrag_acc_t::load(
              dO_frag,
              &dO_smem[(tm + iq * kFragSize + sm) * LDQ +
                       id * kFragSize + sn],
              Int<LDQ>{},
              Int<1>{});

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            MMAFrag_acc_t::mma(
                dVtile.frag_at(ik, id),
                Pt_tile.frag_at(ik, iq),
                dO_frag,
                dVtile.frag_at(ik, id));
          }
        }
      }

      // =======================================================================
      // dP = dO @ V^T
      // Load V^T into KV_smem (aliased, overwrites P^T scatter)
      // =======================================================================
      threadgroup_barrier(mem_flags::mem_threadgroup);

      KtBlockLoader loader_vt(
          V_block, params->V_strides[2], KV_smem, simd_group_id, simd_lane_id);
      if (!align_K_vjp_dkv && kb == params->NK_aligned) {
        loader_vt.load_safe(short2(BD, params->kL_rem));
      } else {
        loader_vt.load_unsafe();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      MMATile<AccumType, TQ, TK, MMAFrag_acc_t> dPtile;
      dPtile.clear();

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        // Reuse Ktile to load V^T row (same KV_smem layout as K^T)
        Ktile.template load<T, 1, 1, LDKt, 1>(
            &KV_smem[Kts_off + dd * kFragSize * LDKt]);

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          // Load dO fragment on-the-fly from smem
          typename MMAFrag_acc_t::frag_type dO_frag;
          MMAFrag_acc_t::load(
              dO_frag,
              &dO_smem[(tm + iq * kFragSize + sm) * LDQ +
                       dd * kFragSize + sn],
              Int<LDQ>{},
              Int<1>{});

          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            MMAFrag_acc_t::mma(
                dPtile.frag_at(iq, ik),
                dO_frag,
                Ktile.frag_at(0, ik),
                dPtile.frag_at(iq, ik));
          }
        }
      }

      // =======================================================================
      // dS = scale * P * (dP - delta)
      // Reuse Stile (which held P) — overwrite with dS
      // =======================================================================
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          Stile.frag_at(iq, j)[0] = VJP_SCALE * Stile.frag_at(iq, j)[0] *
              (dPtile.frag_at(iq, j)[0] - delta_vals[iq]);
          Stile.frag_at(iq, j)[1] = VJP_SCALE * Stile.frag_at(iq, j)[1] *
              (dPtile.frag_at(iq, j)[1] - delta_vals[iq]);
        }
      }
      // Stile now holds dS

      // =======================================================================
      // dK += dS^T @ Q
      // Step 1: Scatter dS^T -> KV_smem[BK x BQ+pad] as type T
      // =======================================================================
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TK; j++) {
          KV_smem[(j * kFragSize + sn) * LDT + tm + iq * kFragSize + sm] =
              static_cast<T>(Stile.frag_at(iq, j)[0]);
          KV_smem[(j * kFragSize + sn + 1) * LDT + tm + iq * kFragSize + sm] =
              static_cast<T>(Stile.frag_at(iq, j)[1]);
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Step 2: Load dSt_tile[TK, TQ] from KV_smem
      MMATile<AccumType, TK, TQ, MMAFrag_acc_t> dSt_tile;
      dSt_tile.template load<T, 1, 1, LDT, 1>(
          &KV_smem[sm * LDT + tm + sn]);

      // Step 3: dK[TK, TD] += dSt[TK, TQ] @ Q[TQ, TD]
      tile_matmad(dKtile, dSt_tile, Qreg, dKtile);

    } // End Q block loop
  } // End GQA loop

  // =========================================================================
  // Multi-warp reduction: sum partial dK/dV across simdgroups
  // For WM=1 this block is eliminated by the compiler (kNWarps == 1).
  // For WM=2: sg0 stores its partial to red_smem, sg1 reads and adds.
  // Two phases: dV first, then dK (reusing the same red_smem buffer).
  // NOTE: red_smem is aliased over QdO_smem. This barrier ensures all
  // threads have finished reading Q_smem/dO_smem from the last iteration
  // before red_smem writes begin overwriting that memory.
  // =========================================================================
  if constexpr (kNWarps > 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr int kChunks = TK / kRedTK;

    // Phase 1: Reduce dV (in chunks of kRedTK tile-rows)
    for (int chunk = 0; chunk < kChunks; chunk++) {
      int ik_base = chunk * kRedTK;
      if (simd_group_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < kRedTK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            short row = ik * kFragSize + sm;
            short col = id * kFragSize + sn;
            red_smem[row * BD + col] = dVtile.frag_at(ik_base + ik, id)[0];
            red_smem[row * BD + col + 1] = dVtile.frag_at(ik_base + ik, id)[1];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group_id == kNWarps - 1) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < kRedTK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            short row = ik * kFragSize + sm;
            short col = id * kFragSize + sn;
            dVtile.frag_at(ik_base + ik, id)[0] += red_smem[row * BD + col];
            dVtile.frag_at(ik_base + ik, id)[1] += red_smem[row * BD + col + 1];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: Reduce dK (same chunked pattern)
    for (int chunk = 0; chunk < kChunks; chunk++) {
      int ik_base = chunk * kRedTK;
      if (simd_group_id == 0) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < kRedTK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            short row = ik * kFragSize + sm;
            short col = id * kFragSize + sn;
            red_smem[row * BD + col] = dKtile.frag_at(ik_base + ik, id)[0];
            red_smem[row * BD + col + 1] = dKtile.frag_at(ik_base + ik, id)[1];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (simd_group_id == kNWarps - 1) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < kRedTK; ik++) {
          STEEL_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            short row = ik * kFragSize + sm;
            short col = id * kFragSize + sn;
            dKtile.frag_at(ik_base + ik, id)[0] += red_smem[row * BD + col];
            dKtile.frag_at(ik_base + ik, id)[1] += red_smem[row * BD + col + 1];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // =========================================================================
  // Write dK and dV to device memory
  // For WM>1, only the last simdgroup writes (it holds the reduced result).
  // =========================================================================
  if (kNWarps == 1 || simd_group_id == (kNWarps - 1)) {
    dV_block += sm * (long)params->dV_strides[2] + sn;
    dK_block += sm * (long)params->dK_strides[2] + sn;

    if (!align_K_vjp_dkv && kb == params->NK_aligned) {
      auto dims = short2((short)(BD - sn), (short)(params->kL_rem - sm));
      if (dims.x > 0 && dims.y > 0) {
        dVtile.template store_safe<T, 1, 1>(
            dV_block, (int)params->dV_strides[2], dims);
        dKtile.template store_safe<T, 1, 1>(
            dK_block, (int)params->dK_strides[2], dims);
      }
    } else {
      dVtile.template store<T, 1, 1>(dV_block, (int)params->dV_strides[2]);
      dKtile.template store<T, 1, 1>(dK_block, (int)params->dK_strides[2]);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Template instantiation macros
///////////////////////////////////////////////////////////////////////////////

// WM=1 kernel names (backward compatible, no _wm suffix)
#define instantiate_attention_vjp_dkv_kernel(tname, dtype, bq, bk, bd, wm, wn) \
  template [[host_name(                                                        \
      "attention_vjp_dkv_" #tname "_" #bq "_" #bk "_" #bd)]] [[kernel]] void   \
  attention_vjp_dkv<dtype, bq, bk, bd, wm, wn>(                                \
      const device dtype*,                                                      \
      const device dtype*,                                                      \
      const device dtype*,                                                      \
      const device float*,                                                      \
      const device dtype*,                                                      \
      const device float*,                                                      \
      device dtype*,                                                            \
      device dtype*,                                                            \
      const constant AttnVJPParams*,                                            \
      const device uint8_t*,                                                    \
      uint,                                                                     \
      uint,                                                                     \
      uint3,                                                                    \
      uint3);

// WM>1 kernel names (includes _wmN suffix for disambiguation)
#define instantiate_attention_vjp_dkv_kernel_wm(tname, dtype, bq, bk, bd, wm, wn) \
  template [[host_name(                                                            \
      "attention_vjp_dkv_" #tname "_" #bq "_" #bk "_" #bd "_wm" #wm)]]            \
  [[kernel]] void attention_vjp_dkv<dtype, bq, bk, bd, wm, wn>(                   \
      const device dtype*,                                                         \
      const device dtype*,                                                         \
      const device dtype*,                                                         \
      const device float*,                                                         \
      const device dtype*,                                                         \
      const device float*,                                                         \
      device dtype*,                                                               \
      device dtype*,                                                               \
      const constant AttnVJPParams*,                                               \
      const device uint8_t*,                                                       \
      uint,                                                                        \
      uint,                                                                        \
      uint3,                                                                       \
      uint3);

// WM=1, WN=1 for all configurations (single simdgroup, MFA-aligned)
// D=64: BK=32 (~14KB smem), D=96/128: BK=16 (~11/14KB smem)
// dKV dispatch always uses BK=16 for D>64 (higher per-thread register pressure
// at BK=32 outweighs benefit of fewer KV iterations for WM=1 kernel).
#define instantiate_attention_vjp_dkv_bd64(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel(tname, dtype, 32, 32, 64, 1, 1)

#define instantiate_attention_vjp_dkv_bd96(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel(tname, dtype, 32, 16, 96, 1, 1)

#define instantiate_attention_vjp_dkv_bd128(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel(tname, dtype, 32, 16, 128, 1, 1)

// WM=2 variants for D>=96 (reduced register pressure: ~280 vs ~428 regs/thread)
#define instantiate_attention_vjp_dkv_bd96_wm2(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel_wm(tname, dtype, 32, 16, 96, 2, 1)

#define instantiate_attention_vjp_dkv_bd128_wm2(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel_wm(tname, dtype, 32, 16, 128, 2, 1)

// BQ=16 WM=2 for D=128: TQ=1 per simdgroup → ~202 regs/thread (no spilling!)
// Smem: max(Q+dO(8704), red(8192)) + KV(6144) = 14848 bytes (2 TGs/core)
// Q/dO and red_smem are aliased (temporally disjoint: iterations vs reduction).
// Trade-off: 2x more Q-tile iterations vs BQ=32, but each runs spill-free.
#define instantiate_attention_vjp_dkv_bd128_bq16_wm2(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel_wm(tname, dtype, 16, 16, 128, 2, 1)

#define instantiate_attention_vjp_dkv_all(tname, dtype)     \
  instantiate_attention_vjp_dkv_bd64(tname, dtype)          \
  instantiate_attention_vjp_dkv_bd96(tname, dtype)          \
  instantiate_attention_vjp_dkv_bd128(tname, dtype)         \
  instantiate_attention_vjp_dkv_bd96_wm2(tname, dtype)      \
  instantiate_attention_vjp_dkv_bd128_wm2(tname, dtype)     \
  instantiate_attention_vjp_dkv_bd128_bq16_wm2(tname, dtype)

#ifdef VJP_UNDEF_DEFINES
  #undef VJP_GQA_FACTOR
  #undef VJP_SCALE
  #undef VJP_SCALE_LOG2
  #undef VJP_UNDEF_DEFINES
#endif
