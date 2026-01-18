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
// For each query block qb:
//   1. Load Q block, O, dO, LSE
//   2. Compute delta = sum(dO * O) per row
//   3. For each KV block kb:
//      a. Load K, V blocks
//      b. Recompute S = scale * Q @ K^T
//      c. Reconstruct P = exp2(S - LSE)
//      d. Compute dP = dO @ V^T
//      e. Compute dS = P * (dP - delta)
//      f. Accumulate dQ += scale * dS @ K
//   4. Write dQ to output
//
// See companion kernel steel_attention_vjp_dkv.h for dK/dV computation.
///////////////////////////////////////////////////////////////////////////////

using namespace mlx::steel;

// Function constants (match forward kernel indices)
constant bool align_Q_vjp_dq [[function_constant(200)]];
constant bool align_K_vjp_dq [[function_constant(201)]];
constant bool do_causal_vjp_dq [[function_constant(301)]];

///////////////////////////////////////////////////////////////////////////////
// Transform for scaling (replicated from steel_attention.h)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct TransformScaleVJPdQ {
  T scale;
  METAL_FUNC TransformScaleVJPdQ(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

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
    const device T* O [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    const device float* LSE [[buffer(5)]],
    // Gradient output (dQ only)
    device T* dQ [[buffer(6)]],
    // Parameters
    const constant AttnVJPParams* params [[buffer(7)]],
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

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  const device T* K_base = K +
      tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1];

  const device T* V_base = V +
      tidl.z * params->V_strides[0] +
      kv_head_idx * params->V_strides[1];

  const device T* O_block = O +
      tidl.z * params->O_strides[0] +
      tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  const device T* dO_block = dO +
      tidl.z * params->O_strides[0] +
      tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  const device float* LSE_block = LSE +
      (tidl.z * params->H + tidl.y) * params->LSE_strides[0] +
      tidl.x * BQ * params->LSE_strides[1];

  device T* dQ_block = dQ +
      tidl.z * params->dQ_strides[0] +
      tidl.y * params->dQ_strides[1] +
      tidl.x * BQ * params->dQ_strides[2];

  // Threadgroup memory setup
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_kv = (BK + padK) * BD > BK * (BD + padV)
                                   ? (BK + padK) * BD
                                   : BK * (BD + padV);

  threadgroup T Q_smem[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_kv];
  threadgroup T O_smem[BQ * LDV_tgp];
  threadgroup T dO_smem[BQ * LDV_tgp];
  threadgroup AccumType delta_smem[BQ];
  threadgroup AccumType lse_smem[BQ];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Block loaders
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;
  using OBlockLoader = BlockLoaderT<T, BQ, BD, LDV_tgp, 1, 1, WM * WN * 32>;

  QBlockLoader loader_q(Q_block, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K_base, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V_base, params->V_strides[2], Vs, simd_group_id, simd_lane_id);
  OBlockLoader loader_o(O_block, params->O_strides[2], O_smem, simd_group_id, simd_lane_id);
  OBlockLoader loader_do(dO_block, params->O_strides[2], dO_smem, simd_group_id, simd_lane_id);

  // Scale transform applied to Q via loader_q.apply_inplace_op(ts) after loading
  TransformScaleVJPdQ<T> ts(static_cast<T>(params->scale * M_LOG2E_F));

  // MMA setup
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "TQ must be 1");

  // MMA tiles
  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> dQtile;

  dQtile.clear();

  // Coordinates
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Os_offset = (tm + sm) * LDV_tgp + sn;
  // For dQ = dS @ K, we need K[k, d] from K_smem[d * LDK_tgp + k]
  // Fragment element at (row=k_local, col=d_local) needs K_smem[(d_tile + sn + d_local) * LDK_tgp + (k_tile + sm + k_local)]
  // Base offset uses sn for D dimension and sm for K dimension (swapped from Ks_offset)
  const short Ks_offset_for_dQ = sn * LDK_tgp + sm;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q, O, dO blocks
  if (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned) {
    loader_q.load_safe(short2(BD, params->qL_rem));
    loader_o.load_safe(short2(BD, params->qL_rem));
    loader_do.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
    loader_o.load_unsafe();
    loader_do.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load LSE
  {
    int thread_idx = simd_group_id * 32 + simd_lane_id;
    if (thread_idx < BQ) {
      if (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned && thread_idx >= params->qL_rem) {
        lse_smem[thread_idx] = 0;
      } else {
        lse_smem[thread_idx] = LSE_block[thread_idx * params->LSE_strides[1]];
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute delta = sum(dO * O) per row
  {
    int thread_idx = simd_group_id * 32 + simd_lane_id;

    // Determine actual valid rows for this query block
    // For aligned blocks: BQ rows; for last partial block: qL_rem rows
    int valid_rows = (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned)
                         ? params->qL_rem
                         : BQ;

    // Initialize
    if (thread_idx < BQ) {
      delta_smem[thread_idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread computes partial delta for its assigned rows
    // Partition work: each simdgroup handles BQ/kNWarps rows
    int rows_per_warp = BQ / kNWarps;
    int row_start = simd_group_id * rows_per_warp;
    int row_end = min(row_start + rows_per_warp, valid_rows);

    for (int row = row_start; row < row_end; row++) {
      AccumType local_delta = 0;

      // Each thread in simdgroup handles BD/32 elements
      constexpr int elems_per_lane = BD / 32;
      int d_start = simd_lane_id * elems_per_lane;

      for (int d = 0; d < elems_per_lane; d++) {
        int col = d_start + d;
        AccumType o_val = O_smem[row * LDV_tgp + col];
        AccumType do_val = dO_smem[row * LDV_tgp + col];
        local_delta += o_val * do_val;
      }

      // Reduce within simdgroup
      local_delta = simd_sum(local_delta);

      // Lane 0 writes result
      if (simd_lane_id == 0) {
        delta_smem[row] = local_delta;
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // KV loop bounds
  int kb_lim = params->NK;
  if (do_causal_vjp_dq) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  // Main loop over KV blocks
  for (int kb = 0; kb < kb_lim; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load K block
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute S = Q @ K^T
    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply sequence length mask
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

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
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

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

    // Reconstruct P = exp2(S - LSE)
    constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
    AccumType lse_vals[kRowsPT];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) {
      int row_idx = tm + sm + i * decltype(Stile)::kFragRows;
      lse_vals[i] = lse_smem[row_idx];
    }

    // P = exp2(S - LSE)
    // IMPORTANT: Clamp exp2 argument to prevent overflow from numerical precision issues.
    // In bfloat16, recomputed S may differ slightly from forward pass S, causing S > LSE.
    // Mathematically S - LSE <= 0 (since LSE = logsumexp(S) >= max(S)), so clamp to 0.
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Stile)::kElemsPerFrag; k++) {
          short row_elem = k / decltype(Stile)::MMAFrag_t::kElemCols;
          AccumType s_val = Stile.frag_at(i, j)[k];
          AccumType lse_val = lse_vals[i * decltype(Stile)::MMAFrag_t::kElemRows + row_elem];

          if (s_val > Limits<AccumType>::finite_min + 1) {
            // Clamp exp_arg to [min_arg, 0] to ensure P in (0, 1]
            // max_arg = 0 ensures P <= 1 (probability)
            // min_arg prevents underflow (exp2(-88) ≈ 0)
            AccumType exp_arg = clamp(s_val - lse_val, AccumType(-88.0f), AccumType(0.0f));
            Stile.frag_at(i, j)[k] = fast::exp2(exp_arg);
          } else {
            Stile.frag_at(i, j)[k] = AccumType(0);
          }
        }
      }
    }
    // Now Stile contains P

    // Load V block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute dP = dO @ V^T
    // We need [BQ x BD] @ [BD x BK] = [BQ x BK]
    // V is stored as [BK x BD] in V_smem, so V^T is [BD x BK]
    //
    // For V^T, the matrix has:
    //   - Row dimension = D (head_dim)
    //   - Column dimension = K (key sequence)
    //
    // V_smem layout: V_smem[k * LDV_tgp + d] = V[k, d]
    // V^T[d, k] = V[k, d] = V_smem[k * LDV_tgp + d]
    //
    // For correct transposed access:
    //   - Fragment column elements (j=0,1) should step along K dimension
    //   - This means str_y = LDV_tgp (step through V rows = V^T columns)
    //   - str_x = 1 (step through V columns = V^T rows, but kElemRows=1 so unused)

    MMATile<AccumType, TQ, TK, MMAFrag_acc_t> dPtile;
    dPtile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      // Load dO column
      MMATile<AccumType, TQ, 1, MMAFrag_acc_t> dOtile;
      dOtile.template load<T, 1, 1, LDV_tgp, 1>(&dO_smem[Os_offset + dd * kFragSize]);

      // Load V^T with transposed access pattern
      // VTtile[0, ik] represents fragment at V^T row dd*8, columns ik*8..(ik+1)*8
      // For fragment element [j], we need V^T[dd*8 + sm, ik*8 + sn + j] = V[ik*8 + sn + j, dd*8 + sm]
      // Base: V_smem[(ik*8 + sn) * LDV_tgp + dd*8 + sm]
      // Step for j: j * LDV_tgp (move to next K = next V row)
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> VTtile;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        // BD == 128 requires an extra barrier pattern.
        // On some Metal GPU/driver combinations the BD=128 configuration uses a
        // more aggressive shared-memory (threadgroup memory) layout and the
        // compiler may reorder the transposed V loads around the MMA usage.
        // The additional barriers here match the forward kernel's sequence and
        // enforce a strict ordering of:
        //   1) all stores to Vs_smem by the producer phase,
        //   2) the transposed loads from Vs into VTtile in this loop, and
        //   3) subsequent consumption by the MMA.
        // This guard is compile-time (`if constexpr`) so the barrier pattern is
        // uniform across the whole simdgroup for a given specialization.
        // Do not remove or alter this BD == 128 barrier pattern without
        // re-validating numerics and race-freedom against the forward kernel on
        // all supported GPU generations.
        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        MMAFrag_acc_t::load(
            VTtile.frag_at(0, ik),
            &Vs[(ik * kFragSize + sn) * LDV_tgp + dd * kFragSize + sm],
            Int<1>{},
            Int<LDV_tgp>{});

        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }
      }

      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(dPtile, dOtile, VTtile, dPtile);
    }

    // Compute dS = P * (dP - delta)
    AccumType delta_vals[kRowsPT];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) {
      int row_idx = tm + sm + i * decltype(Stile)::kFragRows;
      delta_vals[i] = delta_smem[row_idx];  // No clamping - production impls don't clamp delta
    }

    // dS = P * (dP - delta)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Stile)::kElemsPerFrag; k++) {
          short row_elem = k / decltype(Stile)::MMAFrag_t::kElemCols;
          AccumType p_val = Stile.frag_at(i, j)[k];  // P is stored in Stile
          AccumType dp_val = dPtile.frag_at(i, j)[k];
          AccumType d_val = delta_vals[i * decltype(Stile)::MMAFrag_t::kElemRows + row_elem];

          // Compute dS - no clamping needed per production FlashAttention
          AccumType dS_val = p_val * (dp_val - d_val);

          // Store dS back into dPtile (reusing the tile)
          dPtile.frag_at(i, j)[k] = dS_val;
        }
      }
    }
    // Now dPtile contains dS

    // CRITICAL: Reload K since V overwrote it (K and V share KV_smem buffer)
    // Must happen after dS computation (which uses V) but before dQ computation (which uses K)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate dQ += dS @ K
    // dS is [BQ x BK], K is [BK x BD]
    // Result dQ is [BQ x BD]
    //
    // K_smem stores K transposed: K_smem[d * LDK_tgp + k] = K[k, d]
    // Memory layout: Ks[d * LDK_tgp + k] where d is head_dim, k is key position
    //
    // For dQ = dS @ K, we compute:
    //   dQ[q, d] = Σ_k dS[q, k] * K[k, d]
    //
    // MMA operand B (Kfrag) must be shaped [K, D] for the matmul dS @ K.
    // Fragment element at (row=k_local, col=d_local) needs K[k_tile + sm + k_local, d_tile + sn + d_local]
    // From K_smem: K_smem[(d_tile + sn + d_local) * LDK_tgp + (k_tile + sm + k_local)]
    //
    // Base address: Ks[(dd * kFragSize + sn) * LDK_tgp + (ik * kFragSize + sm)]
    //             = Ks[dd * kFragSize * LDK_tgp + Ks_offset_for_dQ + ik * kFragSize]
    //
    // With str_x=1 (row stride) and str_y=LDK_tgp (col stride):
    //   - Moving along fragment rows (K dim): step by 1 in K_smem
    //   - Moving along fragment cols (D dim): step by LDK_tgp in K_smem
    // This correctly reconstructs K[k, d] from the transposed storage.

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        simdgroup_barrier(mem_flags::mem_none);

        MMATile<AccumType, 1, 1, MMAFrag_acc_t> Kfrag;
        MMAFrag_acc_t::load(
            Kfrag.frag_at(0, 0),
            &Ks[dd * kFragSize * LDK_tgp + Ks_offset_for_dQ + ik * kFragSize],
            Int<1>{},         // str_x = 1 (row stride along K dimension)
            Int<LDK_tgp>{});  // str_y = LDK_tgp (col stride along D dimension)

        simdgroup_barrier(mem_flags::mem_none);

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          MMAFrag_acc_t::mma(
              dQtile.frag_at(iq, dd),
              dPtile.frag_at(iq, ik),  // dS
              Kfrag.frag_at(0, 0),
              dQtile.frag_at(iq, dd));
        }
      }
    }

    // NOTE: dK and dV accumulation REMOVED - handled by separate dKV kernel

    loader_k.next();
    loader_v.next();
  }

  // Write dQ output
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Apply scale factor to dQ
  // The mathematical gradient is: dQ = scale * dS @ K
  // where scale = 1/sqrt(head_dim)
  //
  // Comparison with dKV kernel:
  // - dKV uses Q_scaled (which has scale * log2e baked in), then multiplies by inv_log2e
  //   Result: dK = dS^T @ (Q * scale * log2e) * inv_log2e = scale * dS^T @ Q ✓
  // - dQ uses unscaled K, so we must multiply by scale directly
  //   Result: dQ = dS @ K * scale ✓
  const AccumType scale = static_cast<AccumType>(params->scale);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < decltype(dQtile)::kTileRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < decltype(dQtile)::kTileCols; j++) {
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < decltype(dQtile)::kElemsPerFrag; k++) {
        dQtile.frag_at(i, j)[k] *= scale;
      }
    }
  }

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
  [[kernel]] void attention_vjp_dq<dtype, bq, bk, bd, wm, wn>(               \
      const device dtype*,                                                 \
      const device dtype*,                                                 \
      const device dtype*,                                                 \
      const device dtype*,                                                 \
      const device dtype*,                                                 \
      const device float*,                                                 \
      device dtype*,                                                       \
      const constant AttnVJPParams*,                                       \
      uint, uint, uint3, uint3);

// Common configurations - match forward STEEL attention
// For bd < 128: bk=32, for bd=128: bk=16
// Only D=64 fits within Metal's 32KB threadgroup memory limit for VJP kernels.
// D=96 requires ~35KB, D=128 requires ~37KB, so they are not instantiated.
#define instantiate_attention_vjp_dq_all(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 32, 64, 4, 1)
