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

// Lower clamp for exp2 arguments to prevent underflow (exp2(-88) ≈ 3.2e-27)
constexpr constant float kExp2MinArg = -88.0f;

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
  // dO is loaded once into shared memory (constant across KV iterations).
  // O is read from device memory (only used once for delta computation).
  // K and V always have separate buffers (no aliasing), eliminating K register
  // caching and 2 barriers per KV iteration for BD>=128.
  //
  // Memory budget (2 bytes per half element, all fit within 32KB):
  //   BD=64,  BK=32: Q(4608)+K(5120)+V(4608)+dO(4608)+misc(256) = 19,200
  //   BD=96,  BK=16: Q(6656)+K(4608)+V(3328)+dO(6656)+misc(256) = 21,504
  //   BD=128, BK=16: Q(8704)+K(6144)+V(4352)+dO(8704)+misc(256) = 28,160
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  threadgroup T Q_smem[BQ * LDQ_tgp];
  threadgroup T K_smem[BD * LDK_tgp]; // K stored transposed
  threadgroup T V_smem[BK * LDV_tgp]; // V stored row-major (separate from K)
  threadgroup T dO_smem[BQ * LDV_tgp]; // dO loaded once, reused across KV iters
  threadgroup AccumType delta_smem[BQ];
  threadgroup AccumType lse_smem[BQ];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = K_smem;
  threadgroup T* Vs = V_smem;

  // Block loaders
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;
  using dOBlockLoader = BlockLoaderT<T, BQ, BD, LDV_tgp, 1, 1, WM * WN * 32>;

  QBlockLoader loader_q(Q_block, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K_base, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V_base, params->V_strides[2], Vs, simd_group_id, simd_lane_id);
  dOBlockLoader loader_do(dO_block, params->O_strides[2], dO_smem, simd_group_id, simd_lane_id);

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
  const short dOs_offset = (tm + sm) * LDV_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  // For dQ = dS @ K, we need K[k, d] from K_smem[d * LDK_tgp + k]
  // Fragment element at (row=k_local, col=d_local) needs K_smem[(d_tile + sn + d_local) * LDK_tgp + (k_tile + sm + k_local)]
  // Base offset uses sn for D dimension and sm for K dimension (swapped from Ks_offset)
  const short Ks_offset_for_dQ = sn * LDK_tgp + sm;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  // Load Q and dO into shared memory (both are constant across KV iterations)
  if (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned) {
    loader_q.load_safe(short2(BD, params->qL_rem));
    loader_do.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
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
  // Both O and dO are read from device memory (no shared memory buffer).
  {
    int thread_idx = simd_group_id * 32 + simd_lane_id;

    int valid_rows = (!align_Q_vjp_dq && int(tid.x) == params->NQ_aligned)
                         ? params->qL_rem
                         : BQ;

    if (thread_idx < BQ) {
      delta_smem[thread_idx] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int rows_per_warp = BQ / kNWarps;
    int row_start = simd_group_id * rows_per_warp;
    int row_end = min(row_start + rows_per_warp, valid_rows);

    for (int row = row_start; row < row_end; row++) {
      AccumType local_delta = 0;

      constexpr int elems_per_lane = BD / 32;
      int d_start = simd_lane_id * elems_per_lane;

      for (int d = 0; d < elems_per_lane; d++) {
        int col = d_start + d;
        AccumType o_val = static_cast<AccumType>(O_block[row * params->O_strides[2] + col]);
        AccumType do_val = static_cast<AccumType>(dO_block[row * params->O_strides[2] + col]);
        local_delta += o_val * do_val;
      }

      local_delta = simd_sum(local_delta);

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

    // Load K and V blocks (separate buffers, no aliasing)
    if (!align_K_vjp_dq && kb == params->NK_aligned) {
      loader_k.load_safe(short2(BD, params->kL_rem));
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
      loader_v.load_unsafe();
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
            AccumType exp_arg = clamp(s_val - lse_val, AccumType(kExp2MinArg), AccumType(0.0f));
            Stile.frag_at(i, j)[k] = fast::exp2(exp_arg);
          } else {
            Stile.frag_at(i, j)[k] = AccumType(0);
          }
        }
      }
    }
    // Now Stile contains P

    // K stays valid in K_smem (separate from V), no register caching needed.

    // Compute dP = dO @ V^T
    // dO from shared memory (loaded once before KV loop). V^T from shared memory.
    MMATile<AccumType, TQ, TK, MMAFrag_acc_t> dPtile;
    dPtile.clear();

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      MMATile<AccumType, TQ, 1, MMAFrag_acc_t> dOtile;
      dOtile.template load<T, 1, 1, LDV_tgp, 1>(&dO_smem[dOs_offset + dd * kFragSize]);

      MMATile<AccumType, 1, TK, MMAFrag_acc_t> VTtile;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
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
      delta_vals[i] = delta_smem[row_idx];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Stile)::kElemsPerFrag; k++) {
          short row_elem = k / decltype(Stile)::MMAFrag_t::kElemCols;
          AccumType p_val = Stile.frag_at(i, j)[k];
          AccumType dp_val = dPtile.frag_at(i, j)[k];
          AccumType d_val = delta_vals[i * decltype(Stile)::MMAFrag_t::kElemRows + row_elem];
          dPtile.frag_at(i, j)[k] = p_val * (dp_val - d_val);
        }
      }
    }
    // Now dPtile contains dS

    // Accumulate dQ += dS @ K (K is always in K_smem, never overwritten)
    {
      MMATile<AccumType, TK, TD, MMAFrag_acc_t> K_tile;
      simdgroup_barrier(mem_flags::mem_none);
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          MMAFrag_acc_t::load(
              K_tile.frag_at(ik, dd),
              &Ks[dd * kFragSize * LDK_tgp + Ks_offset_for_dQ + ik * kFragSize],
              Int<1>{},
              Int<LDK_tgp>{});
        }
      }
      simdgroup_barrier(mem_flags::mem_none);

      // dQ[TQ x TD] += dS[TQ x TK] @ K[TK x TD]
      tile_matmad(dQtile, dPtile, K_tile, dQtile);
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

// Common configurations (2 bytes per half in threadgroup memory):
// D=64: BK=32, Q+K+V+dO = ~19KB
// D=96: BK=16, Q+K+V+dO = ~22KB
// D=128: BK=16, Q+KV(aliased)+dO = ~24KB
#define instantiate_attention_vjp_dq_bd64(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 32, 64, 4, 1)

#define instantiate_attention_vjp_dq_bd96(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 16, 96, 4, 1)

#define instantiate_attention_vjp_dq_bd128(tname, dtype) \
  instantiate_attention_vjp_dq_kernel(tname, dtype, 32, 16, 128, 4, 1)

#define instantiate_attention_vjp_dq_all(tname, dtype) \
  instantiate_attention_vjp_dq_bd64(tname, dtype)      \
  instantiate_attention_vjp_dq_bd96(tname, dtype)      \
  instantiate_attention_vjp_dq_bd128(tname, dtype)
