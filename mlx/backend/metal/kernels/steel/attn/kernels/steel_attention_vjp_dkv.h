// Copyright © 2024-25 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// STEEL VJP dKV Kernel for Scaled Dot-Product Attention
//
// Part of the two-kernel backward pass optimization that eliminates atomic
// operations. This kernel computes dK and dV gradients using a REVERSED loop
// structure compared to the dQ kernel.
//
// Grid: [NK, num_kv_heads, B] - one threadgroup per (kv_block, kv_head, batch)
// Loop: Over gqa_factor query heads, then over Q blocks to accumulate dK/dV
//
// For each KV block kb:
//   1. Load K, V blocks ONCE (resident for entire kernel)
//   2. Initialize dK, dV accumulators
//   3. For each Q block qb (that can attend to this KV block):
//      a. Load Q, O, dO blocks
//      b. Load LSE for this Q block
//      c. Compute delta = sum(dO * O) per row
//      d. Recompute S = scale * Q @ K^T
//      e. Reconstruct P = exp2(S - LSE)
//      f. Compute dP = dO @ V^T
//      g. Compute dS = P * (dP - delta)
//      h. Accumulate dK += dS^T @ Q (no atomics - local to threadgroup)
//      i. Accumulate dV += P^T @ dO (no atomics - local to threadgroup)
//   4. Write dK, dV to output (each simdgroup writes its owned K-rows directly)
//
// Key insight: By reversing the loop structure (outer=KV, inner=Q), each
// threadgroup owns its dK/dV output entirely, eliminating atomic operations.
//
// See companion kernel steel_attention_vjp_dq.h for dQ computation.
///////////////////////////////////////////////////////////////////////////////

using namespace mlx::steel;

// Function constants (match forward kernel indices)
constant bool align_Q_vjp_dkv [[function_constant(200)]];
constant bool align_K_vjp_dkv [[function_constant(201)]];
constant bool do_causal_vjp_dkv [[function_constant(301)]];

///////////////////////////////////////////////////////////////////////////////
// Transform for scaling (replicated from steel_attention.h)
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct TransformScaleVJPdKV {
  T scale;
  METAL_FUNC TransformScaleVJPdKV(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

///////////////////////////////////////////////////////////////////////////////
// STEEL Attention VJP dKV Kernel
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
void attention_vjp_dkv(
    // Forward inputs
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device T* O [[buffer(3)]],
    const device T* dO [[buffer(4)]],
    const device float* LSE [[buffer(5)]],
    // Gradient outputs (dK and dV)
    device T* dK [[buffer(6)]],
    device T* dV [[buffer(7)]],
    // Parameters
    const constant AttnVJPParams* params [[buffer(8)]],
    // Thread info
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  // clang-format on

  (void)lid;

  // tid.x = kb (KV block index) - REVERSED from dQ kernel where tid.x = qb
  // tid.y = head index
  // tid.z = batch index
  int kb = tid.x;

  ulong3 tidl{tid.x, tid.y, tid.z};

  // KV head index - grid now dispatches over KV heads directly (not query
  // heads) This avoids the race condition where multiple query heads wrote to
  // the same dK/dV
  ulong kv_head_idx = int(tid.y);

  // K, V base pointers - these will be loaded once and stay resident
  const device T* K_block = K + tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1] + kb * BK * params->K_strides[2];

  const device T* V_block = V + tidl.z * params->V_strides[0] +
      kv_head_idx * params->V_strides[1] + kb * BK * params->V_strides[2];

  // Q, O, dO, LSE base pointers are computed per query head inside the GQA loop
  // below They depend on the query head index (q_head_idx), not the KV head
  // index (kv_head_idx)

  // Output pointers - direct write (no atomics)
  device T* dK_block = dK + tidl.z * params->dK_strides[0] +
      kv_head_idx * params->dK_strides[1] + kb * BK * params->dK_strides[2];

  device T* dV_block = dV + tidl.z * params->dV_strides[0] +
      kv_head_idx * params->dV_strides[1] + kb * BK * params->dV_strides[2];

  // Threadgroup memory setup
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  // Memory for Q/dO (reused per iteration) and K/V (resident)
  threadgroup T Q_smem[BQ * LDQ_tgp];
  // K is stored transposed: K_smem[d * LDK_tgp + k] = K[k, d]
  // So K_smem needs BD rows (one per head dim), each with LDK_tgp = BK + pad
  // elements
  threadgroup T K_smem[BD * LDK_tgp]; // K is resident (transposed)
  threadgroup T V_smem[BK * LDV_tgp]; // V is resident
  // O_smem is later reinterpreted as AccumType* (float) for dS_transposed.
  // Use alignas to ensure proper alignment for both T (half/bfloat16) and
  // AccumType (float).
  alignas(sizeof(AccumType)) threadgroup T O_smem[BQ * LDV_tgp];
  threadgroup T dO_smem[BQ * LDV_tgp];
  threadgroup AccumType delta_smem[BQ];
  threadgroup AccumType lse_smem[BQ];

  // Stride for dS transpose operation
  // dS_transposed reuses O_smem (reinterpreted as float)
  // O_smem bytes = BQ * LDV_tgp * sizeof(T)
  // For bfloat16: BQ * (BD + 8) * 2 bytes
  // dS_transposed needs: BK * LD_dST * sizeof(float) = BK * LD_dST * 4 bytes
  // Using LD_dST = BQ ensures it fits: BK * BQ * 4 <= BQ * (BD + 8) * 2
  //   => BK * 4 <= (BD + 8) * 2 => BK * 2 <= BD + 8 => BK <= (BD + 8) / 2
  //   For BD=64: BK <= 36 (BK=32 fits), For BD=128: BK <= 68 (BK=16 fits)
  constexpr short LD_dST =
      BQ; // Stride for dS_transposed (BK x LD_dST), no padding

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = K_smem;
  threadgroup T* Vs = V_smem;

  // Block loaders for K/V (loaded once)
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

  KBlockLoader loader_k(
      K_block, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V_block, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  TransformScaleVJPdKV<T> ts(static_cast<T>(params->scale * M_LOG2E_F));

  // MMA setup
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  // K-row ownership: distribute BK rows across kNWarps simdgroups
  // Each simdgroup owns a contiguous range of K rows for dV computation
  // This eliminates the 75% compute waste from single-SG dV computation
  constexpr short k_rows_per_sg = BK / kNWarps; // e.g., 16/4 = 4 rows per SG

  static_assert(TQ == 1, "TQ must be 1");
  static_assert(
      BK % kNWarps == 0, "BK must be divisible by kNWarps for K-row ownership");

  // O_smem reuse safety: O_smem is reinterpreted as dS_transposed after delta
  // computation. O_smem size:        BQ * LDV_tgp * sizeof(T) = BQ * (BD +
  // padV) * sizeof(T) dS_transposed size: BK * LD_dST * sizeof(AccumType) = BK
  // * BQ * sizeof(float) Constraint: BK * BQ * 4 <= BQ * (BD + padV) *
  // sizeof(T)
  //          => BK * 4 <= (BD + padV) * sizeof(T)
  // For bf16/fp16 (sizeof(T)=2): BK * 4 <= (BD + 8) * 2 => BK * 2 <= BD + 8
  static_assert(
      BK * sizeof(AccumType) <= LDV_tgp * sizeof(T),
      "O_smem buffer too small for dS_transposed reinterpretation: "
      "BK * sizeof(AccumType) must be <= LDV_tgp * sizeof(T)");

  // Coordinates
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  // K-row ownership bounds for this simdgroup
  // Used to distribute dV computation and output writes across all simdgroups
  const short my_k_start = simd_group_id * k_rows_per_sg;
  const short my_k_end = my_k_start + k_rows_per_sg;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Os_offset = (tm + sm) * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  // =========================================================================
  // Load K, V blocks ONCE at the start (resident for entire kernel)
  // =========================================================================
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_K_vjp_dkv && kb == params->NK_aligned) {
    loader_k.load_safe(short2(BD, params->kL_rem));
    loader_v.load_safe(short2(BD, params->kL_rem));
  } else {
    loader_k.load_unsafe();
    loader_v.load_unsafe();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // =========================================================================
  // Initialize dK, dV accumulators (FP32 for precision)
  // These accumulate across all Q blocks AND all query heads in the GQA group
  // =========================================================================
  // Each simdgroup owns a contiguous range of K rows (BK/kNWarps rows per SG)
  // and accumulates dK/dV only for its owned K-rows, then writes directly
  // Full tile: [TK x TD] fragments, each [kFragSize x kFragSize]
  MMATile<AccumType, TK, TD, MMAFrag_acc_t> dKtile;
  MMATile<AccumType, TK, TD, MMAFrag_acc_t> dVtile;
  dKtile.clear();
  dVtile.clear();

  // =========================================================================
  // Determine Q block loop bounds (same for all query heads in the GQA group)
  // For causal masking: only include Q blocks where queries can attend to this
  // KV block Q block qb can attend to KV block kb if: qb*BQ + qL_off >= kb*BK
  // (some query in block) Equivalently: first valid qb is (kb * BK) / BQ
  // (rounded down)
  // =========================================================================
  int qb_start = 0;
  if (do_causal_vjp_dkv) {
    // For causal attention, skip Q blocks that are entirely before this KV
    // block Query at position q can attend to key at position k if q >= k So Q
    // block qb can have some queries attending to K block kb if:
    //   qb*BQ + BQ - 1 + qL_off >= kb*BK
    // => qb >= (kb*BK - qL_off - BQ + 1) / BQ
    int k_start = kb * BK;
    qb_start = max(0, (k_start - params->qL_off) / BQ);
  }

  // =========================================================================
  // Main loop: iterate over Q blocks first, then GQA heads
  // This ordering keeps K/V tiles in registers better across GQA iterations
  // since the same K/V block is reused for all GQA heads at the same position.
  //
  // For GQA, multiple query heads share the same K/V, so we must accumulate
  // dK/dV contributions from all of them (gqa_factor query heads per KV head)
  // =========================================================================
  for (int qb = qb_start; qb < params->NQ; qb++) {
    // Inner loop over GQA heads - K/V tiles stay in registers
    for (int gqa_idx = 0; gqa_idx < params->gqa_factor; gqa_idx++) {
      // Compute query head index for this iteration
      ulong q_head_idx = kv_head_idx * params->gqa_factor + gqa_idx;

      // Q, O, dO, LSE base pointers for this query head
      const device T* Q_base =
          Q + tidl.z * params->Q_strides[0] + q_head_idx * params->Q_strides[1];

      const device T* O_base =
          O + tidl.z * params->O_strides[0] + q_head_idx * params->O_strides[1];

      const device T* dO_base =
          dO + tidl.z * params->O_strides[0] + q_head_idx * params->O_strides[1];

      const device float* LSE_base =
          LSE + (tidl.z * params->H + q_head_idx) * params->LSE_strides[0];
      // Q, O, dO pointers for this Q block
      const device T* Q_block = Q_base + qb * BQ * params->Q_strides[2];
      const device T* O_block = O_base + qb * BQ * params->O_strides[2];
      const device T* dO_block = dO_base + qb * BQ * params->O_strides[2];
      const device float* LSE_block =
          LSE_base + qb * BQ * params->LSE_strides[1];

      // Block loaders for Q, O, dO (loaded fresh each iteration)
      using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
      using OBlockLoader = BlockLoaderT<T, BQ, BD, LDV_tgp, 1, 1, WM * WN * 32>;

      QBlockLoader loader_q(
          Q_block, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
      OBlockLoader loader_o(
          O_block, params->O_strides[2], O_smem, simd_group_id, simd_lane_id);
      OBlockLoader loader_do(
          dO_block, params->O_strides[2], dO_smem, simd_group_id, simd_lane_id);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load Q, O, dO for this Q block
      if (!align_Q_vjp_dkv && qb == params->NQ_aligned) {
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

      // Load LSE for this Q block
      {
        int thread_idx = simd_group_id * 32 + simd_lane_id;
        if (thread_idx < BQ) {
          if (!align_Q_vjp_dkv && qb == params->NQ_aligned &&
              thread_idx >= params->qL_rem) {
            lse_smem[thread_idx] = 0;
          } else {
            lse_smem[thread_idx] =
                LSE_block[thread_idx * params->LSE_strides[1]];
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // =========================================================================
      // Compute delta = sum(dO * O) per row for this Q block
      // =========================================================================
      {
        int thread_idx = simd_group_id * 32 + simd_lane_id;

        int valid_rows = (!align_Q_vjp_dkv && qb == params->NQ_aligned)
            ? params->qL_rem
            : BQ;

        // Initialize
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
            AccumType o_val = O_smem[row * LDV_tgp + col];
            AccumType do_val = dO_smem[row * LDV_tgp + col];
            local_delta += o_val * do_val;
          }

          local_delta = simd_sum(local_delta);

          if (simd_lane_id == 0) {
            delta_smem[row] = local_delta;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // =========================================================================
      // Compute S = Q @ K^T (we have K resident in K_smem)
      // =========================================================================
      MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
      MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
      MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
      Stile.clear();

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
        Ktile.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }

      // Apply sequence length mask (for partial K block)
      if (!align_K_vjp_dkv && kb == params->NK_aligned) {
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
      // For dKV kernel: query row position must be >= key column position
      if (do_causal_vjp_dkv) {
        using stile_t = decltype(Stile);
        constexpr auto neg_inf = Limits<AccumType>::finite_min;

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < stile_t::kTileRows; i++) {
          const int row_pos =
              qb * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
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

      // =========================================================================
      // Reconstruct P = exp2(S - LSE)
      // =========================================================================
      constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
      AccumType lse_vals[kRowsPT];

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; i++) {
        int row_idx = tm + sm + i * decltype(Stile)::kFragRows;
        lse_vals[i] = lse_smem[row_idx];
      }

      // P = exp2(S - LSE)
      // IMPORTANT: Clamp exp2 argument to prevent overflow from numerical
      // precision issues. In bfloat16, recomputed S may differ slightly from
      // forward pass S, causing S > LSE. Mathematically S - LSE <= 0 (since LSE
      // = logsumexp(S) >= max(S)), so clamp to 0.
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < decltype(Stile)::kElemsPerFrag; k++) {
            short row_elem = k / decltype(Stile)::MMAFrag_t::kElemCols;
            AccumType s_val = Stile.frag_at(i, j)[k];
            AccumType lse_val =
                lse_vals[i * decltype(Stile)::MMAFrag_t::kElemRows + row_elem];

            if (s_val > Limits<AccumType>::finite_min + 1) {
              // Clamp exp_arg to [min_arg, 0] to ensure P in (0, 1]
              // max_arg = 0 ensures P <= 1 (probability)
              // min_arg prevents underflow (exp2(-88) ≈ 0)
              AccumType exp_arg =
                  clamp(s_val - lse_val, AccumType(-88.0f), AccumType(0.0f));
              Stile.frag_at(i, j)[k] = fast::exp2(exp_arg);
            } else {
              Stile.frag_at(i, j)[k] = AccumType(0);
            }
          }
        }
      }
      // Now Stile contains P

      // =========================================================================
      // Compute dP = dO @ V^T (we have V resident in V_smem)
      // dP: [BQ x BK]
      //
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
      //   - str_x = 1 (step through V columns = V^T rows, but kElemRows=1 so
      //   unused)
      // =========================================================================
      MMATile<AccumType, TQ, TK, MMAFrag_acc_t> dPtile;
      dPtile.clear();

      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);

        // Load dO column
        MMATile<AccumType, TQ, 1, MMAFrag_acc_t> dOtile;
        dOtile.template load<T, 1, 1, LDV_tgp, 1>(
            &dO_smem[Os_offset + dd * kFragSize]);

        // Load V^T with transposed access pattern
        // VTtile[0, ik] represents fragment at V^T row dd*8, columns
        // ik*8..(ik+1)*8 For fragment element [j], we need V^T[dd*8 + sm, ik*8
        // + sn + j] = V[ik*8 + sn + j, dd*8 + sm] Base: V_smem[(ik*8 + sn) *
        // LDV_tgp + dd*8 + sm] Step for j: j * LDV_tgp (move to next K = next V
        // row)
        MMATile<AccumType, 1, TK, MMAFrag_acc_t> VTtile;
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          // BD==128 barrier pattern: match forward kernel for memory access
          // synchronization
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

      // =========================================================================
      // Compute dS = P * (dP - delta)
      // =========================================================================
      AccumType delta_vals[kRowsPT];

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; i++) {
        int row_idx = tm + sm + i * decltype(Stile)::kFragRows;
        delta_vals[i] = delta_smem[row_idx];
      }

      // dS = P * (dP - delta)
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Stile)::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(Stile)::kTileCols; j++) {
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < decltype(Stile)::kElemsPerFrag; k++) {
            short row_elem = k / decltype(Stile)::MMAFrag_t::kElemCols;
            AccumType p_val = Stile.frag_at(i, j)[k];
            AccumType dp_val = dPtile.frag_at(i, j)[k];
            AccumType d_val = delta_vals
                [i * decltype(Stile)::MMAFrag_t::kElemRows + row_elem];

            // Store dS in dPtile (reusing)
            dPtile.frag_at(i, j)[k] = p_val * (dp_val - d_val);
          }
        }
      }
      // Now dPtile contains dS, Stile contains P

      // =========================================================================
      // Store dS to shared memory and reload transposed for dK computation
      // MMA computes A @ B, but we need A^T @ B (dS^T @ Q)
      // By storing dS as [K x Q] layout, we can read transposed values directly
      // =========================================================================
      // Reuse O_smem (no longer needed after delta computation) for dS
      // transpose Memory requirement: BK * LD_dST * sizeof(float) <= BQ *
      // LDV_tgp * sizeof(T) This is validated at compile time via the stride
      // definitions above
      threadgroup AccumType* dS_transposed =
          reinterpret_cast<threadgroup AccumType*>(O_smem);

      // Initialize dS_transposed to zero (important for partial Q blocks)
      {
        int thread_idx = simd_group_id * 32 + simd_lane_id;
        int total_threads = kNWarps * 32;
        int total_elems = BK * LD_dST;

        for (int idx = thread_idx; idx < total_elems; idx += total_threads) {
          dS_transposed[idx] = AccumType(0);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Store dPtile to shared memory as [Q x K], will be read as [K x Q]
      // Each thread stores its 2 elements of each fragment
      // IMPORTANT: Skip invalid rows for partial Q blocks to avoid NaN
      // propagation
      {
        // Compute valid Q rows for partial block check
        int valid_q_rows = (!align_Q_vjp_dkv && qb == params->NQ_aligned)
            ? params->qL_rem
            : BQ;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          short q_row = tm + iq * kFragSize + sm; // Global Q row within block
          // Skip invalid rows in partial Q blocks
          if (q_row >= valid_q_rows)
            continue;
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            short k_col_base = ik * kFragSize + sn; // Global K column base
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < 2; kk++) {
              short k_col = k_col_base + kk;
              // Store at position [k_col, q_row] for transposed access
              dS_transposed[k_col * LD_dST + q_row] =
                  dPtile.frag_at(iq, ik)[kk];
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // =========================================================================
      // Accumulate dK using the transposed dS
      // dK[k, d] = Σ_q dS^T[k, q] * Q_scaled[q, d]
      // Sum over ALL Q rows in this Q block (BQ rows total)
      //
      // K-row ownership: each simdgroup only accumulates K-rows it owns
      // This distributes dK computation across all kNWarps simdgroups (~4x
      // speedup)
      // =========================================================================
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        short k_row = ik * kFragSize + sm;

        // Only accumulate K-rows owned by this simdgroup
        if (k_row < my_k_start || k_row >= my_k_end)
          continue;

        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
          STEEL_PRAGMA_UNROLL
          for (short df = 0; df < 2; df++) {
            short d_col = dd * kFragSize + sn + df;

            // Accumulate: dK[k_row, d_col] = Σ_q dS^T[k_row, q] * Q[q, d_col]
            AccumType accum = 0;

            STEEL_PRAGMA_UNROLL
            for (short qq = 0; qq < BQ; qq++) {
              AccumType dS_T_val = dS_transposed[k_row * LD_dST + qq];
              // Read Q as T (bfloat16/float16) and explicitly convert to
              // AccumType
              T Q_raw = Qs[qq * LDQ_tgp + d_col];
              AccumType Q_val = static_cast<AccumType>(Q_raw);
              accum += dS_T_val * Q_val;
            }

            dKtile.frag_at(ik, dd)[df] += accum;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // =========================================================================
      // Store P (Stile) to shared memory for transposed loading (for dV)
      // IMPORTANT: Skip invalid rows for partial Q blocks to avoid NaN
      // propagation
      // =========================================================================
      {
        // Compute valid Q rows for partial block check
        int valid_q_rows = (!align_Q_vjp_dkv && qb == params->NQ_aligned)
            ? params->qL_rem
            : BQ;

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          short q_row = tm + iq * kFragSize + sm;
          // Skip invalid rows in partial Q blocks
          if (q_row >= valid_q_rows)
            continue;
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            short k_col_base = ik * kFragSize + sn;
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < 2; kk++) {
              short k_col = k_col_base + kk;
              dS_transposed[k_col * LD_dST + q_row] = Stile.frag_at(iq, ik)[kk];
            }
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Accumulate dV using the transposed P
      // dV[k, d] = Σ_q P^T[k, q] * dO[q, d]
      // Sum over ALL Q rows in this Q block (BQ rows total)
      //
      // K-row ownership: each simdgroup only accumulates K-rows it owns
      // This distributes dV computation across all kNWarps simdgroups (~4x
      // speedup)
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        short k_row = ik * kFragSize + sm;

        // Only accumulate K-rows owned by this simdgroup
        if (k_row < my_k_start || k_row >= my_k_end)
          continue;

        STEEL_PRAGMA_UNROLL
        for (short dd = 0; dd < TD; dd++) {
          STEEL_PRAGMA_UNROLL
          for (short df = 0; df < 2; df++) {
            short d_col = dd * kFragSize + sn + df;

            // Accumulate: dV[k_row, d_col] = Σ_q P^T[k_row, q] * dO[q, d_col]
            AccumType accum = 0;

            STEEL_PRAGMA_UNROLL
            for (short qq = 0; qq < BQ; qq++) {
              AccumType P_T_val = dS_transposed[k_row * LD_dST + qq];
              // Read dO as T (bfloat16/float16) and explicitly convert to
              // AccumType
              T dO_raw = dO_smem[qq * LDV_tgp + d_col];
              AccumType dO_val = static_cast<AccumType>(dO_raw);
              accum += P_T_val * dO_val;
            }

            dVtile.frag_at(ik, dd)[df] += accum;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

    } // End GQA loop
  } // End Q block loop

  // Post-process dK: Q was scaled by (scale * M_LOG2E), need to divide by
  // M_LOG2E dK = dS^T @ Q_scaled / M_LOG2E = dS^T @ (Q * scale * M_LOG2E) /
  // M_LOG2E
  //    = scale * dS^T @ Q
  // So dK is correctly scaled, just need to undo the log2 factor
  constexpr AccumType inv_log2e = AccumType(1.0f / M_LOG2E_F);

  // =========================================================================
  // Write dK and dV to device memory
  // Use direct tile-based writes like dQ kernel - no staging needed
  // This avoids buffer overflow from trying to stage in too-small K_smem/V_smem
  // =========================================================================
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Apply log2e correction to dK
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++) {
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      STEEL_PRAGMA_UNROLL
      for (short df = 0; df < 2; df++) {
        dKtile.frag_at(ik, dd)[df] *= inv_log2e;
      }
    }
  }

  // =========================================================================
  // Write dK and dV to device memory
  // Use direct element-based writes - each thread writes its owned positions
  // This avoids buffer overflow from staging in too-small shared memory buffers
  //
  // K-row ownership: each simdgroup writes its owned K-rows
  // - dK: each simdgroup computed only its owned rows, writes them
  // - dV: each simdgroup computed only its owned rows, writes them
  // This distributes output writes across all kNWarps simdgroups (~4x speedup)
  // =========================================================================
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int k_rem =
      (!align_K_vjp_dkv && kb == params->NK_aligned) ? params->kL_rem : BK;
  int64_t dk_stride = params->dK_strides[2];
  int64_t dv_stride = params->dV_strides[2];

  // Each simdgroup writes K-rows it owns
  // Each thread writes elements based on its (sm, sn) fragment coordinates:
  // - k_row = {sm, sm+8} (via ik=0,1), filtered by K-row ownership
  // - d_col = dd*8 + sn + df, covers cols 0-127
  STEEL_PRAGMA_UNROLL
  for (short ik = 0; ik < TK; ik++) {
    short k_row = ik * kFragSize + sm;

    // Only write K-rows owned by this simdgroup
    if (k_row < my_k_start || k_row >= my_k_end)
      continue;
    if (k_row >= k_rem)
      continue;

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      STEEL_PRAGMA_UNROLL
      for (short df = 0; df < 2; df++) {
        short d_col = dd * kFragSize + sn + df;
        if (d_col >= BD)
          continue;

        AccumType dk_val = dKtile.frag_at(ik, dd)[df];
        AccumType dv_val = dVtile.frag_at(ik, dd)[df];

        dK_block[k_row * dk_stride + d_col] = static_cast<T>(dk_val);
        dV_block[k_row * dv_stride + d_col] = static_cast<T>(dv_val);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Template instantiation macros
///////////////////////////////////////////////////////////////////////////////

// tname is the string name used in kernel lookup (e.g., "float32", "float16")
// dtype is the actual C++ type (e.g., float, half, bfloat16_t)
#define instantiate_attention_vjp_dkv_kernel(tname, dtype, bq, bk, bd, wm, wn) \
  template [[host_name(                                                        \
      "attention_vjp_dkv_" #tname "_" #bq "_" #bk "_" #bd)]] [[kernel]] void   \
  attention_vjp_dkv<dtype, bq, bk, bd, wm, wn>(                                \
      const device dtype*,                                                     \
      const device dtype*,                                                     \
      const device dtype*,                                                     \
      const device dtype*,                                                     \
      const device dtype*,                                                     \
      const device float*,                                                     \
      device dtype*,                                                           \
      device dtype*,                                                           \
      const constant AttnVJPParams*,                                           \
      uint,                                                                    \
      uint,                                                                    \
      uint3,                                                                   \
      uint3);

// Only D=64 fits within Metal's 32KB threadgroup memory limit for VJP kernels.
// D=96 requires ~35KB, D=128 requires ~37KB, so they are not instantiated.
#define instantiate_attention_vjp_dkv_all(tname, dtype) \
  instantiate_attention_vjp_dkv_kernel(tname, dtype, 32, 32, 64, 4, 1)
