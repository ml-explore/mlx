#include <metal_stdlib>

#include "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

using namespace mlx::steel;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderFA {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC BlockLoaderFA(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() {
    src += tile_stride;
  }
  METAL_FUNC void next(short n) {
    src += n * tile_stride;
  }
};

template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    short lda_tgp,
    short ldb_tgp,
    typename AccumType = float,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMAFA {
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = 8 * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = 8 * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Strides of A, B along reduction axis
  STEEL_CONST short simd_stride_a = {
      transpose_a ? TM_stride : TM_stride * lda_tgp};
  STEEL_CONST short simd_stride_b = {
      transpose_b ? TN_stride * ldb_tgp : TN_stride};

  // Jump between elements
  STEEL_CONST short jump_a = {transpose_a ? lda_tgp : 1};
  STEEL_CONST short jump_b = {transpose_b ? ldb_tgp : 1};

  STEEL_CONST short tile_stride_a = {transpose_a ? 8 * lda_tgp : 8};
  STEEL_CONST short tile_stride_b = {transpose_b ? 8 : 8 * ldb_tgp};

  // Simdgroup matrices
  simdgroup_matrix<AccumType, 8, 8> Asimd[TM];
  simdgroup_matrix<AccumType, 8, 8> Bsimd[TN];
  simdgroup_matrix<AccumType, 8, 8> results[TM * TN] = {
      simdgroup_matrix<AccumType, 8, 8>(0)};

  // Offsets within threadgroup
  const short tm;
  const short tn;

  short sm;
  short sn;

  ushort sid;
  ushort slid;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMAFA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : tm(8 * (simd_group_id / WN)), tn(8 * (simd_group_id % WN)) {
    // Determine thread position in simdgroup matrix
    short qid = simd_lane_id / 4;
    slid = simd_lane_id;
    sid = simd_group_id;

    sm = (qid & 4) + (simd_lane_id / 2) % 4;
    sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

    // Determine thread and simdgroup offset
    As_offset =
        transpose_a ? ((sn)*lda_tgp + (tm + sm)) : ((sn) + (tm + sm) * lda_tgp);
    Bs_offset =
        transpose_b ? ((tn + sn) * ldb_tgp + (sm)) : ((sm)*ldb_tgp + (tn + sn));
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of 8
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += 8) {
      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup A as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        Asimd[i].thread_elements()[0] =
            static_cast<AccumType>(As[i * simd_stride_a + 0]);
        Asimd[i].thread_elements()[1] =
            static_cast<AccumType>(As[i * simd_stride_a + jump_a]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Load elements from threadgroup B as simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        Bsimd[j].thread_elements()[0] =
            static_cast<AccumType>(Bs[j * simd_stride_b + 0]);
        Bsimd[j].thread_elements()[1] =
            static_cast<AccumType>(Bs[j * simd_stride_b + jump_b]);
      }

      simdgroup_barrier(mem_flags::mem_none);

      // Multiply and accumulate into result simdgroup matrices
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          short j_serp = (i % 2) ? (TN - 1 - j) : j;

          simdgroup_multiply_accumulate(
              results[i * TN + j_serp],
              Asimd[i],
              Bsimd[j_serp],
              results[i * TN + j_serp]);
        }
      }

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  METAL_FUNC void rescale_output(const threadgroup float* Corrections) {
    // Loop over all simdgroup tiles

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      short row = sm + tm + i * TM_stride;
      float scale_value = Corrections[row];

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto& accum = results[i * TN + j].thread_elements();
        // int offset = (i * TM_stride) * ldc + (j * TN_stride);
        accum[0] *= scale_value;
        accum[1] *= scale_value;
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U* C, const int ldc) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset = (i * TM_stride) * ldc + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {Epilogue::apply(accum[0]), Epilogue::apply(accum[1])};

        // Write out C
        C[offset] = outs[0];
        C[offset + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_to_tgp_memory(
      threadgroup U* C,
      const int ldc,
      short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldc + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            C[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            C[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  METAL_FUNC void
  store_result_safe(device U* C, const int ldc, short2 dst_tile_dims) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn);
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset = (i * TM_stride) * ldc + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            C[offset] = Epilogue::apply(accum[0]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            C[offset + 1] = Epilogue::apply(accum[1]);
          }
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto& accum = results[i * TN + j].thread_elements();
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        U outs[2] = {
            epilogue_op.apply(accum[0], C[offset_c]),
            epilogue_op.apply(accum[1], C[offset_c + fdc])};

        // Write out D
        D[offset_d] = outs[0];
        D[offset_d + 1] = outs[1];
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      const device U* C,
      const int ldc,
      const int fdc,
      short2 dst_tile_dims,
      thread const Epilogue& epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm + tm) * ldc + (tn + sn) * fdc;
    D += (sm + tm) * ldd + tn + sn;
    dst_tile_dims -= short2(tn + sn, sm + tm);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto& accum = results[i * TN + j].thread_elements();
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue and output C
          if (j * TN_stride < dst_tile_dims.x) {
            D[offset_d] = epilogue_op.apply(accum[0], C[offset_c]);
          }

          if (j * TN_stride + 1 < dst_tile_dims.x) {
            D[offset_d + 1] = epilogue_op.apply(accum[1], C[offset_c + fdc]);
          }
        }
      }
    }
  }

  METAL_FUNC void clear_results() {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        results[i * TN + j] = simdgroup_matrix<AccumType, 8, 8>(0);
      }
    }
  }
};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<U, AccumType>>
struct FastAttentionKernel {
  STEEL_CONST short tgp_padding = 16 / sizeof(T);
  STEEL_CONST short float_padding = 16 / sizeof(float);
  STEEL_CONST short tgp_mem_size_q =
      transpose_q ? BK * (BM + tgp_padding) : BM * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_k =
      transpose_k ? BK * (BN + tgp_padding) : BN * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_v =
      transpose_v ? BK * (BN + tgp_padding) : BN * (BK + tgp_padding);
  STEEL_CONST short tgp_mem_size_s = BM * (BN + tgp_padding);

  // maxes, rowsums, rescale
  STEEL_CONST short tgp_mem_size_corrections =
      4 * (BM * sizeof(float) + float_padding);

  STEEL_CONST bool share_kv_smem = transpose_k != transpose_v;

  STEEL_CONST short tgp_mem_size = share_kv_smem
      ? tgp_mem_size_q + tgp_mem_size_k + tgp_mem_size_s +
          tgp_mem_size_corrections
      : tgp_mem_size_q + tgp_mem_size_k + tgp_mem_size_s +
          tgp_mem_size_corrections + tgp_mem_size_v;

  STEEL_CONST short tgp_size = WM * WN * 32;

  static_assert(transpose_q == false, "Expected Q not transposed.");
  static_assert(transpose_k == true, "Expected K transposed.");
  static_assert(transpose_v == false, "Expected V not transposed.");
  static_assert(tgp_mem_size <= 32768, "Excessive tgp memory requested.");

  using loader_q_t = BlockLoaderFA<
      T,
      transpose_q ? BK : BM,
      transpose_q ? BM : BK,
      transpose_q ? BM + tgp_padding : BK + tgp_padding,
      !transpose_q,
      tgp_size>;

  using loader_k_t = BlockLoaderFA<
      T,
      transpose_k ? BN : BK,
      transpose_k ? BK : BN,
      transpose_k ? BK + tgp_padding : BN + tgp_padding,
      transpose_k,
      tgp_size>;

  using loader_v_t = BlockLoaderFA<
      T,
      transpose_v ? BK : BN,
      transpose_v ? BN : BK,
      transpose_v ? BN + tgp_padding : BK + tgp_padding,
      transpose_v,
      tgp_size>;

  using mma_qk_t = BlockMMAFA<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_q,
      transpose_k,
      transpose_q ? BM + tgp_padding : BK + tgp_padding,
      transpose_k ? BK + tgp_padding : BN + tgp_padding,
      AccumType,
      Epilogue>;

  using mma_sv_t = BlockMMAFA<
      T,
      U,
      BM,
      BK,
      BN,
      WM,
      WN,
      false,
      transpose_v,
      BN + tgp_padding,
      BK + tgp_padding,
      AccumType,
      Epilogue>;

  /* Main kernel function */
  template <bool M_aligned, bool N_aligned, bool K_aligned_>
  static METAL_FUNC void gemm_loop(
      threadgroup T* As [[threadgroup(0)]],
      threadgroup T* Bs [[threadgroup(1)]],
      const int gemm_k_iterations,
      thread loader_k_t& loader_b,
      thread mma_qk_t& mma_op,
      thread const short& tgp_bm,
      thread const short& tgp_bn,
      LoopAlignment<M_aligned, N_aligned, K_aligned_> l = {}) {
    // Appease the compiler
    (void)l;
    (void)tgp_bm;

    short2 tile_dims_B = transpose_k ? short2(BK, tgp_bn) : short2(tgp_bn, BK);

    // not valid for gemm_k_iterations > 1 (so, BK == d_k)
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (N_aligned) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dims_B);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);
    }
  }

  static METAL_FUNC void initialize_corrections(
      threadgroup float* C,
      uint simd_lane_id,
      uint simd_group_id) {
    if (simd_group_id == 0) {
      threadgroup float* maxes = C;
      threadgroup float* sums = C + (BM + float_padding);
      threadgroup float* o_rescale = sums + (BM + float_padding);
      threadgroup float* output_rescale = o_rescale + (BM + float_padding);

      if (simd_lane_id < BM) {
        maxes[simd_lane_id] = -INFINITY; // m_i
        sums[simd_lane_id] = 0.f; // l_i
        o_rescale[simd_lane_id] = 1.f; // li * exp(mi - mi_new)
        output_rescale[simd_lane_id] = 1.f; // 1.0 / l_i
      }
    }
  }

  static METAL_FUNC void rescale_ss(
      threadgroup T* Ss,
      threadgroup float* Corrections,
      uint simd_group_id,
      uint simd_lane_id,
      short2 local_blocks,
      float alpha) {
    if (simd_group_id == 0) {
      short row_offset = BM + float_padding;
      threadgroup float* maxes = Corrections;
      threadgroup float* sums = Corrections + row_offset;
      threadgroup float* o_rescale = sums + row_offset;
      threadgroup float* output_scales = o_rescale + row_offset;

      if (simd_lane_id < uint(local_blocks.y)) {
        float m_i_old = maxes[simd_lane_id];
        float l_i_old = sums[simd_lane_id];

        float m_i_new = m_i_old;
        float l_i_new = l_i_old;

        short offset = simd_lane_id * (BN + tgp_padding);

        float m_ij = -INFINITY;

        for (short j = 0; j < local_blocks.x; j++) {
          float val = alpha * float(Ss[offset + j]);
          m_ij = max(m_ij, val);
        }

        m_i_new = max(m_ij, m_i_new);

        float rowsum = 0.f; // lij

        for (short j = 0; j < local_blocks.x; j++) {
          float val = alpha * float(Ss[offset + j]);
          float P_i_j = exp(val - m_ij);
          rowsum += P_i_j;
          P_i_j = P_i_j * exp(m_ij - m_i_new);
          Ss[offset + j] = T(P_i_j);
        }

        l_i_new =
            exp(m_i_old - m_i_new) * l_i_old + exp(m_ij - m_i_new) * rowsum;
        maxes[simd_lane_id] = m_i_new;
        sums[simd_lane_id] = l_i_new;
        float rescale = l_i_old * exp(m_i_old - m_i_new);
        o_rescale[simd_lane_id] = rescale;
        output_scales[simd_lane_id] = 1.0 / l_i_new;
      }
    }
  }

  /* Main kernel function */
  static METAL_FUNC void run(
      const device T* Q [[buffer(0)]],
      const device T* K [[buffer(1)]],
      const device T* V [[buffer(2)]],
      device U* O [[buffer(3)]],
      const constant MLXFastAttentionParams* params [[buffer(4)]],
      threadgroup T* Qs [[threadgroup(0)]],
      threadgroup T* Ks [[threadgroup(1)]],
      threadgroup T* Ss [[threadgroup(2)]],
      threadgroup T* Vs [[threadgroup(3)]],
      threadgroup float* Corrections [[threadgroup(4)]],
      uint simd_lane_id [[thread_index_in_simdgroup]],
      uint simd_group_id [[simdgroup_index_in_threadgroup]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]]) {
    // Pacifying compiler
    (void)lid;

    const int tid_y = ((tid.y) << params->swizzle_log) +
        ((tid.x) & ((1 << params->swizzle_log) - 1));
    const int tid_x = (tid.x) >> params->swizzle_log;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Find block in Q, O; and head in K, V.
    const int c_row = tid_y * BM;

    Q += transpose_q ? c_row : c_row * params->ldq;
    thread loader_q_t loader_q(Q, params->ldq, Qs, simd_group_id, simd_lane_id);

    short tgp_bm = min(BM, params->M - c_row);
    short2 tile_dims_Q = transpose_q ? short2(tgp_bm, BK) : short2(BK, tgp_bm);

    loader_q.load_safe(tile_dims_Q);

    initialize_corrections(Corrections, simd_lane_id, simd_group_id);

    O += c_row * params->ldo;

    // Prepare threadgroup mma operation
    thread mma_qk_t mma_qk_op(simd_group_id, simd_lane_id);
    thread mma_sv_t mma_softmax_sv_op(simd_group_id, simd_lane_id);
    thread loader_k_t loader_k(K, params->ldk, Ks, simd_group_id, simd_lane_id);
    thread loader_v_t loader_v(V, params->ldv, Vs, simd_group_id, simd_lane_id);

    for (short n_block = 0; n_block < params->gemm_n_iterations_aligned;
         n_block++) {
      short c_col = BN;

      // Prepare threadgroup loading operations
      short gemm_k_iterations = params->gemm_k_iterations_aligned;
      short tgp_bn_qk = min(BN, params->N - c_col * n_block);
      threadgroup_barrier(mem_flags::mem_none);

      ///////////////////////////////////////////////////////////////////////////////
      { // Loop over K - unaligned case

        if (tgp_bm == BM && tgp_bn_qk == BN) {
          gemm_loop<true, true, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);
        } else if (tgp_bn_qk == BN) {
          gemm_loop<false, true, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);

        } else if (tgp_bm == BM) {
          gemm_loop<true, false, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);

        } else {
          gemm_loop<false, false, K_aligned>(
              Qs,
              Ks,
              gemm_k_iterations,
              loader_k,
              mma_qk_op,
              tgp_bm,
              tgp_bn_qk);
        }
      }

      mma_qk_op.store_result_to_tgp_memory(
          Ss, BN + tgp_padding, short2(BN, BM));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      rescale_ss(
          Ss,
          Corrections,
          simd_group_id,
          simd_lane_id,
          short2(tgp_bn_qk, tgp_bm),
          params->alpha);

      loader_v.load_safe(short2(BK, tgp_bn_qk));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      threadgroup float* o_scales = Corrections + 2 * (BM + float_padding);
      mma_softmax_sv_op.rescale_output(o_scales);

      mma_softmax_sv_op.mma(Ss, Vs);

      threadgroup float* final_output_scales =
          Corrections + 3 * (BM + float_padding);

      mma_softmax_sv_op.rescale_output(final_output_scales);

      loader_v.next();
      loader_k.next(BN);

      mma_qk_op.clear_results();
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_softmax_sv_op.store_result_safe(O, params->ldo, short2(BK, tgp_bm));
  }
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_q,
    bool transpose_k,
    bool transpose_v,
    bool MN_aligned,
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant MLXFastAttentionParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using attention_kernel = FastAttentionKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_q,
      transpose_k,
      transpose_v,
      MN_aligned,
      K_aligned>;

  // Adjust for batch
  if (params->batch_ndim > 1) {
    const constant size_t* Q_bstrides = batch_strides;
    const constant size_t* KV_bstrides = batch_strides + params->batch_ndim;

    ulong2 batch_offsets = elem_to_loc_broadcast(
        tid.z, batch_shape, Q_bstrides, KV_bstrides, params->batch_ndim);

    Q += batch_offsets.x;
    K += batch_offsets.y;
    V += batch_offsets.y;

  } else {
    Q += params->batch_stride_q * tid.z;
    K += params->batch_stride_k * tid.z;
    V += params->batch_stride_v * tid.z;
  }

  // same shape as input
  O += params->batch_stride_o * tid.z;
  threadgroup T Qs[attention_kernel::tgp_mem_size_q];
  threadgroup T Ss[attention_kernel::tgp_mem_size_s];
  threadgroup float Corrections[attention_kernel::tgp_mem_size_corrections];

  if (attention_kernel::share_kv_smem) {
    threadgroup T Ks[attention_kernel::tgp_mem_size_k];
    threadgroup T* Vs = Ks; //[attention_kernel::tgp_mem_size_v];
    attention_kernel::run(
        Q,
        K,
        V,
        O,
        params,
        Qs,
        Ks,
        Ss,
        Vs,
        Corrections,
        simd_lane_id,
        simd_group_id,
        tid,
        lid);
  } else {
    threadgroup T Ks[attention_kernel::tgp_mem_size_k];
    threadgroup T Vs[attention_kernel::tgp_mem_size_v];
    attention_kernel::run(
        Q,
        K,
        V,
        O,
        params,
        Qs,
        Ks,
        Ss,
        Vs,
        Corrections,
        simd_lane_id,
        simd_group_id,
        tid,
        lid);
  }
}

// clang-format off

// SDPA full instantiations
#define instantiate_fast_inference_self_attention_kernel(                   \
    itype, otype, bm, bn, bk, wm, wn)                                       \
  template [[host_name("steel_gemm_attention_bm_" #bm "_bn_" #bn "_bk_" #bk \
                       "_itype_" #itype)]] [[kernel]] void                  \
  attention<itype, bm, bn, bk, wm, wn, false, true, false, false, true>(    \
      const device itype* Q [[buffer(0)]],                                  \
      const device itype* K [[buffer(1)]],                                  \
      const device itype* V [[buffer(2)]],                                  \
      device otype* O [[buffer(3)]],                                        \
      const constant MLXFastAttentionParams* params [[buffer(4)]],          \
      const constant int* batch_shape [[buffer(6)]],                        \
      const constant size_t* batch_strides [[buffer(7)]],                   \
      uint simd_lane_id [[thread_index_in_simdgroup]],                      \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                \
      uint3 tid [[threadgroup_position_in_grid]],                           \
      uint3 lid [[thread_position_in_threadgroup]]);

instantiate_fast_inference_self_attention_kernel(
    float,
    float,
    16,
    16,
    64,
    2,
    2);
instantiate_fast_inference_self_attention_kernel(
    float,
    float,
    16,
    16,
    128,
    2,
    2);
instantiate_fast_inference_self_attention_kernel(half, half, 16, 16, 64, 2, 2);
instantiate_fast_inference_self_attention_kernel(half, half, 16, 16, 128, 2, 2);

// SDPA vector instantiations
#define instantiate_sdpa_vector(type, head_dim)                              \
  instantiate_kernel("sdpa_vector_" #type "_" #head_dim, sdpa_vector, type, head_dim)

#define instantiate_sdpa_vector_heads(type) \
  instantiate_sdpa_vector(type, 64)         \
  instantiate_sdpa_vector(type, 96)         \
  instantiate_sdpa_vector(type, 128)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)

// Quantized SDPA vector instantiations
#define instantiate_quant_sdpa_vector(type, head_dim, group_size, bits) \
  instantiate_kernel(                                                   \
    "quant_sdpa_vector_" #type "_" #head_dim "_" #group_size "_" #bits, \
    quant_sdpa_vector, type, head_dim, group_size, bits)

#define instantiate_quant_sdpa_vector_bits(type, heads, group_size) \
  instantiate_quant_sdpa_vector(type, heads, group_size, 4)         \
  instantiate_quant_sdpa_vector(type, heads, group_size, 8)

#define instantiate_quant_sdpa_vector_group_size(type, heads) \
  instantiate_quant_sdpa_vector_bits(type, heads, 32)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 64)         \
  instantiate_quant_sdpa_vector_bits(type, heads, 128)

#define instantiate_quant_sdpa_vector_heads(type) \
  instantiate_quant_sdpa_vector_group_size(type, 64)         \
  instantiate_quant_sdpa_vector_group_size(type, 96)         \
  instantiate_quant_sdpa_vector_group_size(type, 128)


instantiate_quant_sdpa_vector_heads(float)
instantiate_quant_sdpa_vector_heads(bfloat16_t)
instantiate_quant_sdpa_vector_heads(float16_t)

    // clang-format on
