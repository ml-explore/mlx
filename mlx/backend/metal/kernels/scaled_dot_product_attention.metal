#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/transforms.h"
#include "mlx/backend/metal/kernels/steel/utils.h"

#include "mlx/backend/metal/kernels/scaled_dot_product_attention_params.h"
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

template <
    typename T,
    typename T2,
    typename T4,
    uint16_t TILE_SIZE_CONST,
    uint16_t NSIMDGROUPS>
[[kernel]] void fast_inference_sdpa_compute_partials_template(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device uint64_t& L [[buffer(3)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(4)]],
    device float* O_partials [[buffer(5)]],
    device float* p_lse [[buffer(6)]],
    device float* p_maxes [[buffer(7)]],
    threadgroup T* threadgroup_block [[threadgroup(0)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr const size_t DK = 128;
  constexpr const ulong SIMDGROUP_MATRIX_LOAD_FACTOR = 8;
  constexpr const size_t THREADS_PER_SIMDGROUP = 32;
  constexpr const uint iter_offset = NSIMDGROUPS * 4;
  const bool is_gqa = params.N_KV_HEADS != params.N_Q_HEADS;
  uint kv_head_offset_factor = tid.x;
  if (is_gqa) {
    int q_kv_head_ratio = params.N_Q_HEADS / params.N_KV_HEADS;
    kv_head_offset_factor = tid.x / q_kv_head_ratio;
  }
  constexpr const uint16_t P_VEC4 = TILE_SIZE_CONST / NSIMDGROUPS / 4;
  constexpr const size_t MATRIX_LOADS_PER_SIMDGROUP =
      TILE_SIZE_CONST / (SIMDGROUP_MATRIX_LOAD_FACTOR * NSIMDGROUPS);
  constexpr const size_t MATRIX_COLS = DK / SIMDGROUP_MATRIX_LOAD_FACTOR;
  constexpr const uint totalSmemV = SIMDGROUP_MATRIX_LOAD_FACTOR *
      SIMDGROUP_MATRIX_LOAD_FACTOR * (MATRIX_LOADS_PER_SIMDGROUP + 1) *
      NSIMDGROUPS;

  threadgroup T4* smemFlush = (threadgroup T4*)threadgroup_block;
#pragma clang loop unroll(full)
  for (uint i = 0; i < 8; i++) {
    smemFlush
        [simd_lane_id + simd_group_id * THREADS_PER_SIMDGROUP +
         i * NSIMDGROUPS * THREADS_PER_SIMDGROUP] = T4(0.f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // TODO: multiple query sequence length for speculative decoding
  const uint tgroup_query_head_offset =
      tid.x * DK + tid.z * (params.N_Q_HEADS * DK);

  const uint tgroup_k_head_offset = kv_head_offset_factor * DK * L;
  const uint tgroup_k_tile_offset = tid.y * TILE_SIZE_CONST * DK;
  const uint tgroup_k_batch_offset = tid.z * L * params.N_KV_HEADS * DK;

  const device T* baseK =
      K + tgroup_k_batch_offset + tgroup_k_tile_offset + tgroup_k_head_offset;
  const device T* baseQ = Q + tgroup_query_head_offset;

  device T4* simdgroupQueryData = (device T4*)baseQ;

  constexpr const size_t ACCUM_PER_GROUP = TILE_SIZE_CONST / NSIMDGROUPS;
  float threadAccum[ACCUM_PER_GROUP];

#pragma clang loop unroll(full)
  for (size_t threadAccumIndex = 0; threadAccumIndex < ACCUM_PER_GROUP;
       threadAccumIndex++) {
    threadAccum[threadAccumIndex] = -INFINITY;
  }

  uint KROW_ACCUM_INDEX = 0;

  const int32_t SEQUENCE_LENGTH_LESS_TILE_SIZE = L - TILE_SIZE_CONST;
  const bool LAST_TILE = (tid.y + 1) * TILE_SIZE_CONST >= L;
  const bool LAST_TILE_ALIGNED =
      (SEQUENCE_LENGTH_LESS_TILE_SIZE == int32_t(tid.y * TILE_SIZE_CONST));

  T4 thread_data_x4;
  T4 thread_data_y4;
  if (!LAST_TILE || LAST_TILE_ALIGNED) {
    thread_data_x4 = *(simdgroupQueryData + simd_lane_id);
#pragma clang loop unroll(full)
    for (size_t KROW = simd_group_id; KROW < TILE_SIZE_CONST;
         KROW += NSIMDGROUPS) {
      const uint KROW_OFFSET = KROW * DK;
      const device T* baseKRow = baseK + KROW_OFFSET;
      device T4* keysData = (device T4*)baseKRow;
      thread_data_y4 = *(keysData + simd_lane_id);
      T kq_scalar = dot(thread_data_x4, thread_data_y4);
      threadAccum[KROW_ACCUM_INDEX] = float(kq_scalar);
      KROW_ACCUM_INDEX++;
    }
  } else {
    thread_data_x4 = *(simdgroupQueryData + simd_lane_id);
    const uint START_ROW = tid.y * TILE_SIZE_CONST;
    const device T* baseKThisHead =
        K + tgroup_k_batch_offset + tgroup_k_head_offset;

    for (size_t KROW = START_ROW + simd_group_id; KROW < L;
         KROW += NSIMDGROUPS) {
      const uint KROW_OFFSET = KROW * DK;
      const device T* baseKRow = baseKThisHead + KROW_OFFSET;
      device T4* keysData = (device T4*)baseKRow;
      thread_data_y4 = *(keysData + simd_lane_id);
      T kq_scalar = dot(thread_data_x4, thread_data_y4);
      threadAccum[KROW_ACCUM_INDEX] = float(kq_scalar);
      KROW_ACCUM_INDEX++;
    }
  }
  threadgroup float* smemP = (threadgroup float*)threadgroup_block;

#pragma clang loop unroll(full)
  for (size_t i = 0; i < P_VEC4; i++) {
    thread_data_x4 =
        T4(threadAccum[4 * i],
           threadAccum[4 * i + 1],
           threadAccum[4 * i + 2],
           threadAccum[4 * i + 3]);
    simdgroup_barrier(mem_flags::mem_none);
    thread_data_y4 = simd_sum(thread_data_x4);
    if (simd_lane_id == 0) {
      const uint base_smem_p_offset = i * iter_offset + simd_group_id;
      smemP[base_smem_p_offset + NSIMDGROUPS * 0] = float(thread_data_y4.x);
      smemP[base_smem_p_offset + NSIMDGROUPS * 1] = float(thread_data_y4.y);
      smemP[base_smem_p_offset + NSIMDGROUPS * 2] = float(thread_data_y4.z);
      smemP[base_smem_p_offset + NSIMDGROUPS * 3] = float(thread_data_y4.w);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float groupMax;
  float lse = 0.f;

  constexpr const size_t THREADS_PER_THREADGROUP_TIMES_4 = 4 * 32;
  constexpr const size_t ACCUM_ARRAY_LENGTH =
      TILE_SIZE_CONST / THREADS_PER_THREADGROUP_TIMES_4 + 1;
  float4 pvals[ACCUM_ARRAY_LENGTH];

#pragma clang loop unroll(full)
  for (uint accum_array_iter = 0; accum_array_iter < ACCUM_ARRAY_LENGTH;
       accum_array_iter++) {
    pvals[accum_array_iter] = float4(-INFINITY);
  }

  if (TILE_SIZE_CONST == 64) {
    threadgroup float2* smemPtrFlt2 = (threadgroup float2*)threadgroup_block;
    float2 vals = smemPtrFlt2[simd_lane_id];
    vals *= params.INV_ALPHA;
    float maxval = max(vals.x, vals.y);
    simdgroup_barrier(mem_flags::mem_none);
    groupMax = simd_max(maxval);

    float2 expf_shifted = exp(vals - groupMax);
    float sumExpLocal = expf_shifted.x + expf_shifted.y;
    simdgroup_barrier(mem_flags::mem_none);
    float tgroupExpSum = simd_sum(sumExpLocal);

    lse = log(tgroupExpSum);
    float2 local_p_hat = expf_shifted / tgroupExpSum;
    pvals[0].x = local_p_hat.x;
    pvals[0].y = local_p_hat.y;
    smemPtrFlt2[simd_lane_id] = float2(0.f);
  }
  constexpr const bool TILE_SIZE_LARGER_THAN_64 = TILE_SIZE_CONST > 64;
  constexpr const int TILE_SIZE_ITERS_128 = TILE_SIZE_CONST / 128;

  if (TILE_SIZE_LARGER_THAN_64) {
    float maxval = -INFINITY;
    threadgroup float4* smemPtrFlt4 = (threadgroup float4*)threadgroup_block;
#pragma clang loop unroll(full)
    for (int i = 0; i < TILE_SIZE_ITERS_128; i++) {
      float4 vals = smemPtrFlt4[simd_lane_id + i * THREADS_PER_SIMDGROUP];
      vals *= params.INV_ALPHA;
      pvals[i] = vals;
      maxval = fmax3(vals.x, vals.y, maxval);
      maxval = fmax3(vals.z, vals.w, maxval);
    }
    simdgroup_barrier(mem_flags::mem_none);
    groupMax = simd_max(maxval);

    float sumExpLocal = 0.f;
#pragma clang loop unroll(full)
    for (int i = 0; i < TILE_SIZE_ITERS_128; i++) {
      pvals[i] = exp(pvals[i] - groupMax);
      sumExpLocal += pvals[i].x + pvals[i].y + pvals[i].z + pvals[i].w;
    }
    simdgroup_barrier(mem_flags::mem_none);
    float tgroupExpSum = simd_sum(sumExpLocal);
    lse = log(tgroupExpSum);
#pragma clang loop unroll(full)
    for (int i = 0; i < TILE_SIZE_ITERS_128; i++) {
      pvals[i] = pvals[i] / tgroupExpSum;
      smemPtrFlt4[simd_lane_id + i * THREADS_PER_SIMDGROUP] = float4(0.f);
    }
  }

  threadgroup T* smemV = (threadgroup T*)threadgroup_block;

  const size_t v_batch_offset = tid.z * params.N_KV_HEADS * L * DK;
  const size_t v_head_offset = kv_head_offset_factor * L * DK;

  const size_t v_tile_offset = tid.y * TILE_SIZE_CONST * DK;
  const size_t v_offset = v_batch_offset + v_head_offset + v_tile_offset;
  device T* baseV = (device T*)V + v_offset;

  threadgroup float* smemOpartial = (threadgroup float*)(smemV + totalSmemV);

  if (!LAST_TILE || LAST_TILE_ALIGNED) {
#pragma clang loop unroll(full)
    for (size_t col = 0; col < MATRIX_COLS; col++) {
      uint matrix_load_loop_iter = 0;
      constexpr const size_t TILE_SIZE_CONST_DIV_8 = TILE_SIZE_CONST / 8;

      for (size_t tile_start = simd_group_id;
           tile_start < TILE_SIZE_CONST_DIV_8;
           tile_start += NSIMDGROUPS) {
        simdgroup_matrix<T, 8, 8> tmp;
        ulong simdgroup_matrix_offset =
            matrix_load_loop_iter * NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR +
            simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR;
        ulong2 matrixOrigin =
            ulong2(col * SIMDGROUP_MATRIX_LOAD_FACTOR, simdgroup_matrix_offset);
        simdgroup_load(tmp, baseV, DK, matrixOrigin, true);
        const ulong2 matrixOriginSmem = ulong2(simdgroup_matrix_offset, 0);
        const ulong elemsPerRowSmem = TILE_SIZE_CONST;
        simdgroup_store(tmp, smemV, elemsPerRowSmem, matrixOriginSmem, false);
        matrix_load_loop_iter++;
      };

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (TILE_SIZE_CONST == 64) {
        T2 local_p_hat = T2(pvals[0].x, pvals[0].y);
        uint loop_iter = 0;
        threadgroup float* oPartialSmem =
            smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;

#pragma clang loop unroll(full)
        for (size_t row = simd_group_id; row < SIMDGROUP_MATRIX_LOAD_FACTOR;
             row += NSIMDGROUPS) {
          threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row);
          threadgroup T2* smemV2 = (threadgroup T2*)smemV_row;
          T2 v_local = *(smemV2 + simd_lane_id);

          T val = dot(local_p_hat, v_local);
          simdgroup_barrier(mem_flags::mem_none);

          T row_sum = simd_sum(val);
          oPartialSmem[simd_group_id + loop_iter * NSIMDGROUPS] =
              float(row_sum);
          loop_iter++;
        }
      }

      if (TILE_SIZE_CONST > 64) {
        constexpr const size_t TILE_SIZE_CONST_DIV_128 =
            (TILE_SIZE_CONST + 1) / 128;
        threadgroup float* oPartialSmem =
            smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
        uint loop_iter = 0;
        for (size_t row = simd_group_id; row < SIMDGROUP_MATRIX_LOAD_FACTOR;
             row += NSIMDGROUPS) {
          threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row);

          T row_sum = 0.f;
          for (size_t i = 0; i < TILE_SIZE_CONST_DIV_128; i++) {
            threadgroup T4* smemV2 = (threadgroup T4*)smemV_row;
            T4 v_local = *(smemV2 + simd_lane_id + i * THREADS_PER_SIMDGROUP);
            T4 p_local = T4(pvals[i]);
            T val = dot(p_local, v_local);
            row_sum += val;
          }
          simdgroup_barrier(mem_flags::mem_none);
          row_sum = simd_sum(row_sum);
          oPartialSmem[simd_group_id + loop_iter * NSIMDGROUPS] =
              float(row_sum);
          loop_iter++;
        }
      }
    }
  } else {
    const int32_t START_ROW = tid.y * TILE_SIZE_CONST;
    const int32_t MAX_START_ROW = L - SIMDGROUP_MATRIX_LOAD_FACTOR + 1;
    const device T* baseVThisHead = V + v_batch_offset + v_head_offset;
    constexpr const int ROWS_PER_ITER = 8;
#pragma clang loop unroll(full)
    for (size_t col = 0; col < MATRIX_COLS; col++) {
      uint smem_col_index = simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR;
      int32_t tile_start;
      for (tile_start =
               START_ROW + simd_group_id * SIMDGROUP_MATRIX_LOAD_FACTOR;
           tile_start < MAX_START_ROW;
           tile_start += NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR) {
        simdgroup_matrix<T, 8, 8> tmp;
        ulong2 matrixOrigin =
            ulong2(col * SIMDGROUP_MATRIX_LOAD_FACTOR, tile_start);
        simdgroup_load(
            tmp, baseVThisHead, DK, matrixOrigin, /* transpose */ true);
        const ulong2 matrixOriginSmem = ulong2(smem_col_index, 0);
        constexpr const ulong elemsPerRowSmem = TILE_SIZE_CONST;
        simdgroup_store(
            tmp,
            smemV,
            elemsPerRowSmem,
            matrixOriginSmem,
            /* transpose */ false);
        smem_col_index += NSIMDGROUPS * SIMDGROUP_MATRIX_LOAD_FACTOR;
      };

      tile_start =
          ((L / SIMDGROUP_MATRIX_LOAD_FACTOR) * SIMDGROUP_MATRIX_LOAD_FACTOR);

      const int32_t INT_L = int32_t(L);
      for (int row_index = tile_start + simd_group_id; row_index < INT_L;
           row_index += NSIMDGROUPS) {
        if (simd_lane_id < SIMDGROUP_MATRIX_LOAD_FACTOR) {
          const uint elems_per_row_gmem = DK;
          const uint col_index_v_gmem =
              col * SIMDGROUP_MATRIX_LOAD_FACTOR + simd_lane_id;
          const uint row_index_v_gmem = row_index;

          const uint elems_per_row_smem = TILE_SIZE_CONST;
          const uint col_index_v_smem = row_index % TILE_SIZE_CONST;
          const uint row_index_v_smem = simd_lane_id;

          const uint scalar_offset_gmem =
              row_index_v_gmem * elems_per_row_gmem + col_index_v_gmem;
          const uint scalar_offset_smem =
              row_index_v_smem * elems_per_row_smem + col_index_v_smem;
          T vdata = T(*(baseVThisHead + scalar_offset_gmem));
          smemV[scalar_offset_smem] = vdata;
          smem_col_index += NSIMDGROUPS;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (TILE_SIZE_CONST == 64) {
        T2 local_p_hat = T2(pvals[0].x, pvals[0].y);
        threadgroup float* oPartialSmem =
            smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
        for (size_t smem_row_index = simd_group_id;
             smem_row_index < ROWS_PER_ITER;
             smem_row_index += NSIMDGROUPS) {
          threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * smem_row_index);
          threadgroup T2* smemV2 = (threadgroup T2*)smemV_row;
          T2 v_local = *(smemV2 + simd_lane_id);
          T val = dot(local_p_hat, v_local);
          simdgroup_barrier(mem_flags::mem_none);
          T row_sum = simd_sum(val);
          oPartialSmem[smem_row_index] = float(row_sum);
        }
      }

      if (TILE_SIZE_CONST > 64) {
        threadgroup float* oPartialSmem =
            smemOpartial + SIMDGROUP_MATRIX_LOAD_FACTOR * col;
        uint loop_count = 0;
        for (size_t row_index = simd_group_id; row_index < ROWS_PER_ITER;
             row_index += NSIMDGROUPS) {
          T row_sum = 0.f;
          for (size_t tile_iters = 0; tile_iters < TILE_SIZE_ITERS_128;
               tile_iters++) {
            threadgroup T* smemV_row = smemV + (TILE_SIZE_CONST * row_index);
            threadgroup T4* smemV2 = (threadgroup T4*)smemV_row;
            T4 v_local =
                *(smemV2 + simd_lane_id + tile_iters * THREADS_PER_SIMDGROUP);
            T4 p_local = T4(pvals[tile_iters]);
            row_sum += dot(p_local, v_local);
          }
          simdgroup_barrier(mem_flags::mem_none);
          row_sum = simd_sum(row_sum);
          oPartialSmem[simd_group_id + NSIMDGROUPS * loop_count] =
              float(row_sum);
          loop_count++;
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    threadgroup float4* oPartialVec4 = (threadgroup float4*)smemOpartial;
    float4 vals = *(oPartialVec4 + simd_lane_id);
    device float* oPartialGmem =
        O_partials + tid.x * DK * params.KV_TILES + tid.y * DK;
    device float4* oPartialGmemVec4 = (device float4*)oPartialGmem;
    oPartialGmemVec4[simd_lane_id] = vals;
  }

  if (simd_group_id == 0 && simd_lane_id == 0) {
    const uint tileIndex = tid.y;
    const uint gmem_partial_scalar_offset =
        tid.z * params.N_Q_HEADS * params.KV_TILES + tid.x * params.KV_TILES +
        tileIndex;
    p_lse[gmem_partial_scalar_offset] = lse;
    p_maxes[gmem_partial_scalar_offset] = groupMax;
  }
}

#define instantiate_fast_inference_sdpa_to_partials_kernel(                  \
    itype, itype2, itype4, tile_size, nsimdgroups)                           \
  template [[host_name("fast_inference_sdpa_compute_partials_" #itype        \
                       "_" #tile_size "_" #nsimdgroups)]] [[kernel]] void    \
  fast_inference_sdpa_compute_partials_template<                             \
      itype,                                                                 \
      itype2,                                                                \
      itype4,                                                                \
      tile_size,                                                             \
      nsimdgroups>(                                                          \
      const device itype* Q [[buffer(0)]],                                   \
      const device itype* K [[buffer(1)]],                                   \
      const device itype* V [[buffer(2)]],                                   \
      const device uint64_t& L [[buffer(3)]],                                \
      const device MLXScaledDotProductAttentionParams& params [[buffer(4)]], \
      device float* O_partials [[buffer(5)]],                                \
      device float* p_lse [[buffer(6)]],                                     \
      device float* p_maxes [[buffer(7)]],                                   \
      threadgroup itype* threadgroup_block [[threadgroup(0)]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],                       \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                 \
      uint3 tid [[threadgroup_position_in_grid]]);

// clang-format off
#define instantiate_fast_inference_sdpa_to_partials_shapes_helper( \
    itype, itype2, itype4, tile_size)                              \
  instantiate_fast_inference_sdpa_to_partials_kernel(              \
      itype, itype2, itype4, tile_size, 4)                         \
  instantiate_fast_inference_sdpa_to_partials_kernel(              \
      itype, itype2, itype4, tile_size, 8) // clang-format on

instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    float,
    float2,
    float4,
    64);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    float,
    float2,
    float4,
    128);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    float,
    float2,
    float4,
    256);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    float,
    float2,
    float4,
    512);

instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    half,
    half2,
    half4,
    64);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    half,
    half2,
    half4,
    128);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    half,
    half2,
    half4,
    256);
instantiate_fast_inference_sdpa_to_partials_shapes_helper(
    half,
    half2,
    half4,
    512);

template <typename T>
void fast_inference_sdpa_reduce_tiles_template(
    const device float* O_partials [[buffer(0)]],
    const device float* p_lse [[buffer(1)]],
    const device float* p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device T* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  constexpr const int DK = 128;
  const ulong offset_rows =
      tid.z * params.KV_TILES * params.N_Q_HEADS + tid.x * params.KV_TILES;
  const device float* p_lse_row = p_lse + offset_rows;
  const device float* p_rowmax_row = p_maxes + offset_rows;
  // reserve some number of registers.  this constitutes an assumption on max
  // value of KV TILES.
  constexpr const uint8_t reserve = 128;
  float p_lse_regs[reserve];
  float p_rowmax_regs[reserve];
  float weights[reserve];

  float true_max = -INFINITY;
  for (size_t i = 0; i < params.KV_TILES; i++) {
    p_lse_regs[i] = float(*(p_lse_row + i));
    p_rowmax_regs[i] = float(*(p_rowmax_row + i));
    true_max = fmax(p_rowmax_regs[i], true_max);
    weights[i] = exp(p_lse_regs[i]);
  }

  float denom = 0.f;
  for (size_t i = 0; i < params.KV_TILES; i++) {
    weights[i] *= exp(p_rowmax_regs[i] - true_max);
    denom += weights[i];
  }

  const device float* O_partials_with_offset = O_partials +
      tid.z * params.N_Q_HEADS * DK * params.KV_TILES +
      tid.x * DK * params.KV_TILES;

  float o_value = 0.f;
  for (size_t i = 0; i < params.KV_TILES; i++) {
    float val = *(O_partials_with_offset + i * DK + lid.x);
    o_value += val * weights[i] / denom;
  }
  device T* O_gmem = O + tid.z * params.N_Q_HEADS * DK + tid.x * DK;
  O_gmem[lid.x] = T(o_value);
  return;
}

kernel void fast_inference_sdpa_reduce_tiles_float(
    const device float* O_partials [[buffer(0)]],
    const device float* p_lse [[buffer(1)]],
    const device float* p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device float* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  fast_inference_sdpa_reduce_tiles_template<float>(
      O_partials, p_lse, p_maxes, params, O, tid, lid);
}

kernel void fast_inference_sdpa_reduce_tiles_half(
    const device float* O_partials [[buffer(0)]],
    const device float* p_lse [[buffer(1)]],
    const device float* p_maxes [[buffer(2)]],
    const device MLXScaledDotProductAttentionParams& params [[buffer(3)]],
    device half* O [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  fast_inference_sdpa_reduce_tiles_template<half>(
      O_partials, p_lse, p_maxes, params, O, tid, lid);
}
