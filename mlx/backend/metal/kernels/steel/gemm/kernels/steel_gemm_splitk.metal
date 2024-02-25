// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

template <typename T,
          typename U,
          int BM,
          int BN,
          int BK,
          int WM,
          int WN,
          bool transpose_a, 
          bool transpose_b,
          bool MN_aligned,
          bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm_splitk(
    const device T *A [[buffer(0)]],
    const device T *B [[buffer(1)]],
    device U *C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { 

    (void)lid;
    
    using gemm_kernel = GEMMKernel<T, U, BM, BN, BK, WM, WN, transpose_a, transpose_b, MN_aligned, K_aligned>;
    using loader_a_t = typename gemm_kernel::loader_a_t;
    using loader_b_t = typename gemm_kernel::loader_b_t;
    using mma_t = typename gemm_kernel::mma_t;
    
    threadgroup T As[gemm_kernel::tgp_mem_size_a];
    threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

    const int tid_x = tid.x;
    const int tid_y = tid.y;
    const int tid_z = tid.z;

    if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
      return;
    }

    // Find block in A, B, C
    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const int k_start = params->split_k_partition_size * tid_z;

    A += transpose_a ? (c_row + k_start * params->lda) : (k_start + c_row * params->lda);
    B += transpose_b ? (k_start + c_col * params->ldb) : (c_col + k_start * params->ldb);
    C += (params->split_k_partition_stride * tid_z) + (c_row * params->ldc + c_col);

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    int gemm_k_iterations = params->gemm_k_iterations_aligned;

    short tgp_bm = min(BM, params->M - c_row);
    short tgp_bn = min(BN, params->N - c_col);
    short leftover_bk = params->K % BK;

    if(MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, true, true>{});
    } else if (tgp_bn == BN) {
      gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<false, true, true>{});
    } else if (tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, false, true>{});
    } else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, true>{});
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((tid_z + 1) == (params->split_k_partitions)) {
      int gemm_k_iter_remaining = (params->K - (k_start + params->split_k_partition_size)) / BK;
      if(!K_aligned || gemm_k_iter_remaining > 0)
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iter_remaining,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, K_aligned>{});
    }

    if(MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
      mma_op.store_result(C, params->ldc);
    } else {
      mma_op.store_result_safe(C, params->ldc, short2(tgp_bn, tgp_bm));
    }
}

///////////////////////////////////////////////////////////////////////////////
// GEMM kernel initializations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, aname, mn_aligned, kname, k_aligned) \
  template [[host_name("steel_gemm_splitk_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_MN_" #aname "_K_" #kname)]] \
  [[kernel]] void gemm_splitk<itype, otype, bm, bn, bk, wm, wn, trans_a, trans_b, mn_aligned, k_aligned>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      device otype *C [[buffer(2)]], \
      const constant GEMMSpiltKParams* params [[buffer(3)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_aligned_helper(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, taligned, true, naligned, false) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, taligned, true) \
  instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn, naligned, false, naligned, false)

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm_aligned_helper(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 16, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 16, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float32, float);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, float32, float);

instantiate_gemm_shapes_helper(float32, float, float32, float);

///////////////////////////////////////////////////////////////////////////////
// Split k accumulation kernel 
///////////////////////////////////////////////////////////////////////////////

template <typename AccT,
          typename OutT,
          typename Epilogue = TransformNone<OutT, AccT>>
[[kernel]] void gemm_splitk_accum(
    const device AccT *C_split [[buffer(0)]],
    device OutT *D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {

  // Ajust D and C
  D += gid.x + gid.y * ldd;
  C_split += gid.x + gid.y * ldd;

  int offset = 0;
  AccT out = 0;

  for(int i = 0; i < k_partitions; i++) {
    out += C_split[offset];
    offset += partition_stride;
  }

  // Write output 
  D[0] = Epilogue::apply(out);

}

template <typename AccT,
          typename OutT,
          typename Epilogue = TransformAxpby<OutT, AccT>>
[[kernel]] void gemm_splitk_accum_axpby(
    const device AccT *C_split [[buffer(0)]],
    device OutT *D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    const device OutT *C [[buffer(5)]],
    const constant int& ldc [[buffer(6)]],
    const constant int& fdc [[buffer(7)]],
    const constant float& alpha [[buffer(8)]],
    const constant float& beta [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]) {

  // Ajust D and C
  C += gid.x * fdc + gid.y * ldc;
  D += gid.x + gid.y * ldd;
  C_split += gid.x + gid.y * ldd;

  int offset = 0;
  AccT out = 0;

  for(int i = 0; i < k_partitions; i++) {
    out += C_split[offset];
    offset += partition_stride;
  }

  // Write output 
  Epilogue op(alpha, beta);
  D[0] = op.apply(out, *C);

}

#define instantiate_accum(oname, otype, aname, atype) \
  template [[host_name("steel_gemm_splitk_accum_" #oname "_"  #aname)]] \
  [[kernel]] void gemm_splitk_accum<atype, otype>(                                    \
      const device atype *C_split [[buffer(0)]],                         \
      device otype *D [[buffer(1)]],                                     \
      const constant int& k_partitions [[buffer(2)]],                   \
      const constant int& partition_stride [[buffer(3)]],               \
      const constant int& ldd [[buffer(4)]],                            \
      uint2 gid [[thread_position_in_grid]]);                         \
  template [[host_name("steel_gemm_splitk_accum_" #oname "_"  #aname "_axpby")]] \
  [[kernel]] void gemm_splitk_accum_axpby<atype, otype>( \
      const device atype *C_split [[buffer(0)]], \
      device otype *D [[buffer(1)]], \
      const constant int& k_partitions [[buffer(2)]], \
      const constant int& partition_stride [[buffer(3)]], \
      const constant int& ldd [[buffer(4)]], \
      const device otype *C [[buffer(5)]],  \
      const constant int& ldc [[buffer(6)]], \
      const constant int& fdc [[buffer(7)]], \
      const constant float& alpha [[buffer(8)]], \
      const constant float& beta [[buffer(9)]], \
      uint2 gid [[thread_position_in_grid]]);

instantiate_accum(bfloat16, bfloat16_t, float32, float);
instantiate_accum(float16, half, float32, float);
instantiate_accum(float32, float, float32, float);