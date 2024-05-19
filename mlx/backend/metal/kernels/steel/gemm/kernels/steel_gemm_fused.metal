// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"

#define instantiate_gemm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  template [[host_name("steel_gemm_fused_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]] \
  [[kernel]] void gemm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, float>( \
      const device itype *A [[buffer(0)]], \
      const device itype *B [[buffer(1)]], \
      const device itype *C [[buffer(2), function_constant(use_out_source)]], \
      device itype *D [[buffer(3)]], \
      const constant GEMMParams* params [[buffer(4)]], \
      const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]], \
      const constant int* batch_shape [[buffer(6)]], \
      const constant size_t* batch_strides [[buffer(7)]], \
      const constant uint32_t* lhs_indices [[buffer(10), function_constant(do_gather)]], \
      const constant uint32_t* rhs_indices [[buffer(11), function_constant(do_gather)]], \
      const constant uint32_t* C_indices [[buffer(12), function_constant(gather_bias)]], \
      const constant int* operand_shape [[buffer(13), function_constant(do_gather)]], \
      const constant size_t* operand_strides [[buffer(14), function_constant(do_gather)]], \
      const constant packed_int3& operand_batch_ndim [[buffer(15), function_constant(do_gather)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]]);

#define instantiate_gemm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
    instantiate_gemm(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 16, 2, 2) \
    instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 64, 16, 2, 2)

instantiate_gemm_shapes_helper(float16, half, float16, half);
instantiate_gemm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);

instantiate_gemm_shapes_helper(float32, float, float32, float);
// clang-format on
