// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/gemv_masked.h"

#define instantiate_gemv_helper(                                           \
    outm_n, outm_t, opm_n, opm_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  template [[host_name("gemv_outmask_" #outm_n "_opmask_" #opm_n "_" #name \
                       "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm   \
                       "_tn" #tn "_nc" #nc)]] [[kernel]] void              \
  gemv_masked<itype, outm_t, opm_t, bm, bn, sm, sn, tm, tn, nc>(           \
      const device itype* mat [[buffer(0)]],                               \
      const device itype* in_vec [[buffer(1)]],                            \
      device itype* out_vec [[buffer(3)]],                                 \
      const constant int& in_vec_size [[buffer(4)]],                       \
      const constant int& out_vec_size [[buffer(5)]],                      \
      const constant int& marix_ld [[buffer(6)]],                          \
      const constant int& batch_ndim [[buffer(9)]],                        \
      const constant int* batch_shape [[buffer(10)]],                      \
      const constant size_t* vector_batch_stride [[buffer(11)]],           \
      const constant size_t* matrix_batch_stride [[buffer(12)]],           \
      const device outm_t* out_mask [[buffer(20)]],                        \
      const device opm_t* mat_mask [[buffer(21)]],                         \
      const device opm_t* vec_mask [[buffer(22)]],                         \
      const constant int* mask_strides [[buffer(23)]],                     \
      const constant size_t* mask_batch_strides [[buffer(24)]],            \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint3 lid [[thread_position_in_threadgroup]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                    \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(bool_, bool, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(name, itype, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(bool_, bool, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(name, itype, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_helper(nomask, nomask_t, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(nomask, nomask_t, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(bool_, bool, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_helper(name, itype, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)   \
  instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, 0) \
  instantiate_gemv_base(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_blocks(name, itype) \
  instantiate_gemv(name, itype, 2, 1, 4,  8, 1, 4) \
  instantiate_gemv(name, itype, 2, 1, 4,  8, 4, 4) \
  instantiate_gemv(name, itype, 2, 1, 2, 16, 1, 4) \
  instantiate_gemv(name, itype, 2, 1, 2, 16, 4, 4) \
  instantiate_gemv(name, itype, 4, 1, 2, 16, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_t_helper(                                           \
    outm_n, outm_t, opm_n, opm_t, name, itype, bm, bn, sm, sn, tm, tn, nc)   \
  template [[host_name("gemv_t_outmask_" #outm_n "_opmask_" #opm_n "_" #name \
                       "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm     \
                       "_tn" #tn "_nc" #nc)]] [[kernel]] void                \
  gemv_t_masked<itype, outm_t, opm_t, bm, bn, sm, sn, tm, tn, nc>(           \
      const device itype* mat [[buffer(0)]],                                 \
      const device itype* in_vec [[buffer(1)]],                              \
      device itype* out_vec [[buffer(3)]],                                   \
      const constant int& in_vec_size [[buffer(4)]],                         \
      const constant int& out_vec_size [[buffer(5)]],                        \
      const constant int& marix_ld [[buffer(6)]],                            \
      const constant int& batch_ndim [[buffer(9)]],                          \
      const constant int* batch_shape [[buffer(10)]],                        \
      const constant size_t* vector_batch_stride [[buffer(11)]],             \
      const constant size_t* matrix_batch_stride [[buffer(12)]],             \
      const device outm_t* out_mask [[buffer(20)]],                          \
      const device opm_t* mat_mask [[buffer(21)]],                           \
      const device opm_t* vec_mask [[buffer(22)]],                           \
      const constant int* mask_strides [[buffer(23)]],                       \
      const constant size_t* mask_batch_strides [[buffer(24)]],              \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                      \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(bool_, bool, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(name, itype, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(bool_, bool, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(name, itype, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc)      \
  instantiate_gemv_t_helper(nomask, nomask_t, name, itype, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(nomask, nomask_t, bool_, bool, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(bool_, bool, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_gemv_t_helper(name, itype, nomask, nomask_t, name, itype, bm, bn, sm, sn, tm, tn, nc)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)   \
  instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, 0) \
  instantiate_gemv_t_base(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_t_blocks(name, itype) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 8, 1) \
  instantiate_gemv_t(name, itype, 1, 1,  8, 4, 8, 4) \
  instantiate_gemv_t(name, itype, 1, 2,  8, 4, 8, 4) \
  instantiate_gemv_t(name, itype, 1, 4,  8, 4, 8, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t); // clang-format on
