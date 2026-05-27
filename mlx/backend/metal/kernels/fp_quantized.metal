// Copyright © 2025 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

#define instantiate_quantized(mode, name, type, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits, \
      fp_ ## name, \
      type, \
      group_size,   \
      bits)

#define instantiate_quantized_batched(mode, name, type, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      batched)

#define instantiate_quantized_aligned(mode, name, type, aligned, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      aligned)

#define instantiate_quantized_aligned_batched(mode, name, type, aligned, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      aligned, \
      batched)

#define instantiate_quantized_quad(mode, name, type, D, batched, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      D,       \
      batched)

#define instantiate_quantized_split_k(mode, name, type, split_k, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      fp_ ## name,    \
      type,    \
      group_size,      \
      bits,       \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, bm, bn, bk, wm, wn, transpose, mode, group_size, bits) \
  instantiate_kernel( \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,    \
      type,    \
      group_size,      \
      bits,       \
      bm,      \
      bn,      \
      bk,      \
      wm,      \
      wn,      \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, mode, group_size, bits) \
  instantiate_quantized_batched(mode, name, type, 1, group_size, bits)         \
  instantiate_quantized_batched(mode, name, type, 0, group_size, bits)

#define instantiate_quantized_all_batched(type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmv_fast, type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmv, type, mode, group_size, bits)      \
  instantiate_quantized_batched_wrap(qvm, type, mode, group_size, bits) \
  instantiate_quantized_batched_wrap(qmm_n, type, mode, group_size, bits)

#define instantiate_quantized_all_single(type, mode, group_size, bits) \
  instantiate_quantized(mode, gather_qmv_fast, type, group_size, bits) \
  instantiate_quantized(mode, gather_qmv, type, group_size, bits)      \
  instantiate_quantized(mode, gather_qvm, type, group_size, bits) \
  instantiate_quantized(mode, gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, true, group_size, bits)      \
  instantiate_quantized_aligned(mode, gather_qmm_t, type, false, group_size, bits)     \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, true, 1, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, true, 0, group_size, bits)  \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, false, 1, group_size, bits) \
  instantiate_quantized_aligned_batched(mode, qmm_t, type, false, 0, group_size, bits)

#define instantiate_quantized_all_quad(type, mode, group_size, bits) \
  instantiate_quantized_quad(mode, qmv_quad, type, 64, 1, group_size, bits)  \
  instantiate_quantized_quad(mode, qmv_quad, type, 64, 0, group_size, bits)  \
  instantiate_quantized_quad(mode, qmv_quad, type, 128, 1, group_size, bits) \
  instantiate_quantized_quad(mode, qmv_quad, type, 128, 0, group_size, bits)

#define instantiate_quantized_all_splitk(type, mode, group_size, bits) \
  instantiate_quantized_split_k(mode, qvm_split_k, type, 8, group_size, bits) \
  instantiate_quantized_split_k(mode, qvm_split_k, type, 32, group_size, bits) \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, true, group_size, bits) \
  instantiate_quantized_aligned(mode, qmm_t_splitk, type, false, group_size, bits)

#define instantiate_quantized_all_rhs(type, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, gather_qmm_rhs_nt, type, 16, 32, 32, 1, 2, true, mode, group_size, bits) \
  instantiate_gather_qmm_rhs(fp_gather_qmm_rhs, gather_qmm_rhs_nn, type, 16, 32, 32, 1, 2, false, mode, group_size, bits)

#define instantiate_quantize_dequantize(type, mode, group_size, bits) \
  instantiate_kernel( \
    #mode "_quantize_dequantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_quantize_dequantize, \
    type, \
    group_size,  \
    bits) \
  instantiate_kernel( \
    #mode "_quantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_quantize, \
    type, \
    group_size,  \
    bits) \
  instantiate_kernel( \
    #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits, \
    fp_dequantize, \
    type, \
    group_size,  \
    bits)

#define instantiate_quantized_modes(type, mode, group_size, bits) \
  instantiate_quantized_all_batched(type, mode, group_size, bits) \
  instantiate_quantized_all_single(type, mode, group_size, bits)  \
  instantiate_quantized_all_quad(type, mode, group_size, bits)    \
  instantiate_quantized_all_splitk(type, mode, group_size, bits)  \
  instantiate_quantized_all_aligned(type, mode, group_size, bits) \
  instantiate_quantized_all_rhs(type, mode, group_size, bits)     \
  instantiate_quantize_dequantize(type, mode, group_size, bits)

#define instantiate_quantized_types(type) \
  instantiate_quantized_modes(type, nvfp4, 16, 4) \
  instantiate_quantized_modes(type, mxfp8, 32, 8) \
  instantiate_quantized_modes(type, mxfp4, 32, 4)

instantiate_quantized_types(float)
instantiate_quantized_types(bfloat16_t)
instantiate_quantized_types(float16_t)

// ----- block_fp8 (DeepSeek-V3 / MiMo) instantiations -----
// Only qmv_fast is implemented in the first pass. qmv_quad, qmv, qvm, qmm,
// gather variants all produce "kernel not found" until subsequent patches.
#define instantiate_block_fp8_qmv_fast(type) \
  instantiate_kernel( \
      "block_fp8_qmv_fast_" #type "_gs_128_b_8_batch_0", \
      block_fp8_qmv_fast, type, 128, 8, false) \
  instantiate_kernel( \
      "block_fp8_qmv_fast_" #type "_gs_128_b_8_batch_1", \
      block_fp8_qmv_fast, type, 128, 8, true)

instantiate_block_fp8_qmv_fast(float)
instantiate_block_fp8_qmv_fast(bfloat16_t)
instantiate_block_fp8_qmv_fast(float16_t)

#define instantiate_block_fp8_gather_qmv_fast(type) \
  instantiate_kernel( \
      "block_fp8_gather_qmv_fast_" #type "_gs_128_b_8", \
      block_fp8_gather_qmv_fast, type, 128, 8)

instantiate_block_fp8_gather_qmv_fast(float)
instantiate_block_fp8_gather_qmv_fast(bfloat16_t)
instantiate_block_fp8_gather_qmv_fast(float16_t)

// qmm_t: prefill matmul. Per-row qmv math, 8 N rows per threadgroup.
// 4 variants per type: (aligned_N x batched). Dispatcher selects the right
// suffix based on N%32 and batch size.
#define instantiate_block_fp8_qmm_t(type) \
  instantiate_kernel( \
      "block_fp8_qmm_t_" #type "_gs_128_b_8_alN_true_batch_0", \
      block_fp8_qmm_t, type, 128, 8, true, false) \
  instantiate_kernel( \
      "block_fp8_qmm_t_" #type "_gs_128_b_8_alN_true_batch_1", \
      block_fp8_qmm_t, type, 128, 8, true, true) \
  instantiate_kernel( \
      "block_fp8_qmm_t_" #type "_gs_128_b_8_alN_false_batch_0", \
      block_fp8_qmm_t, type, 128, 8, false, false) \
  instantiate_kernel( \
      "block_fp8_qmm_t_" #type "_gs_128_b_8_alN_false_batch_1", \
      block_fp8_qmm_t, type, 128, 8, false, true)

instantiate_block_fp8_qmm_t(float)
instantiate_block_fp8_qmm_t(bfloat16_t)
instantiate_block_fp8_qmm_t(float16_t)

#define instantiate_block_fp8_gather_qmm_t(type) \
  template [[host_name("block_fp8_gather_qmm_t_" #type "_gs_128_b_8_alN_true")]] \
  [[kernel]] void block_fp8_gather_qmm_t<type, 128, 8, true>( \
      const device uint8_t* w, const device float* scales, \
      const device uint32_t* rhs_indices, const device type* x, device type* y, \
      const constant int& K, const constant int& N, const constant int& M, \
      const constant int& x_batch_ndims, const constant int* x_shape, \
      const constant int64_t* x_strides, const constant int& w_batch_ndims, \
      const constant int* w_shape, const constant int64_t* w_strides, \
      const constant int64_t* s_strides, const constant int& batch_ndims, \
      const constant int* batch_shape, const constant int64_t* lhs_strides, \
      const constant int64_t* rhs_strides, \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]); \
  template [[host_name("block_fp8_gather_qmm_t_" #type "_gs_128_b_8_alN_false")]] \
  [[kernel]] void block_fp8_gather_qmm_t<type, 128, 8, false>( \
      const device uint8_t* w, const device float* scales, \
      const device uint32_t* rhs_indices, const device type* x, device type* y, \
      const constant int& K, const constant int& N, const constant int& M, \
      const constant int& x_batch_ndims, const constant int* x_shape, \
      const constant int64_t* x_strides, const constant int& w_batch_ndims, \
      const constant int* w_shape, const constant int64_t* w_strides, \
      const constant int64_t* s_strides, const constant int& batch_ndims, \
      const constant int* batch_shape, const constant int64_t* lhs_strides, \
      const constant int64_t* rhs_strides, \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint simd_gid [[simdgroup_index_in_threadgroup]], \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_block_fp8_gather_qmm_t(float)
instantiate_block_fp8_gather_qmm_t(bfloat16_t)
instantiate_block_fp8_gather_qmm_t(float16_t)
    // clang-format on
