// Copyright © 2023-2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits,                    \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)     \
  instantiate_kernel(                                                    \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched, \
      name,                                                              \
      type,                                                              \
      group_size,                                                        \
      bits,                                                              \
      batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched)     \
  instantiate_kernel(                                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched, \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched)

// Like instantiate_quantized_aligned_batched, but also pins BM/BK/BN (the
// qmm_t tile size), for kernels that need a non-default tile ahead-of-time
// compiled. Name must match quantized.cpp's qmm() kname exactly ("_bmX_bnY"
// suffix, no separate "_bk" component since BK is always 32 here).
#define instantiate_quantized_aligned_batched_tile(                          \
    name, type, group_size, bits, aligned, batched, bm, bk, bn)              \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned        \
            "_batch_" #batched "_bm" #bm "_bn" #bn,                          \
      name,                                                                  \
      type,                                                                  \
      group_size,                                                            \
      bits,                                                                  \
      aligned,                                                               \
      batched,                                                               \
      bm,                                                                    \
      bk,                                                                    \
      bn)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      D,                                                            \
      batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched)               \
  instantiate_kernel(                                                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched,   \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      vecs_per_tg,                                                  \
      k_lanes,                                                      \
      batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)     \
  instantiate_kernel(                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_spk_" #split_k, \
      name,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      split_k)

#define instantiate_gather_qmm_rhs(func, name, type, group_size, bits, bm, bn, bk, wm, wn, transpose)        \
  instantiate_kernel(                                                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      func,                                                         \
      type,                                                         \
      group_size,                                                   \
      bits,                                                         \
      bm,                                                           \
      bn,                                                           \
      bk,                                                           \
      wm,                                                           \
      wn,                                                           \
      transpose)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits) \
  instantiate_quantized_batched(name, type, group_size, bits, 1)      \
  instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_batched_wrap(affine_qmv_fast, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmv, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qvm, type, group_size, bits)     \
  instantiate_quantized_batched_wrap(affine_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_single(type, group_size, bits) \
  instantiate_quantized(affine_quantize, type, group_size, bits) \
  instantiate_quantized(affine_dequantize, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qvm, type, group_size, bits)     \
  instantiate_quantized(affine_gather_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)   \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true) \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

// vecs_per_tg (input-vector tile) 2..5; affine uses k_lanes=8 (more rows per
// simdgroup) where the fp path uses 16.
#define instantiate_quantized_wide_wrap(name, type, group_size, bits, vecs_per_tg, k_lanes) \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 0)         \
  instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, 1)

#define instantiate_quantized_all_wide(type, group_size, bits) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 2, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 3, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 4, 8) \
  instantiate_quantized_wide_wrap(affine_qmv_wide, type, group_size, bits, 5, 8)

#define instantiate_quantized_all_splitk(type, group_size, bits)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 8)   \
  instantiate_quantized_split_k(affine_qvm_split_k, type, group_size, bits, 32)  \

#define instantiate_quantized_splitk_qmm(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                           \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                     \
      type,                                                                     \
      group_size,                                                               \
      bits,                                                                     \
      aligned)

#define instantiate_quantized_all_splitk_qmm(type, group_size, bits)                    \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_splitk_qmm(affine_qmm_t_splitk, type, group_size, bits, false)

#define instantiate_quantized_all_rhs(type, group_size, bits) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nt, type, group_size, bits, 16, 32, 32, 1, 2, true) \
  instantiate_gather_qmm_rhs(affine_gather_qmm_rhs, affine_gather_qmm_rhs_nn, type, group_size, bits, 16, 32, 32, 1, 2, false)

#define instantiate_quantized_funcs(type, group_size, bits) \
  instantiate_quantized_all_single(type, group_size, bits)  \
  instantiate_quantized_all_batched(type, group_size, bits) \
  instantiate_quantized_all_aligned(type, group_size, bits) \
  instantiate_quantized_all_quad(type, group_size, bits)    \
  instantiate_quantized_all_wide(type, group_size, bits)    \
  instantiate_quantized_all_splitk(type, group_size, bits)  \
  instantiate_quantized_all_splitk_qmm(type, group_size, bits) \
  instantiate_quantized_all_rhs(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)       \
  instantiate_quantized_funcs(float, group_size, bits)      \
  instantiate_quantized_funcs(float16_t, group_size, bits)  \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits) \
  instantiate_quantized_types(128, bits)   \
  instantiate_quantized_types(64, bits)    \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_all() \
  instantiate_quantized_groups(2) \
  instantiate_quantized_groups(3) \
  instantiate_quantized_groups(4) \
  instantiate_quantized_groups(5) \
  instantiate_quantized_groups(6) \
  instantiate_quantized_groups(8)

instantiate_quantized_all()

// Large-M tile (BM=128, BN=64, BK=32) for affine_qmm_t, ahead-of-time
// compiled so the default (MLX_METAL_JIT=OFF) build -- what `pip install
// mlx` actually ships -- has this kernel available without falling back to
// runtime JIT compilation. See qmm()'s large-M dispatch branch in
// quantized.cpp: it only takes this path for the (group_size, bits)
// combos instantiated here, so a shape that lands outside this set at
// large M safely falls through to the original 32x32x32 tile instead of
// hitting a missing kernel. Scoped to bf16/fp16 x {gs=32,64} x {bits=4,8}
// -- the combo this fork's actual workload (evidence/2026-08-09-mlx-fork/)
// uses -- rather than instantiated across the full type/group_size/bits
// matrix above, to keep this addition's compile-time and binary-size cost
// bounded until a wider need is demonstrated. Naming must exactly match
// the kname built in quantized.cpp's qmm().
#define instantiate_quantized_large_m_tile(type, group_size, bits)          \
  instantiate_quantized_aligned_batched_tile(                               \
      affine_qmm_t, type, group_size, bits, true, 1, 128, 32, 64)           \
  instantiate_quantized_aligned_batched_tile(                               \
      affine_qmm_t, type, group_size, bits, true, 0, 128, 32, 64)           \
  instantiate_quantized_aligned_batched_tile(                               \
      affine_qmm_t, type, group_size, bits, false, 1, 128, 32, 64)          \
  instantiate_quantized_aligned_batched_tile(                               \
      affine_qmm_t, type, group_size, bits, false, 0, 128, 32, 64)

#define instantiate_quantized_large_m_tile_types(group_size, bits)  \
  instantiate_quantized_large_m_tile(float16_t, group_size, bits)   \
  instantiate_quantized_large_m_tile(bfloat16_t, group_size, bits)

instantiate_quantized_large_m_tile_types(32, 4)
instantiate_quantized_large_m_tile_types(32, 8)
instantiate_quantized_large_m_tile_types(64, 4)
instantiate_quantized_large_m_tile_types(64, 8) // clang-format on
