// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

// Naive GEMM implementation for when rocBLAS is not available
// C = alpha * op(A) * op(B) + beta * C
// where op(X) = X if not transposed, X^T if transposed
void naive_gemm(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    float alpha = 1.0f,
    float beta = 0.0f);

// Batched naive GEMM
void naive_gemm_batched(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    float alpha = 1.0f,
    float beta = 0.0f);

// Batched gather GEMM where matrix selection is driven by index arrays.
void naive_gemm_gather(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    float alpha = 1.0f,
    float beta = 0.0f);

// Naive GEMM with explicit offsets (for non-uniform batch strides)
void naive_gemm_with_offset(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t out_offset,
    float alpha = 1.0f,
    float beta = 0.0f);

// Naive GEMM with explicit offsets and custom ldc (for grouped conv)
void naive_gemm_with_offset_ldc(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t ldc,
    int64_t out_offset,
    float alpha = 1.0f,
    float beta = 0.0f);

// Device-side MoE sorted-gather GEMM: no host D2H / stream sync.
// A is [batch, K] contiguous (M=1 token rows pre-stacked). B is expert-batched
// with 1-D batch dim E (stride b_expert_stride elements). rhs_indices[batch]
// must be sorted by expert id in 0..E-1. Each expert's token run is found by
// binary search on-device; C[start:end] = A[start:end] @ B[e].
void moe_sorted_expert_gemm(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& rhs_indices,
    array& out,
    int batch,
    int N,
    int K,
    int n_experts,
    bool b_transposed,
    int64_t ldb,
    int64_t b_expert_stride);

// Pack sorted MoE tokens into a dense [E, M_fixed, K] buffer (atomic slots).
// slot_map[e, s] = source token row, or -1 if unused. counts[e] = tokens packed.
void moe_pack_tokens(
    CommandEncoder& encoder,
    const array& a, // [batch, K]
    const array& rhs_indices, // [batch] expert ids
    array& packed_a, // [E, M_fixed, K]
    array& slot_map, // [E, M_fixed] int32
    array& counts, // [E] int32
    int batch,
    int K,
    int n_experts,
    int M_fixed);

// Scatter packed [E, M_fixed, N] gemm output back to [batch, N] via slot_map.
void moe_unpack_tokens(
    CommandEncoder& encoder,
    const array& packed_c, // [E, M_fixed, N]
    const array& slot_map, // [E, M_fixed] int32
    array& out, // [batch, N] (M=1 layout OK as flat batch*N)
    int n_experts,
    int M_fixed,
    int N);

// h = silu(g) * u for n bf16 elements (fused MoE SwiGLU mid).
void silu_mul_bf16(
    CommandEncoder& encoder,
    const void* gate,
    const void* up,
    void* h,
    int n);

// SwiGLU mid + elementwise bwd (all length-n bf16):
//   h  = silu(g) * u
//   du = dh * silu(g)
//   dg = dh * u * dsilu(g)
// where silu(g)=g*sigmoid(g), dsilu=s*(1+g*(1-s)), s=sigmoid(g).
void swiglu_bwd_elem_bf16(
    CommandEncoder& encoder,
    const void* gate,
    const void* up,
    const void* dh,
    void* h,
    void* dg,
    void* du,
    int n);

// out[i] += src[i] for n bf16 elements (used by padded VJP beta=1 path).
void bf16_add_inplace(
    CommandEncoder& encoder,
    const void* src,
    void* out,
    int n);

// out[j,i] = in[i,j] for matrix (rows, cols) bf16 (row-major).
void bf16_transpose_2d(
    CommandEncoder& encoder,
    const void* in,
    void* out,
    int rows,
    int cols);

// out[i] = concat(a[i], b[i]) for T rows of width K (bf16). out is [T, 2K].
void bf16_concat_rows(
    CommandEncoder& encoder,
    const void* a,
    const void* b,
    void* out,
    int T,
    int K);

// Split [T, 2K] into a[T,K] and b[T,K] (or [E,M,2K]→ two [E,M,K] with same layout).
void bf16_split_rows(
    CommandEncoder& encoder,
    const void* ab,
    void* a,
    void* b,
    int rows,
    int K);

// Write segments[e] = {lo, hi} for sorted expert_ids[T] over e=0..E-1 (uint32).
// Fully device-side; no host sync.
void moe_sorted_segments(
    CommandEncoder& encoder,
    const array& expert_ids, // [T] uint32 sorted
    array& segments, // [E, 2] uint32
    int T,
    int E);

// Device-side SegmentedMM: out[s] = A[:,k0:k1] @ B[k0:k1,:] with (k0,k1) from
// device segments[2*s], segments[2*s+1]. No host D2H / stream sync.
void segmented_mm_device(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& segments,
    array& out,
    int M,
    int N,
    int num_segments,
    bool a_transposed,
    int64_t lda,
    int64_t a_k_stride,
    bool b_transposed,
    int64_t ldb,
    int64_t b_k_stride,
    int64_t out_stride);

} // namespace mlx::core::rocm
