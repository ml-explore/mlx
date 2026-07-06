// Optimized quantized matrix-vector multiply (GEMV) kernel for RDNA 3.5.
//
// Each warp (32 threads) cooperatively computes ONE output element by
// iterating along the K dimension with coalesced uint32 loads.
// 8 warps per block → 8 output elements per block.
//
// Key optimizations vs naive kernel:
//   1. Coalesced global memory access (adjacent threads read adjacent words)
//   2. Vectorized uint32 loads (8 values per word for 4-bit)
//   3. Warp shuffle reduction (no shared memory needed for reduction)
//   4. LDS for x vector sharing across 8 warps in a block

#include "mlx/backend/rocm/quantized/qdequant.hpp"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// ---------------------------------------------------------------------------
// qmv_fast_kernel: Warp-cooperative quantized GEMV
// ---------------------------------------------------------------------------
// Grid:  dim3(M, ceildiv(N, ROWS_PER_BLOCK))
// Block: dim3(WARP_SIZE, ROWS_PER_BLOCK) = dim3(32, 8) = 256 threads
//
// Each warp (threadIdx.y selects the warp) computes one output element.
// All 32 lanes iterate over K together with coalesced weight loads.

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ __launch_bounds__(256)
void qmv_fast_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint32_t* __restrict__ w,    // [N, K/pack_factor_u32] as uint32
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias)
{
  constexpr int PF = pack_factor_u32<BITS>;      // values per uint32 (8 for 4-bit)
  constexpr int PPT = packs_per_thread<BITS>;    // uint32 loads per thread (2 for 4-bit)
  constexpr int VPT = values_per_thread<BITS>;   // values per thread per step (16)
  constexpr int BSK = VPT * WARP_SIZE;           // K-elements per warp per step (512)

  const int m = blockIdx.x;                              // output row
  const int n = blockIdx.y * ROWS_PER_BLOCK + threadIdx.y; // output column
  const int lane = threadIdx.x;                           // lane within warp
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;  // flat thread id

  // NOTE: Do NOT early-return here — all threads must participate in __syncthreads.
  const bool valid = (m < M && n < N);

  // --- LDS for x vector (shared across all 8 warps) ---
  __shared__ float x_shared[BSK];

  // Per-warp pointers (safe even if n >= N: we just won't write output)
  const int w_stride = K / PF;  // number of uint32 per weight row
  const int clamped_n = (n < N) ? n : 0;  // clamp to avoid OOB on pointer setup
  const uint32_t* w_row = w + clamped_n * w_stride;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const ScaleT* s_row = scales + clamped_n * num_groups;
  const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;
  const T* x_row = x + m * K;

  float acc = 0.0f;

  for (int k_base = 0; k_base < K; k_base += BSK) {
    // --- Cooperative load of x into LDS ---
    // All 256 threads participate (including invalid ones) to avoid barrier mismatch.
    __syncthreads();
    #pragma unroll
    for (int i = tid; i < BSK; i += ROWS_PER_BLOCK * WARP_SIZE) {
      int k = k_base + i;
      x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
    }
    __syncthreads();

    if (!valid) continue;  // Skip compute but still participate in barriers

    // --- Each lane loads its slice of x from LDS ---
    float x_local[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
      x_local[i] = x_shared[lane * VPT + i];
    }

    // --- Coalesced weight load + dequant + accumulate ---
    // Metal-compatible accumulation: separate integer dot product from scaling.
    // We accumulate dot(x, q_int) and sum(x) across ALL packs in the same
    // group, then apply: acc += scale * total_qdot + bias * total_xsum.
    // This matches Metal's qdot() which computes scale*accum + sum*bias
    // over all values_per_thread at once.
    int w_offset = k_base / PF + lane * PPT;

    // Accumulate integer dot and x-sum across all packs (same group for all)
    float group_qdot = 0.0f;
    float group_xsum = 0.0f;

    // All PPT packs share the same group (thread's 16 values are contiguous)
    int k_val = k_base + lane * VPT;
    int group_idx = k_val / GROUP_SIZE;

    #pragma unroll
    for (int p = 0; p < PPT; p++) {
      uint32_t packed = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      dequant_and_dot<BITS>(packed, &x_local[p * PF], group_qdot, group_xsum);
    }

    // Apply scale and bias ONCE for the whole group (matches Metal)
    float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
    float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
    acc += scale * group_qdot + bias * group_xsum;
  }

  if (!valid) return;

  // --- Warp reduction ---
  acc = warp_reduce_sum(acc);

  // --- Lane 0 writes output ---
  if (lane == 0) {
    out[m * N + n] = from_float<T>(acc);
  }
}

// ---------------------------------------------------------------------------
// gather_qmv_fast_kernel: Warp-cooperative gather-based quantized GEMV
// ---------------------------------------------------------------------------
// Same as qmv_fast_kernel but with batch index indirection for MoE models.

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ __launch_bounds__(256)
void gather_qmv_fast_kernel(
    const T* __restrict__ x,                     // [LHS_B, M, K]
    const uint32_t* __restrict__ w,              // [E, N, K/pack_factor] as uint32
    const ScaleT* __restrict__ scales,           // [E, N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases,           // [E, N, K/GROUP_SIZE] or nullptr
    const uint32_t* __restrict__ lhs_indices,    // [B]
    const uint32_t* __restrict__ rhs_indices,    // [B]
    T* __restrict__ out,                         // [B, M, N]
    int B, int M, int N, int K, int E, int LHS_B,
    bool has_bias)
{
  constexpr int PF = pack_factor_u32<BITS>;
  constexpr int PPT = packs_per_thread<BITS>;
  constexpr int VPT = values_per_thread<BITS>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int batch = blockIdx.z;
  const int m = blockIdx.x;
  const int n = blockIdx.y * ROWS_PER_BLOCK + threadIdx.y;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

  const bool valid = (batch < B && m < M && n < N);

  uint32_t lhs_idx = valid ? lhs_indices[batch] : 0;
  uint32_t rhs_idx = valid ? rhs_indices[batch] : 0;

  // Clamp indices to valid range to prevent catastrophic OOB on corrupt data.
  if (lhs_idx >= static_cast<uint32_t>(LHS_B)) lhs_idx = 0;
  if (rhs_idx >= static_cast<uint32_t>(E)) rhs_idx = 0;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const int clamped_n = (n < N) ? n : 0;
  const uint32_t* w_row = w + rhs_idx * N * w_stride + clamped_n * w_stride;
  const ScaleT* s_row = scales + rhs_idx * N * num_groups + clamped_n * num_groups;
  const ScaleT* b_row = has_bias ? (biases + rhs_idx * N * num_groups + clamped_n * num_groups) : nullptr;
  const T* x_row = x + lhs_idx * M * K + m * K;

  float acc = 0.0f;

  for (int k_base = 0; k_base < K; k_base += BSK) {
    __syncthreads();
    #pragma unroll
    for (int i = tid; i < BSK; i += ROWS_PER_BLOCK * WARP_SIZE) {
      int k = k_base + i;
      x_shared[i] = (k < K && valid) ? to_float(x_row[k]) : 0.0f;
    }
    __syncthreads();

    if (!valid) continue;

    float x_local[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
      x_local[i] = x_shared[lane * VPT + i];
    }

    int w_offset = k_base / PF + lane * PPT;

    // Accumulate integer dot and x-sum across all packs (same group)
    float group_qdot = 0.0f;
    float group_xsum = 0.0f;

    int k_val = k_base + lane * VPT;
    int group_idx = k_val / GROUP_SIZE;

    #pragma unroll
    for (int p = 0; p < PPT; p++) {
      uint32_t packed = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      dequant_and_dot<BITS>(packed, &x_local[p * PF], group_qdot, group_xsum);
    }

    float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
    float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
    acc += scale * group_qdot + bias * group_xsum;
  }

  if (!valid) return;

  acc = warp_reduce_sum(acc);

  if (lane == 0) {
    out[batch * M * N + m * N + n] = from_float<T>(acc);
  }
}

} // namespace mlx::core::rocm
