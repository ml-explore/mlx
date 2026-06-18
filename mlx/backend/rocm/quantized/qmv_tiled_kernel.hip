// L2 cache-optimized quantized GEMV kernel for RDNA 3/3.5.
//
// Key difference from qmv_fast_kernel: processes TILE_N output columns per
// block instead of ROWS_PER_BLOCK=8. Within each K-tile, all TILE_N columns
// read from the same K-range of the weight matrix. Because adjacent columns
// access adjacent weight rows in the same K-range, these rows are likely to
// be in L2 cache, improving L2 hit rate from ~10% to ~40-70%.
//
// Grid:  dim3(M, ceildiv(N, TILE_N))
// Block: dim3(WARP_SIZE, TILE_N) — one warp per output column
//
// Each warp computes one output element by reducing along K.
// All warps in the block share the same X chunk via LDS.

#include "mlx/backend/rocm/quantized/qdequant.hpp"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// TILE_N is passed as a runtime kernel argument. The host selects it from
// rocm::ArchTuning::qmv_tile_n (per-arch config) and sets block dim to
// (WARP_SIZE, tile_n). The kernel reads tile_n to compute column indices.
// Performance: the shared memory load loop runs exactly 1 iteration for
// standard configs (BSK=512, stride=tile_n*32=512), so no unrolling loss.
static constexpr int TILE_N_MAX = 32; // Max for __launch_bounds__

template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ __launch_bounds__(TILE_N_MAX * WARP_SIZE)
void qmv_tiled_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint32_t* __restrict__ w,    // [N, K/pack_factor] as uint32
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,                        // Runtime TILE_N from arch config
    int n_tiles)                       // ceil(N / tile_n) — for grid-stride
{
  constexpr int PF = pack_factor_u32<BITS>;
  constexpr int PPT = packs_per_thread<BITS>;
  constexpr int VPT = values_per_thread<BITS>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK];

  const int w_stride = K / PF;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  // Grid-stride over column tiles (grid is CU-bounded, not one block per tile) to
  // keep the live block count steady. threadIdx.y is warp-uniform, so the
  // __syncthreads below stay block-uniform.
  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint32_t* w_row = w + clamped_n * w_stride;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BSK) {
      __syncthreads();
      for (int i = tid; i < BSK; i += nthreads) {
        int k = k_base + i;
        x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
      }
      __syncthreads();

      if (!valid) continue;

      // Each lane loads its X slice from LDS
      float x_local[VPT];
      #pragma unroll
      for (int i = 0; i < VPT; i++) {
        x_local[i] = x_shared[lane * VPT + i];
      }

      // Vectorized weight load + dequant + accumulate
      int w_offset = k_base / PF + lane * PPT;

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;

      uint32_t w_local[PPT];
      if (k_base + BSK <= K) {
        load_weight_vec_streaming<BITS>(w_row + w_offset, w_local);
      } else {
        #pragma unroll
        for (int p = 0; p < PPT; p++) {
          w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
        }
      }

      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        dequant_and_dot4<BITS>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);
      }

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
    }

    if (!valid) continue;

    // Warp reduction
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

// 6-bit tiled QMV.
//
// 6-bit packing is non-power-of-two, so pack_factor_u32 = 32/6 is not an
// integer and the generic qmv_tiled_kernel template cannot be used. This
// specialization reproduces the TILED occupancy/column-tiling/LDS-X-sharing
// structure (full Wave32 waves, tile_n columns/block sharing X in LDS) but
// loads 6-bit weights with explicit byte-aligned math.
//
// Layout: each lane owns VPT6=16 contiguous K-values. 16 six-bit weights =
// 96 bits = 12 bytes = 3 uint32. K%64==0 is enforced upstream for 6-bit, and
// BSK6 = VPT6 * WARP_SIZE = 512 (=> 384 bytes/warp-iter), so every lane's
// 16-weight slice begins on a byte boundary:
//   k0 = k_base + lane*16  =>  k0*6 = (k_base*6) + lane*96 bits  (byte-aligned)
// We process two byte-aligned sub-blocks of 8 weights each, extracted with the
// EXACT same uint64_t memcpy + shift + 0x3F mask used by the warp_shared 6-bit
// branch in qmm.hip, so dequant values are bit-identical.
//
// Accumulation uses the SAME Metal-compatible per-group order as the 4/8-bit
// tiled kernel: raw integer qdot + x-sum accumulated separately, then
// acc += scale * reduce_qdot4(qdot) + bias * xsum  once per group.
template <typename T, typename ScaleT, int GROUP_SIZE>
__device__ __forceinline__ void dequant_and_dot4_6bit(
    const uint8_t* __restrict__ w_bytes, // pointer to lane's 12-byte slice
    const float* __restrict__ x_local,   // 16 X values for this lane
    float (&qdot)[4],
    float& x_sum) {
  // Load exactly 12 bytes (3 uint32) — never over-reads past the lane slice,
  // so the final lane of a K-tile that ends exactly at K stays in-range.
  // The lane slice is byte-aligned (verified by the caller), so a uint32
  // triple load is well-defined.
  uint32_t w0 = *reinterpret_cast<const uint32_t*>(w_bytes + 0);
  uint32_t w1 = *reinterpret_cast<const uint32_t*>(w_bytes + 4);
  uint32_t w2 = *reinterpret_cast<const uint32_t*>(w_bytes + 8);

  // Sub-block 0: weights 0..7  (bytes 0..5)  = w0 | (low16(w1) << 32).
  // Sub-block 1: weights 8..15 (bytes 6..11) = high16(w1) | (w2 << 16).
  // bit_offset == 0 for both (k0*6 is byte-aligned), matching warp_shared.
  uint64_t sb0 = static_cast<uint64_t>(w0) |
      (static_cast<uint64_t>(w1 & 0xFFFFu) << 32);
  uint64_t sb1 = static_cast<uint64_t>(w1 >> 16) |
      (static_cast<uint64_t>(w2) << 16);

  #pragma unroll
  for (int sb = 0; sb < 2; sb++) {
    uint64_t w_packed = (sb == 0) ? sb0 : sb1;
    float q0 = static_cast<float>(w_packed & 0x3F);
    float q1 = static_cast<float>((w_packed >> 6) & 0x3F);
    float q2 = static_cast<float>((w_packed >> 12) & 0x3F);
    float q3 = static_cast<float>((w_packed >> 18) & 0x3F);
    float q4 = static_cast<float>((w_packed >> 24) & 0x3F);
    float q5 = static_cast<float>((w_packed >> 30) & 0x3F);
    float q6 = static_cast<float>((w_packed >> 36) & 0x3F);
    float q7 = static_cast<float>((w_packed >> 42) & 0x3F);
    const float* xb = x_local + sb * 8;
    qdot[0] += xb[0] * q0;
    qdot[1] += xb[1] * q1;
    qdot[2] += xb[2] * q2;
    qdot[3] += xb[3] * q3;
    qdot[0] += xb[4] * q4;
    qdot[1] += xb[5] * q5;
    qdot[2] += xb[6] * q6;
    qdot[3] += xb[7] * q7;
    x_sum += xb[0] + xb[1] + xb[2] + xb[3] + xb[4] + xb[5] + xb[6] + xb[7];
  }
}

template <typename T, typename ScaleT, int GROUP_SIZE>
__global__ __launch_bounds__(TILE_N_MAX * WARP_SIZE)
void qmv_tiled_6bit_kernel(
    const T* __restrict__ x,           // [M, K]
    const uint8_t* __restrict__ w,     // [N, row_bytes] raw bytes
    const ScaleT* __restrict__ scales, // [N, K/GROUP_SIZE]
    const ScaleT* __restrict__ biases, // [N, K/GROUP_SIZE] or nullptr
    T* __restrict__ out,               // [M, N]
    int M,
    int N,
    int K,
    bool has_bias,
    int tile_n,
    int n_tiles) {
  constexpr int BITS = 6;
  constexpr int VPT6 = 16;          // values per thread
  constexpr int BSK6 = VPT6 * WARP_SIZE; // 512
  constexpr int BYTES_PER_LANE = (VPT6 * BITS) / 8; // 12

  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  __shared__ float x_shared[BSK6];

  const int row_bytes = (K * BITS + 7) / 8;
  const int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;
  const T* x_row = x + m * K;

  for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
    const int n = tile * tile_n + threadIdx.y;
    const bool valid = (m < M && n < N);
    const int clamped_n = (n < N) ? n : 0;
    const uint8_t* w_row = w + clamped_n * row_bytes;
    const ScaleT* s_row = scales + clamped_n * num_groups;
    const ScaleT* b_row = has_bias ? (biases + clamped_n * num_groups) : nullptr;

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += BSK6) {
      __syncthreads();
      for (int i = tid; i < BSK6; i += nthreads) {
        int k = k_base + i;
        x_shared[i] = (k < K) ? to_float(x_row[k]) : 0.0f;
      }
      __syncthreads();

      if (!valid) continue;

      float x_local[VPT6];
      #pragma unroll
      for (int i = 0; i < VPT6; i++) {
        x_local[i] = x_shared[lane * VPT6 + i];
      }

      float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float group_xsum = 0.0f;

      const int k_val = k_base + lane * VPT6;
      const int group_idx = k_val / GROUP_SIZE;

      if (k_base + BSK6 <= K) {
        // Fast path: full warp tile in range, byte-aligned 12-byte lane slice.
        // k_base is a multiple of BSK6=512, so (k_base*6)/8 = k_base*3/4 is an
        // exact byte offset, and lane*12 keeps each lane byte-aligned.
        const uint8_t* w_lane = w_row + ((k_base * BITS) / 8) + lane * BYTES_PER_LANE;
        dequant_and_dot4_6bit<T, ScaleT, GROUP_SIZE>(
            w_lane, x_local, group_qdot4, group_xsum);
      } else {
        // Tail: extract each value with the EXACT bounded warp_shared bit math
        // (unpack_packed_value general branch): read byte_idx, plus byte_idx+1
        // only if in range.
        #pragma unroll
        for (int i = 0; i < VPT6; i++) {
          int k = k_val + i;
          if (k < K) {
            float xv = x_local[i];
            int bit_index = k * BITS;
            int byte_idx = bit_index >> 3;
            int bit_offset = bit_index & 0x7;
            uint32_t window = static_cast<uint32_t>(w_row[byte_idx]);
            if (byte_idx + 1 < row_bytes) {
              window |= static_cast<uint32_t>(w_row[byte_idx + 1]) << 8;
            }
            float q = static_cast<float>((window >> bit_offset) & 0x3Fu);
            group_qdot4[i & 3] += xv * q;
            group_xsum += xv;
          }
        }
      }

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
    }

    if (!valid) continue;

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
      out[m * N + n] = from_float<T>(acc);
    }
  }
}

// Gather variant for MoE models
template <typename T, typename ScaleT, int BITS, int GROUP_SIZE>
__global__ __launch_bounds__(TILE_N_MAX * WARP_SIZE)
void gather_qmv_tiled_kernel(
    const T* __restrict__ x,
    const uint32_t* __restrict__ w,
    const ScaleT* __restrict__ scales,
    const ScaleT* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    T* __restrict__ out,
    int B, int M, int N, int K, int E, int LHS_B,
    bool has_bias,
    int tile_n)
{
  constexpr int PF = pack_factor_u32<BITS>;
  constexpr int PPT = packs_per_thread<BITS>;
  constexpr int VPT = values_per_thread<BITS>;
  constexpr int BSK = VPT * WARP_SIZE;

  const int batch = blockIdx.z;
  const int m = blockIdx.x;
  const int n = blockIdx.y * tile_n + threadIdx.y;
  const int lane = threadIdx.x;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int nthreads = tile_n * WARP_SIZE;

  const bool valid = (batch < B && m < M && n < N);

  uint32_t lhs_idx = valid ? lhs_indices[batch] : 0;
  uint32_t rhs_idx = valid ? rhs_indices[batch] : 0;
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
    for (int i = tid; i < BSK; i += nthreads) {
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
    float group_qdot4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float group_xsum = 0.0f;
    int k_val = k_base + lane * VPT;
    int group_idx = k_val / GROUP_SIZE;

    uint32_t w_local[PPT];
    if (k_base + BSK <= K) {
      load_weight_vec_streaming<BITS>(w_row + w_offset, w_local);
    } else {
      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }
    }

    #pragma unroll
    for (int p = 0; p < PPT; p++) {
      dequant_and_dot4<BITS>(w_local[p], &x_local[p * PF], group_qdot4, group_xsum);
    }

    float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
    float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
    acc += scale * reduce_qdot4(group_qdot4) + bias * group_xsum;
  }

  if (!valid) return;
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    out[batch * M * N + m * N + n] = from_float<T>(acc);
  }
}

} // namespace mlx::core::rocm
