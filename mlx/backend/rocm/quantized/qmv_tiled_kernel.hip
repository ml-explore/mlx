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

  // Persistent grid-stride over output-column tiles. The grid is launched with a
  // CU-bounded number of blocks (not one per tile), and each block walks several
  // column tiles. This keeps the number of live blocks — and therefore the weight
  // stream's bandwidth — constant across the whole kernel, flattening the launch
  // ramp/tail that otherwise shows up as oscillating memory bandwidth. The whole
  // warp shares threadIdx.y, so `valid` and the __syncthreads below stay
  // warp/block-uniform.
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

      float group_qdot = 0.0f;
      float group_xsum = 0.0f;

      int k_val = k_base + lane * VPT;
      int group_idx = k_val / GROUP_SIZE;

      uint32_t w_local[PPT];
      // Warp-uniform branch: all lanes in bounds except possibly last K-tile.
      // Stream the read-once weights (non-temporal) so they don't evict the
      // reused X / scales from the small (2 MB on gfx1151) L2.
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
        dequant_and_dot<BITS>(w_local[p], &x_local[p * PF], group_qdot, group_xsum);
      }

      float scale = (group_idx < num_groups) ? to_float(s_row[group_idx]) : 0.0f;
      float bias = (has_bias && group_idx < num_groups) ? to_float(b_row[group_idx]) : 0.0f;
      acc += scale * group_qdot + bias * group_xsum;
    }

    if (!valid) continue;

    // Warp reduction
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
    float group_qdot = 0.0f;
    float group_xsum = 0.0f;
    int k_val = k_base + lane * VPT;
    int group_idx = k_val / GROUP_SIZE;

    uint32_t w_local[PPT];
    if (k_base + BSK <= K) {
      load_weight_vec<BITS>(w_row + w_offset, w_local);
    } else {
      #pragma unroll
      for (int p = 0; p < PPT; p++) {
        w_local[p] = (w_offset + p < w_stride) ? w_row[w_offset + p] : 0u;
      }
    }

    #pragma unroll
    for (int p = 0; p < PPT; p++) {
      dequant_and_dot<BITS>(w_local[p], &x_local[p * PF], group_qdot, group_xsum);
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
