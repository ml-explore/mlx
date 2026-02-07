// Copyright © 2025 Apple Inc.

#define _USE_MATH_DEFINES

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>

namespace mlx::core {

namespace rocm {

// Virtual warp size for SDPA - always 32 threads for consistent behavior
// across RDNA (32-wide) and CDNA (64-wide) architectures
constexpr int SDPA_TILE_SIZE = 32;

struct AttnParams {
  int B;
  int H;
  int D;
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

// Tile-based reduction for 32-thread groups (works on both RDNA and CDNA)
template <typename T>
__device__ __forceinline__ T tile_reduce_sum_32(T val) {
  // Reduce within a 32-thread tile using shuffle operations
  val += __shfl_xor(val, 16);
  val += __shfl_xor(val, 8);
  val += __shfl_xor(val, 4);
  val += __shfl_xor(val, 2);
  val += __shfl_xor(val, 1);
  return val;
}

template <typename T>
__device__ __forceinline__ T tile_reduce_max_32(T val) {
  // Reduce within a 32-thread tile using shuffle operations
  T other;
  other = __shfl_xor(val, 16); val = val > other ? val : other;
  other = __shfl_xor(val, 8);  val = val > other ? val : other;
  other = __shfl_xor(val, 4);  val = val > other ? val : other;
  other = __shfl_xor(val, 2);  val = val > other ? val : other;
  other = __shfl_xor(val, 1);  val = val > other ? val : other;
  return val;
}

// Single-pass SDPA kernel for short sequences
// Uses 32-thread tiles for consistent behavior across architectures
template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_1pass(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const T* sinks,
    const AttnParams params) {
  
  // BN = number of 32-thread tiles, BD = tile size (32)
  constexpr int BN = 32;  // Number of tiles processing keys in parallel
  constexpr int BD = 32;  // Tile size (always 32 for consistency)
  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = BN * params.K_strides[2];
  const int inner_v_stride = BN * params.V_strides[2];

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * 1.44269504089f; // M_LOG2E

  // Use virtual 32-thread tiles instead of hardware warps
  const int lane_idx = threadIdx.x % SDPA_TILE_SIZE;  // 0-31 within tile
  const int tile_idx = threadIdx.x / SDPA_TILE_SIZE;  // Which tile (0-31)

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;
  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = tile_idx;

  const T* Q_ptr = Q + batch_idx * params.Q_strides[0] + head_idx * params.Q_strides[1] + q_seq_idx * params.Q_strides[2];
  const T* K_ptr = K + batch_idx * params.K_strides[0] + kv_head_idx * params.K_strides[1] + kv_seq_idx * params.K_strides[2];
  const T* V_ptr = V + batch_idx * params.V_strides[0] + kv_head_idx * params.V_strides[1] + kv_seq_idx * params.V_strides[2];
  T* O_ptr = O + batch_idx * params.O_strides[0] + head_idx * params.O_strides[1] + q_seq_idx * params.O_strides[2];

  // Read query and initialize output
  #pragma unroll
  for (int i = 0; i < v_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q_ptr[v_per_thread * lane_idx + i]);
    o[i] = 0.f;
  }

  U max_score = -1e9f;
  U sum_exp_score = 0.f;

  // Process keys
  for (int i = kv_seq_idx; i < params.kL; i += BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    if (use_key) {
      #pragma unroll
      for (int j = 0; j < v_per_thread; j++) {
        k[j] = K_ptr[v_per_thread * lane_idx + j];
      }

      U score = 0.f;
      #pragma unroll
      for (int j = 0; j < v_per_thread; j++) {
        score += q[j] * static_cast<U>(k[j]);
      }

      // Reduce within 32-thread tile
      score = tile_reduce_sum_32(score);

      U new_max = max(max_score, score);
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      #pragma unroll
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(V_ptr[v_per_thread * lane_idx + j]);
      }
    }

    K_ptr += inner_k_stride;
    V_ptr += inner_v_stride;
  }

  // Store per-tile results to shared memory
  if (lane_idx == 0) {
    max_scores[tile_idx] = max_score;
    sum_exp_scores[tile_idx] = sum_exp_score;
  }
  __syncthreads();

  // Cross-tile reduction
  max_score = max_scores[lane_idx % BN];
  U new_max = tile_reduce_max_32(max_score);
  U factor = exp2f(max_score - new_max);
  sum_exp_score = tile_reduce_sum_32(sum_exp_scores[lane_idx % BN] * factor);
  sum_exp_score = sum_exp_score == 0 ? 0 : 1.0f / sum_exp_score;

  // Aggregate outputs across tiles
  #pragma unroll
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][tile_idx] = o[i];
    __syncthreads();
    U ot = outputs[tile_idx][lane_idx] * factor;
    o[i] = tile_reduce_sum_32(ot) * sum_exp_score;
    __syncthreads();
  }

  // Write final output
  if (lane_idx == 0) {
    #pragma unroll
    for (int i = 0; i < v_per_thread; i++) {
      O_ptr[v_per_thread * tile_idx + i] = static_cast<T>(o[i]);
    }
  }
}

} // namespace rocm

// Forward declarations
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp);

void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp) {
  // Temporarily disable optimized SDPA to debug memory fault
  // The memory fault occurs even with SDPA disabled, so the issue is elsewhere
  return false;
}

void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s) {
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  int B = q.shape(0);
  int H = q.shape(1);
  int qL = q.shape(2);
  int kL = k.shape(2);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  // Allocate output
  o.set_data(allocator::malloc(o.nbytes()));

  // Build params struct
  rocm::AttnParams params;
  params.B = B;
  params.H = H;
  params.D = D;
  params.qL = qL;
  params.kL = kL;
  params.gqa_factor = gqa_factor;
  params.scale = scale;
  params.Q_strides[0] = q.strides(0);
  params.Q_strides[1] = q.strides(1);
  params.Q_strides[2] = q.strides(2);
  params.K_strides[0] = k.strides(0);
  params.K_strides[1] = k.strides(1);
  params.K_strides[2] = k.strides(2);
  params.V_strides[0] = v.strides(0);
  params.V_strides[1] = v.strides(1);
  params.V_strides[2] = v.strides(2);
  params.O_strides[0] = o.strides(0);
  params.O_strides[1] = o.strides(1);
  params.O_strides[2] = o.strides(2);

  encoder.launch_kernel([&](hipStream_t stream) {
    dim3 grid_dim(H, qL, B);
    dim3 block_dim(1024, 1, 1);  // 32 tiles * 32 threads = 1024

    auto launch_kernel = [&](auto type_tag, auto causal_tag, auto headdim_tag) {
      using DataType = decltype(type_tag);
      constexpr bool causal = decltype(causal_tag)::value;
      constexpr int headdim = decltype(headdim_tag)::value;
      
      hipLaunchKernelGGL(
          (rocm::kernel_sdpav_1pass<DataType, causal, headdim>),
          grid_dim, block_dim, 0, stream,
          q.data<DataType>(),
          k.data<DataType>(),
          v.data<DataType>(),
          o.data<DataType>(),
          sinks ? sinks->data<DataType>() : nullptr,
          params);
    };

    // Dispatch based on dtype, causal, and head dimension
    if (o.dtype() == float32) {
      if (do_causal) {
        if (D == 64) launch_kernel(float(), std::true_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(float(), std::true_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(float(), std::true_type(), std::integral_constant<int, 128>());
      } else {
        if (D == 64) launch_kernel(float(), std::false_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(float(), std::false_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(float(), std::false_type(), std::integral_constant<int, 128>());
      }
    } else if (o.dtype() == float16) {
      if (do_causal) {
        if (D == 64) launch_kernel(__half(), std::true_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(__half(), std::true_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(__half(), std::true_type(), std::integral_constant<int, 128>());
      } else {
        if (D == 64) launch_kernel(__half(), std::false_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(__half(), std::false_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(__half(), std::false_type(), std::integral_constant<int, 128>());
      }
    } else if (o.dtype() == bfloat16) {
      if (do_causal) {
        if (D == 64) launch_kernel(hip_bfloat16(), std::true_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(hip_bfloat16(), std::true_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(hip_bfloat16(), std::true_type(), std::integral_constant<int, 128>());
      } else {
        if (D == 64) launch_kernel(hip_bfloat16(), std::false_type(), std::integral_constant<int, 64>());
        else if (D == 96) launch_kernel(hip_bfloat16(), std::false_type(), std::integral_constant<int, 96>());
        else if (D == 128) launch_kernel(hip_bfloat16(), std::false_type(), std::integral_constant<int, 128>());
      }
    }
  });
}

} // namespace mlx::core
