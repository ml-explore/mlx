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

// WARP_SIZE is defined in device/config.h based on target architecture

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

template <typename T>
__device__ T warp_reduce_sum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename T>
__device__ T warp_reduce_max(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other = __shfl_down(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// Single-pass SDPA kernel for short sequences
template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_1pass(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const T* sinks,
    int B, int H, int qL, int kL,
    int gqa_factor, float scale,
    const int64_t* Q_strides,
    const int64_t* K_strides,
    const int64_t* V_strides,
    const int64_t* O_strides) {
  
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = BN * K_strides[2];
  const int inner_v_stride = BN * V_strides[2];

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = scale * 1.44269504089f; // M_LOG2E

  const int lane_idx = threadIdx.x % WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / gqa_factor;
  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = warp_idx;

  const T* Q_ptr = Q + batch_idx * Q_strides[0] + head_idx * Q_strides[1] + q_seq_idx * Q_strides[2];
  const T* K_ptr = K + batch_idx * K_strides[0] + kv_head_idx * K_strides[1] + kv_seq_idx * K_strides[2];
  const T* V_ptr = V + batch_idx * V_strides[0] + kv_head_idx * V_strides[1] + kv_seq_idx * V_strides[2];
  T* O_ptr = O + batch_idx * O_strides[0] + head_idx * O_strides[1] + q_seq_idx * O_strides[2];

  // Read query and initialize output
  #pragma unroll
  for (int i = 0; i < v_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q_ptr[v_per_thread * lane_idx + i]);
    o[i] = 0.f;
  }

  U max_score = -1e9f;
  U sum_exp_score = 0.f;

  // Process keys
  for (int i = kv_seq_idx; i < kL; i += BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (kL - qL + q_seq_idx);
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

      score = warp_reduce_sum(score);

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

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }
  __syncthreads();

  max_score = max_scores[lane_idx % BN];
  U new_max = warp_reduce_max(max_score);
  U factor = exp2f(max_score - new_max);
  sum_exp_score = warp_reduce_sum(sum_exp_scores[lane_idx % BN] * factor);
  sum_exp_score = sum_exp_score == 0 ? 0 : 1.0f / sum_exp_score;

  #pragma unroll
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    __syncthreads();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = warp_reduce_sum(ot) * sum_exp_score;
    __syncthreads();
  }

  if (lane_idx == 0) {
    #pragma unroll
    for (int i = 0; i < v_per_thread; i++) {
      O_ptr[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
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
  // Disable optimized SDPA kernel for now - the kernel has warp size assumptions
  // that don't work correctly across all ROCm architectures (RDNA vs CDNA).
  // The kernel assumes 32 warps with 32 threads each (1024 total), but CDNA
  // architectures use 64-wide wavefronts, resulting in only 16 warps.
  // This causes out-of-bounds shared memory access.
  // TODO: Rewrite kernel to be warp-size agnostic.
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

  // Allocate stride arrays on device
  array Q_strides_arr({3}, int64, nullptr, {});
  array K_strides_arr({3}, int64, nullptr, {});
  array V_strides_arr({3}, int64, nullptr, {});
  array O_strides_arr({3}, int64, nullptr, {});
  
  Q_strides_arr.set_data(allocator::malloc(Q_strides_arr.nbytes()));
  K_strides_arr.set_data(allocator::malloc(K_strides_arr.nbytes()));
  V_strides_arr.set_data(allocator::malloc(V_strides_arr.nbytes()));
  O_strides_arr.set_data(allocator::malloc(O_strides_arr.nbytes()));
  
  encoder.add_temporary(Q_strides_arr);
  encoder.add_temporary(K_strides_arr);
  encoder.add_temporary(V_strides_arr);
  encoder.add_temporary(O_strides_arr);

  int64_t q_strides[3] = {q.strides(0), q.strides(1), q.strides(2)};
  int64_t k_strides[3] = {k.strides(0), k.strides(1), k.strides(2)};
  int64_t v_strides[3] = {v.strides(0), v.strides(1), v.strides(2)};
  int64_t o_strides[3] = {o.strides(0), o.strides(1), o.strides(2)};

  encoder.launch_kernel([&](hipStream_t stream) {
    (void)hipMemcpyAsync(Q_strides_arr.data<int64_t>(), q_strides, 3 * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(K_strides_arr.data<int64_t>(), k_strides, 3 * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(V_strides_arr.data<int64_t>(), v_strides, 3 * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    (void)hipMemcpyAsync(O_strides_arr.data<int64_t>(), o_strides, 3 * sizeof(int64_t), hipMemcpyHostToDevice, stream);

    dim3 grid_dim(H, qL, B);
    dim3 block_dim(1024, 1, 1);

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
          B, H, qL, kL, gqa_factor, scale,
          Q_strides_arr.data<int64_t>(),
          K_strides_arr.data<int64_t>(),
          V_strides_arr.data<int64_t>(),
          O_strides_arr.data<int64_t>());
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
