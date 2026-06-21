// Copyright © 2025 Apple Inc.

#define _USE_MATH_DEFINES

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cmath>

namespace mlx::core {
namespace rocm {

struct AttnParams {
  int B;
  int H;
  int D_q; // Query/Key head dimension
  int D_v; // Value head dimension
  int qL;
  int kL;
  int gqa_factor;
  float scale;
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
  int64_t M_strides[4]; // Mask strides [B, H, qL, kL]
  bool has_mask;
};

// Standard flash attention kernel (D_q == D_v, no array mask)
template <
    typename T,
    bool do_causal,
    int D,
    int BLOCK_M = 128,
    int BLOCK_N = 64>
__global__ __launch_bounds__(BLOCK_M) void kernel_sdpa_flash_opt(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const T* __restrict__ sinks,
    const AttnParams params) {
  // Grid: (H, ceil(qL / BLOCK_M), B)
  // Block: (BLOCK_M, 1, 1) -> 128 threads

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.x;
  int kv_head_idx = head_idx / params.gqa_factor;
  int q_seq_start = blockIdx.y * BLOCK_M;
  int thread_idx = threadIdx.x; // 0 to BLOCK_M - 1
  int q_seq_idx = q_seq_start + thread_idx;

  if (q_seq_start >= params.qL)
    return;

  const T* Q_ptr = Q + batch_idx * params.Q_strides[0] +
      head_idx * params.Q_strides[1] + q_seq_idx * params.Q_strides[2];
  T* O_ptr = O + batch_idx * params.O_strides[0] +
      head_idx * params.O_strides[1] + q_seq_idx * params.O_strides[2];

  bool valid_q = q_seq_idx < params.qL;

  typedef float U;

  // Registers for Q and O - use max of 256 for MLA value dimension
  U q[256];
  U o[256];

  if (valid_q) {
#pragma unroll
    for (int i = 0; i < D; i++) {
      q[i] = static_cast<U>(Q_ptr[i]);
      o[i] = 0.f;
    }
  }

  U max_score = -__int_as_float(0x7f7fffff); // -FLT_MAX
  U sum_exp_score = 0.f;

  if (sinks) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  __shared__ T K_sh[BLOCK_N][D];
  __shared__ T V_sh[BLOCK_N][D];

  const int K_seq_len = params.kL;

  for (int k_seq_start = 0; k_seq_start < K_seq_len; k_seq_start += BLOCK_N) {
    if constexpr (do_causal) {
      int earliest_valid_key = (K_seq_len - params.qL) + q_seq_start;
      int block_end_key = k_seq_start + BLOCK_N - 1;
      if (earliest_valid_key < block_end_key) {
        int max_q_seq_idx = min(q_seq_start + BLOCK_M - 1, params.qL - 1);
        int latest_valid_key = (K_seq_len - params.qL) + max_q_seq_idx;
        if (latest_valid_key < k_seq_start) {
          continue; // Block is completely causal-masked
        }
      }
    }

    __syncthreads();

    // Collaborative loading of K_sh and V_sh
    // BLOCK_N * D total elements = 64 * 128 = 8192.
    // We have BLOCK_M = 128 threads.
    // Each thread loads 8192 / 128 = 64 elements.
    const int elements_per_thread = (BLOCK_N * D) / BLOCK_M;

#pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
      int load_idx = i * BLOCK_M + thread_idx;
      int r = load_idx / D;
      int c = load_idx % D;
      int k_idx = k_seq_start + r;
      if (k_idx < K_seq_len) {
        K_sh[r][c] =
            K[batch_idx * params.K_strides[0] +
              kv_head_idx * params.K_strides[1] + k_idx * params.K_strides[2] +
              c];
        V_sh[r][c] =
            V[batch_idx * params.V_strides[0] +
              kv_head_idx * params.V_strides[1] + k_idx * params.V_strides[2] +
              c];
      } else {
        K_sh[r][c] = static_cast<T>(0.f);
        V_sh[r][c] = static_cast<T>(0.f);
      }
    }

    __syncthreads();

    if (valid_q) {
      // Loop over keys in the shared memory
      for (int i = 0; i < BLOCK_N; i++) {
        int k_idx = k_seq_start + i;
        if (k_idx >= K_seq_len)
          break;

        bool use_key = true;
        if constexpr (do_causal) {
          use_key = k_idx <= (K_seq_len - params.qL + q_seq_idx);
        }

        if (use_key) {
          U score = 0.f;

#pragma unroll 16
          for (int j = 0; j < D; j++) {
            score += q[j] * static_cast<U>(K_sh[i][j]);
          }

          score *= params.scale;

          U new_max = max(max_score, score);
          U factor = expf(max_score - new_max);
          U exp_score = expf(score - new_max);

          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;

#pragma unroll 16
          for (int j = 0; j < D; j++) {
            o[j] = o[j] * factor + exp_score * static_cast<U>(V_sh[i][j]);
          }
        }
      }
    }
  }

  if (valid_q) {
    U inv_sum = sum_exp_score == 0 ? 0.f : 1.0f / sum_exp_score;
#pragma unroll 16
    for (int i = 0; i < D; i++) {
      O_ptr[i] = static_cast<T>(o[i] * inv_sum);
    }
  }
}

// MLA flash attention kernel with array mask support
// Supports different Q and V dimensions and additive mask (pe_scores)
// Note: BLOCK_N=32 to fit shared memory constraints (K_sh: 24KB + V_sh: 32KB =
// 56KB < 64KB)
template <
    typename T,
    bool do_causal,
    int D_Q,
    int D_V,
    int BLOCK_M = 64,
    int BLOCK_N = 32>
__global__ __launch_bounds__(BLOCK_M) void kernel_sdpa_flash_mla(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    const T* __restrict__ mask, // Additive mask (pe_scores) [B, H, qL, kL]
    T* __restrict__ O,
    const T* __restrict__ sinks,
    const AttnParams params) {
  // Grid: (H, ceil(qL / BLOCK_M), B)
  // Block: (BLOCK_M, 1, 1)

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.x;
  int kv_head_idx = head_idx / params.gqa_factor;
  int q_seq_start = blockIdx.y * BLOCK_M;
  int thread_idx = threadIdx.x;
  int q_seq_idx = q_seq_start + thread_idx;

  if (q_seq_start >= params.qL)
    return;

  const T* Q_ptr = Q + batch_idx * params.Q_strides[0] +
      head_idx * params.Q_strides[1] + q_seq_idx * params.Q_strides[2];
  T* O_ptr = O + batch_idx * params.O_strides[0] +
      head_idx * params.O_strides[1] + q_seq_idx * params.O_strides[2];

  // Mask pointer for this query position
  const T* M_ptr = params.has_mask
      ? (mask + batch_idx * params.M_strides[0] +
         head_idx * params.M_strides[1] + q_seq_idx * params.M_strides[2])
      : nullptr;

  bool valid_q = q_seq_idx < params.qL;

  typedef float U;

  // Registers for Q and O
  U q[D_Q];
  U o[D_V];

  if (valid_q) {
#pragma unroll
    for (int i = 0; i < D_Q; i++) {
      q[i] = static_cast<U>(Q_ptr[i]);
    }
#pragma unroll
    for (int i = 0; i < D_V; i++) {
      o[i] = 0.f;
    }
  }

  U max_score = -__int_as_float(0x7f7fffff); // -FLT_MAX
  U sum_exp_score = 0.f;

  if (sinks) {
    max_score = static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  __shared__ T K_sh[BLOCK_N][D_Q];
  __shared__ T V_sh[BLOCK_N][D_V];

  const int K_seq_len = params.kL;

  for (int k_seq_start = 0; k_seq_start < K_seq_len; k_seq_start += BLOCK_N) {
    if constexpr (do_causal) {
      int earliest_valid_key = (K_seq_len - params.qL) + q_seq_start;
      int block_end_key = k_seq_start + BLOCK_N - 1;
      if (earliest_valid_key < block_end_key) {
        int max_q_seq_idx = min(q_seq_start + BLOCK_M - 1, params.qL - 1);
        int latest_valid_key = (K_seq_len - params.qL) + max_q_seq_idx;
        if (latest_valid_key < k_seq_start) {
          continue;
        }
      }
    }

    __syncthreads();

    // Collaborative loading of K_sh (D_Q elements per row)
    {
      const int total_k_elements = BLOCK_N * D_Q;
      const int k_per_thread = (total_k_elements + BLOCK_M - 1) / BLOCK_M;
#pragma unroll
      for (int i = 0; i < k_per_thread; i++) {
        int load_idx = i * BLOCK_M + thread_idx;
        if (load_idx < total_k_elements) {
          int r = load_idx / D_Q;
          int c = load_idx % D_Q;
          int k_idx = k_seq_start + r;
          if (k_idx < K_seq_len) {
            K_sh[r][c] =
                K[batch_idx * params.K_strides[0] +
                  kv_head_idx * params.K_strides[1] +
                  k_idx * params.K_strides[2] + c];
          } else {
            K_sh[r][c] = static_cast<T>(0.f);
          }
        }
      }
    }

    // Collaborative loading of V_sh (D_V elements per row)
    {
      const int total_v_elements = BLOCK_N * D_V;
      const int v_per_thread = (total_v_elements + BLOCK_M - 1) / BLOCK_M;
#pragma unroll
      for (int i = 0; i < v_per_thread; i++) {
        int load_idx = i * BLOCK_M + thread_idx;
        if (load_idx < total_v_elements) {
          int r = load_idx / D_V;
          int c = load_idx % D_V;
          int k_idx = k_seq_start + r;
          if (k_idx < K_seq_len) {
            V_sh[r][c] =
                V[batch_idx * params.V_strides[0] +
                  kv_head_idx * params.V_strides[1] +
                  k_idx * params.V_strides[2] + c];
          } else {
            V_sh[r][c] = static_cast<T>(0.f);
          }
        }
      }
    }

    __syncthreads();

    if (valid_q) {
// Loop over keys in the shared memory
#pragma unroll 4
      for (int i = 0; i < BLOCK_N; i++) {
        int k_idx = k_seq_start + i;
        if (k_idx >= K_seq_len)
          break;

        bool use_key = true;
        if constexpr (do_causal) {
          use_key = k_idx <= (K_seq_len - params.qL + q_seq_idx);
        }

        if (use_key) {
          // Compute Q @ K score
          U score = 0.f;

#pragma unroll 16
          for (int j = 0; j < D_Q; j++) {
            score += q[j] * static_cast<U>(K_sh[i][j]);
          }

          score *= params.scale;

          // Add mask bias (pe_scores) if present
          if (M_ptr) {
            score += static_cast<U>(M_ptr[k_idx * params.M_strides[3]]);
          }

          U new_max = max(max_score, score);
          U factor = expf(max_score - new_max);
          U exp_score = expf(score - new_max);

          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;

#pragma unroll 16
          for (int j = 0; j < D_V; j++) {
            o[j] = o[j] * factor + exp_score * static_cast<U>(V_sh[i][j]);
          }
        }
      }
    }
  }

  if (valid_q) {
    U inv_sum = sum_exp_score == 0 ? 0.f : 1.0f / sum_exp_score;
#pragma unroll 16
    for (int i = 0; i < D_V; i++) {
      O_ptr[i] = static_cast<T>(o[i] * inv_sum);
    }
  }
}

} // namespace rocm

bool supports_sdpa_flash(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp) {
  if (output_logsumexp) {
    return false;
  }
  if (q.dtype() != float32 && q.dtype() != float16 && q.dtype() != bfloat16) {
    return false;
  }
  const int D_q = q.shape(-1);
  const int D_v = v.shape(-1);

  // Standard attention dimensions (D_q == D_v)
  bool standard_dims = (D_q == 64 || D_q == 96 || D_q == 128 || D_q == 256);

  // MLA attention dimensions (D_q=192, D_v=256)
  bool mla_dims = (D_q == 192 && D_v == 256);

  if (D_q == D_v && standard_dims) {
    if (D_q == 256 && q.dtype() == float32) {
      return false;
    }
    // Standard attention: no array mask needed for flash kernel
    return !has_arr_mask;
  } else if (mla_dims) {
    // MLA attention: supports array mask (additive bias)
    return true;
  }
  return false;
}

void sdpa_flash(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks,
    Stream s) {
  auto& d = rocm::device(s.device);
  auto& encoder = d.get_command_encoder(s);

  int B = q.shape(0);
  int H = q.shape(1);
  int qL = q.shape(2);
  int kL = k.shape(2);
  int D_q = q.shape(3);
  int D_v = v.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  o.set_data(mlx::core::rocm::malloc_async(o.nbytes(), encoder));

  rocm::AttnParams params;
  params.B = B;
  params.H = H;
  params.D_q = D_q;
  params.D_v = D_v;
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

  params.has_mask = mask.has_value();
  if (mask) {
    params.M_strides[0] = mask->strides(0);
    params.M_strides[1] = mask->strides(1);
    params.M_strides[2] = mask->strides(2);
    params.M_strides[3] = mask->strides(3);
  }

  bool has_sinks = sinks.has_value();
  bool has_mask_val = mask.has_value();
  bool is_mla = (D_q == 192 && D_v == 256);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (mask) {
    encoder.set_input_array(*mask);
  }
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  encoder.set_output_array(o);

  {
    if (is_mla) {
      // MLA kernel with D_q=192, D_v=256
      // Use BLOCK_N=32 to fit shared memory (K_sh: 24KB + V_sh: 32KB = 56KB <
      // 64KB limit)
      constexpr int BLOCK_M = 64;
      constexpr int BLOCK_N = 32;
      int grid_y = (qL + BLOCK_M - 1) / BLOCK_M;
      dim3 grid_dim(H, grid_y, B);
      dim3 block_dim(BLOCK_M, 1, 1);

      auto launch_mla_kernel = [&](auto type_tag, auto causal_tag) {
        using DataType = decltype(type_tag);
        constexpr bool causal = decltype(causal_tag)::value;

        encoder.add_kernel_node(
            &rocm::kernel_sdpa_flash_mla<
                DataType,
                causal,
                192,
                256,
                BLOCK_M,
                BLOCK_N>,
            grid_dim,
            block_dim,
            0,
            gpu_ptr<DataType>(q),
            gpu_ptr<DataType>(k),
            gpu_ptr<DataType>(v),
            has_mask_val ? gpu_ptr<DataType>(*mask) : nullptr,
            gpu_ptr<DataType>(o),
            has_sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
            params);
      };

      if (o.dtype() == float32) {
        if (do_causal)
          launch_mla_kernel(float(), std::true_type());
        else
          launch_mla_kernel(float(), std::false_type());
      } else if (o.dtype() == float16) {
        if (do_causal)
          launch_mla_kernel(__half(), std::true_type());
        else
          launch_mla_kernel(__half(), std::false_type());
      } else if (o.dtype() == bfloat16) {
        if (do_causal)
          launch_mla_kernel(hip_bfloat16(), std::true_type());
        else
          launch_mla_kernel(hip_bfloat16(), std::false_type());
      }
    } else {
      // Standard flash attention kernel
      constexpr int BLOCK_M = 128;
      constexpr int BLOCK_N = 64;
      int grid_y = (qL + BLOCK_M - 1) / BLOCK_M;
      dim3 grid_dim(H, grid_y, B);
      dim3 block_dim(BLOCK_M, 1, 1);

      auto launch_kernel =
          [&](auto type_tag, auto causal_tag, auto headdim_tag) {
            using DataType = decltype(type_tag);
            constexpr bool causal = decltype(causal_tag)::value;
            constexpr int headdim = decltype(headdim_tag)::value;

            encoder.add_kernel_node(
                &rocm::kernel_sdpa_flash_opt<
                    DataType,
                    causal,
                    headdim,
                    BLOCK_M,
                    BLOCK_N>,
                grid_dim,
                block_dim,
                0,
                gpu_ptr<DataType>(q),
                gpu_ptr<DataType>(k),
                gpu_ptr<DataType>(v),
                gpu_ptr<DataType>(o),
                has_sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
                params);
          };

      if (o.dtype() == float32) {
        if (do_causal) {
          if (D_q == 64)
            launch_kernel(
                float(), std::true_type(), std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                float(), std::true_type(), std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                float(), std::true_type(), std::integral_constant<int, 128>());
        } else {
          if (D_q == 64)
            launch_kernel(
                float(), std::false_type(), std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                float(), std::false_type(), std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                float(), std::false_type(), std::integral_constant<int, 128>());
        }
      } else if (o.dtype() == float16) {
        if (do_causal) {
          if (D_q == 64)
            launch_kernel(
                __half(), std::true_type(), std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                __half(), std::true_type(), std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                __half(), std::true_type(), std::integral_constant<int, 128>());
          else if (D_q == 256)
            launch_kernel(
                __half(), std::true_type(), std::integral_constant<int, 256>());
        } else {
          if (D_q == 64)
            launch_kernel(
                __half(), std::false_type(), std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                __half(), std::false_type(), std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                __half(),
                std::false_type(),
                std::integral_constant<int, 128>());
          else if (D_q == 256)
            launch_kernel(
                __half(),
                std::false_type(),
                std::integral_constant<int, 256>());
        }
      } else if (o.dtype() == bfloat16) {
        if (do_causal) {
          if (D_q == 64)
            launch_kernel(
                hip_bfloat16(),
                std::true_type(),
                std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                hip_bfloat16(),
                std::true_type(),
                std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                hip_bfloat16(),
                std::true_type(),
                std::integral_constant<int, 128>());
          else if (D_q == 256)
            launch_kernel(
                hip_bfloat16(),
                std::true_type(),
                std::integral_constant<int, 256>());
        } else {
          if (D_q == 64)
            launch_kernel(
                hip_bfloat16(),
                std::false_type(),
                std::integral_constant<int, 64>());
          else if (D_q == 96)
            launch_kernel(
                hip_bfloat16(),
                std::false_type(),
                std::integral_constant<int, 96>());
          else if (D_q == 128)
            launch_kernel(
                hip_bfloat16(),
                std::false_type(),
                std::integral_constant<int, 128>());
          else if (D_q == 256)
            launch_kernel(
                hip_bfloat16(),
                std::false_type(),
                std::integral_constant<int, 256>());
        }
      }
    }
  }
}

} // namespace mlx::core
