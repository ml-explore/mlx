// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;

template <
    typename DataType,
    bool vectorized_g,
    int N_V = 16,
    int V_PER_WARP = 8,
    int V_TILE = 8,
    int K_LOCAL = 4,
    uint32_t BLOCK_THREADS = 32>
__global__ __launch_bounds__(BLOCK_THREADS, 12) void gated_delta_seq(
    const DataType* q_ptr,
    const DataType* k_ptr,
    const DataType* v_ptr,
    const float* g_ptr,
    const float* beta_ptr,
    const float* h0_ptr,
    DataType* out,
    float* new_state,
    int Hk,
    int Hv,
    int seq_len) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  const int pid = grid.block_rank();
  const int warp_id = warp.meta_group_rank();
  const int lane = warp.thread_rank();

  const int bh = pid / N_V;
  const int v_tile_idx = pid - bh * N_V;
  const int b_idx = bh / Hv;
  const int h_idx = bh - b_idx * Hv;
  const int qh = h_idx / (Hv / Hk);
  const int v_base = v_tile_idx * V_TILE;

  const int k_lo = lane * K_LOCAL;
  const int v_warp = v_base + warp_id * V_PER_WARP;

  const int state_off = (b_idx * Hv + h_idx) * V_DIM * K_DIM;
  const float* s_bh = h0_ptr + state_off;
  float S[V_PER_WARP][K_LOCAL];
#pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    const int row_off = (v_warp + jt) * K_DIM + k_lo;
    const float4& v4 = *reinterpret_cast<const float4*>(s_bh + row_off);
    S[jt][0] = v4.x;
    S[jt][1] = v4.y;
    S[jt][2] = v4.z;
    S[jt][3] = v4.w;
  }

  for (int t = 0; t < seq_len; ++t) {
    const float beta = beta_ptr[(b_idx * seq_len + t) * Hv + h_idx];

    const int qk_base = ((b_idx * seq_len + t) * Hk + qh) * K_DIM + k_lo;
    float q_arr[K_LOCAL], k_arr[K_LOCAL];
#pragma unroll
    for (int i = 0; i < K_LOCAL; ++i) {
      q_arr[i] = cast_to<float>(q_ptr[qk_base + i]);
      k_arr[i] = cast_to<float>(k_ptr[qk_base + i]);
    }

    const int v_base_t = ((b_idx * seq_len + t) * Hv + h_idx) * V_DIM + v_warp;
    float v_mine = 0.f;
    if (lane < V_PER_WARP) {
      v_mine = cast_to<float>(v_ptr[v_base_t + lane]);
    }

    // Decay the state: g is [B, T, Hv] or [B, T, Hv, Dk]
    if constexpr (vectorized_g) {
      const float4 g4 = *reinterpret_cast<const float4*>(
          g_ptr + ((b_idx * seq_len + t) * Hv + h_idx) * K_DIM + k_lo);
      const float g_arr[K_LOCAL] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
      for (int jt = 0; jt < V_PER_WARP; ++jt) {
#pragma unroll
        for (int s = 0; s < K_LOCAL; ++s) {
          S[jt][s] *= g_arr[s];
        }
      }
    } else {
      const float g = g_ptr[(b_idx * seq_len + t) * Hv + h_idx];
#pragma unroll
      for (int jt = 0; jt < V_PER_WARP; ++jt) {
#pragma unroll
        for (int s = 0; s < K_LOCAL; ++s) {
          S[jt][s] *= g;
        }
      }
    }

    // kS[jt] = sum_k k[k] * S[jt,k]
    float kS[V_PER_WARP];
#pragma unroll
    for (int jt = 0; jt < V_PER_WARP; ++jt) {
      float local = 0.f;
#pragma unroll
      for (int s = 0; s < K_LOCAL; ++s) {
        local += k_arr[s] * S[jt][s];
      }
      kS[jt] = cg::reduce(warp, local, cg::plus<float>{});
    }

#pragma unroll
    for (int jt = 0; jt < V_PER_WARP; ++jt) {
      const float v_jt = warp.shfl(v_mine, jt);
      const float r = v_jt - kS[jt];
#pragma unroll
      for (int s = 0; s < K_LOCAL; ++s) {
        S[jt][s] += beta * k_arr[s] * r;
      }
    }

    // out[jt] = sum_k q[k] * S_new[jt,k]
#pragma unroll
    for (int jt = 0; jt < V_PER_WARP; ++jt) {
      float local = 0.f;
#pragma unroll
      for (int s = 0; s < K_LOCAL; ++s) {
        local += q_arr[s] * S[jt][s];
      }
      const float o = cg::reduce(warp, local, cg::plus<float>{});
      if (lane == jt) {
        out[v_base_t + jt] = cast_to<DataType>(o);
      }
    }
  }

  float* ns_bh = new_state + state_off;
#pragma unroll
  for (int jt = 0; jt < V_PER_WARP; ++jt) {
    const float4 nv = make_float4(S[jt][0], S[jt][1], S[jt][2], S[jt][3]);
    *reinterpret_cast<float4*>(ns_bh + (v_warp + jt) * K_DIM + k_lo) = nv;
  }
}

} // namespace cu

namespace {

array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& encoder,
    Stream s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    encoder.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

} // namespace

namespace fast {

bool GatedDeltaUpdate::use_fallback(
    int Hk,
    int Dk,
    int Hv,
    int Dv,
    bool has_mask,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (has_mask) {
    return true;
  }
  if (Dk != 128 || Dv != 128) {
    return true;
  }
  return Hv % Hk;
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("GatedDeltaUpdate::eval_gpu");

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto q = ensure_row_contiguous(inputs[0], encoder, s);
  auto k = ensure_row_contiguous(inputs[1], encoder, s);
  auto v = ensure_row_contiguous(inputs[2], encoder, s);
  auto g = ensure_row_contiguous(inputs[3], encoder, s);
  auto beta = ensure_row_contiguous(inputs[4], encoder, s);
  auto h0 = ensure_row_contiguous(inputs[5], encoder, s);

  auto& out = outputs[0];
  auto& hf = outputs[1];

  out.set_data(cu::malloc_async(out.nbytes(), encoder));
  hf.set_data(cu::malloc_async(hf.nbytes(), encoder));

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = k.shape(2);
  int Dk = k.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_input_array(g);
  encoder.set_input_array(beta);
  encoder.set_input_array(h0);
  encoder.set_output_array(out);
  encoder.set_output_array(hf);

  constexpr int N_V = 16;
  constexpr int V_PER_WARP = 8;
  constexpr int V_TILE = 8;
  constexpr int K_LOCAL = 4;
  constexpr uint32_t BLOCK_THREADS = 32;
  dispatch_float_types(out.dtype(), "gated_delta_update", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_bool(g.ndim() == 4, [&](auto vectorized_g) {
      encoder.add_kernel_node(
          &cu::gated_delta_seq<
              DataType,
              vectorized_g.value,
              N_V,
              V_PER_WARP,
              V_TILE,
              K_LOCAL,
              BLOCK_THREADS>,
          dim3{uint32_t(B * Hv * N_V)},
          dim3{BLOCK_THREADS},
          gpu_ptr<DataType>(q),
          gpu_ptr<DataType>(k),
          gpu_ptr<DataType>(v),
          gpu_ptr<float>(g),
          gpu_ptr<float>(beta),
          gpu_ptr<float>(h0),
          gpu_ptr<DataType>(out),
          gpu_ptr<float>(hf),
          Hk,
          Hv,
          T);
    });
  });
}

} // namespace fast

} // namespace mlx::core
