// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"

#include <nvtx3/nvtx3.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

#define PRAGMA_LOOP_UNROLL #pragma unroll

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

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_1pass(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const T* sinks,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = BN * int(params.K_strides[2]);
  const int inner_v_stride = BN * int(params.V_strides[2]);

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * M_LOG2E;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  // Read the query and 0 the output accumulator
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q[v_per_thread * lane_idx + i]);
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    if (use_key) {
      // Read the key
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        k[j] = K[v_per_thread * lane_idx + j];
      }

      // Compute the i-th score
      U score = 0.f;
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        score += q[j] * k[j];
      }

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      U new_max = max(max_score, score);
      bool is_neg_inf = new_max == -INFINITY;
      U factor = is_neg_inf ? 1 : exp2f(max_score - new_max);
      U exp_score = is_neg_inf ? 0 : exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor +
            exp_score * static_cast<U>(V[v_per_thread * lane_idx + j]);
      }
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }
  block.sync();

  max_score = max_scores[lane_idx];
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score =
      cg::reduce(warp, sum_exp_scores[lane_idx] * factor, cg::plus<U>());
  sum_exp_score = __frcp_rn(sum_exp_score);

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_2pass_1(
    const T* Q,
    const T* K,
    const T* V,
    const T* sinks,
    float* partials,
    float* sums,
    float* maxs,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int blocks = 32;

  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = blocks * BN * int(params.K_strides[2]);
  const int inner_v_stride = blocks * BN * int(params.V_strides[2]);

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * 1.44269504089f;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z / blocks;
  const int block_idx = blockIdx.z % blocks;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = block_idx * BN + warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s + // Sequence
      block_idx; // Block

  partials += p_offset * D;
  sums += p_offset;
  maxs += p_offset;

  // Read the query and 0 the output accumulator
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q[v_per_thread * lane_idx + i]);
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0 && block_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += blocks * BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    if (use_key) {
      // Read the key
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        k[j] = K[v_per_thread * lane_idx + j];
      }

      // Compute the i-th score
      U score = 0.f;
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        score += q[j] * k[j];
      }

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      U new_max = max(max_score, score);
      bool is_neg_inf = new_max == -INFINITY;
      U factor = is_neg_inf ? 1 : exp2f(max_score - new_max);
      U exp_score = is_neg_inf ? 0 : exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor +
            exp_score * static_cast<U>(V[v_per_thread * lane_idx + j]);
      }
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }

  block.sync();

  max_score = (lane_idx < BN) ? max_scores[lane_idx] : -1e9;
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score = (lane_idx < BN) ? sum_exp_scores[lane_idx] : 0.f;
  sum_exp_score = cg::reduce(warp, sum_exp_score * factor, cg::plus<U>());

  // Write the sum and new max
  if (warp_idx == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  auto ff = exp2f(max_scores[warp_idx] - new_max);
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[warp_idx][lane_idx] = o[i] * ff;
    block.sync();

    if (warp_idx == 0) {
      U ot = outputs[0][lane_idx];
      PRAGMA_LOOP_UNROLL
      for (int j = 1; j < BN; j++) {
        ot += outputs[j][lane_idx];
        warp.sync();
      }
      o[i] = ot;
    }
    block.sync();
  }

  if (warp_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      partials[v_per_thread * lane_idx + i] = o[i];
    }
  }
}

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_2pass_2(
    const float* partials,
    const float* sums,
    const float* maxs,
    T* O,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int blocks = 32;

  constexpr int v_per_thread = D / BD;

  typedef float U;

  U o[v_per_thread];
  __shared__ U outputs[BN][BD + 1];

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int q_seq_idx = blockIdx.y;

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s; // Sequence

  partials += p_offset * D + warp_idx * D;
  sums += p_offset;
  maxs += p_offset;

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  U max_score = maxs[lane_idx];
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  U sum_exp_score = cg::reduce(warp, sums[lane_idx] * factor, cg::plus<U>());
  sum_exp_score = __frcp_rn(sum_exp_score);

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = partials[v_per_thread * lane_idx + i];
  }

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
    }
  }
}

} // namespace cu

namespace {

template <typename F>
void dispatch_headdim(int n, F&& f) {
  switch (n) {
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 96:
      f(std::integral_constant<int, 96>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

void sdpa_vector_1pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  encoder.set_output_array(o);

  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  dim3 grid_dim(params.H, params.qL, params.B);
  dim3 block_dim(1024, 1, 1);

  dispatch_float_types(o.dtype(), "kernel_sdpav_1pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        auto kernel =
            cu::kernel_sdpav_1pass<DataType, do_causal.value, headdim.value>;
        encoder.add_kernel_node(
            kernel,
            grid_dim,
            block_dim,
            0,
            q.data<DataType>(),
            k.data<DataType>(),
            v.data<DataType>(),
            o.data<DataType>(),
            sinks ? (*sinks).data<DataType>() : nullptr,
            params);
      });
    });
  });
}

void sdpa_vector_2pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  // Allocate the intermediates
  int blocks = 32;

  Shape intermediate_shape;
  intermediate_shape.reserve(o.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), o.shape().begin(), o.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(o.shape().back());

  array intermediate(intermediate_shape, float32, nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});

  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));

  encoder.add_temporary(intermediate);
  encoder.add_temporary(sums);
  encoder.add_temporary(maxs);

  dispatch_float_types(o.dtype(), "kernel_sdpav_2pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        {
          auto kernel = cu::
              kernel_sdpav_2pass_1<DataType, do_causal.value, headdim.value>;

          encoder.set_input_array(q);
          encoder.set_input_array(k);
          encoder.set_input_array(v);
          if (sinks) {
            encoder.set_input_array(*sinks);
          }

          encoder.set_output_array(intermediate);
          encoder.set_output_array(sums);
          encoder.set_output_array(maxs);

          dim3 grid_dim(params.H, params.qL, params.B * 32);
          dim3 block_dim(8 * 32, 1, 1);

          encoder.add_kernel_node(
              kernel,
              grid_dim,
              block_dim,
              0,
              q.data<DataType>(),
              k.data<DataType>(),
              v.data<DataType>(),
              sinks ? (*sinks).data<DataType>() : nullptr,
              intermediate.data<float>(),
              sums.data<float>(),
              maxs.data<float>(),
              params);
        }

        {
          auto kernel = cu::
              kernel_sdpav_2pass_2<DataType, do_causal.value, headdim.value>;

          encoder.set_input_array(intermediate);
          encoder.set_input_array(sums);
          encoder.set_input_array(maxs);
          encoder.set_output_array(o);

          dim3 grid_dim(params.H, params.qL, params.B);
          dim3 block_dim(1024, 1, 1);

          encoder.add_kernel_node(
              kernel,
              grid_dim,
              block_dim,
              0,
              intermediate.data<float>(),
              sums.data<float>(),
              maxs.data<float>(),
              o.data<DataType>(),
              params);
        }
      });
    });
  });
}

void sdpa_vector_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks) {
  int kL = k.shape(2);

  if (kL > 1024) {
    return sdpa_vector_2pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, sinks);
  } else {
    return sdpa_vector_1pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, sinks);
  }
}

} // namespace

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  if (detail::in_grad_tracing()) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);

  const bool sdpa_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128);

  const bool supported_vector_config =
      sdpa_supported_head_dim && query_sequence_length < 4;

  const bool supported_config = supported_vector_config;

  return has_arr_mask || !supported_config;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = out;

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(inputs.size());
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }

  // We are in vector mode ie single query
  if (q_pre.shape(2) < 4) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    const auto& q = copy_unless(q_copy_unless, q_pre);
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      int64_t str_oD = 1;
      int64_t str_oH = o.shape(3);
      int64_t str_oL = o.shape(1) * str_oH;
      int64_t str_oB = o.shape(2) * str_oL;

      array::Flags flags{
          /* bool contiguous = */ 1,
          /* bool row_contiguous = */ o.shape(2) == 1,
          /* bool col_contiguous = */ o.size() == o.shape(3),
      };

      o.set_data(
          allocator::malloc(o.nbytes()),
          o.size(),
          {str_oB, str_oH, str_oL, str_oD},
          flags);
    }

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    return sdpa_vector_fallback(
        s, encoder, q, k, v, scale_, o, do_causal_, sinks);
  }

  // Full attention mode should never reach here
  else {
    throw std::runtime_error("Doesn't support matrix yet.");
  }
}

} // namespace fast

} // namespace mlx::core
