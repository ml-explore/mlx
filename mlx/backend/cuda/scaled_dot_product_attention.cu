// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

// cudnn_frontend.h redefines this macro.
#undef CHECK_CUDA_ERROR

#include <cudnn_frontend.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace fe = cudnn_frontend;

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

  const U scale_log2 = params.scale * 1.44269504089f;

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
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

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

  U max_score = -1e9;
  U sum_exp_score = 0.f;

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
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

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
    bool do_causal_ = false) {
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
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
    dispatch_bool(do_causal_, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        auto kernel = cu::kernel_sdpav_1pass<DataType, do_causal(), headdim()>;
        encoder.add_kernel_node(
            kernel,
            grid_dim,
            block_dim,
            0,
            q.data<DataType>(),
            k.data<DataType>(),
            v.data<DataType>(),
            o.data<DataType>(),
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
    bool do_causal_ = false) {
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
    dispatch_bool(do_causal_, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        {
          auto kernel =
              cu::kernel_sdpav_2pass_1<DataType, do_causal(), headdim()>;

          encoder.set_input_array(q);
          encoder.set_input_array(k);
          encoder.set_input_array(v);
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
              intermediate.data<float>(),
              sums.data<float>(),
              maxs.data<float>(),
              params);
        }

        {
          auto kernel =
              cu::kernel_sdpav_2pass_2<DataType, do_causal(), headdim()>;

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
    bool do_causal_ = false) {
  int kL = k.shape(2);

  if (kL > 1024) {
    return sdpa_vector_2pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal_);
  } else {
    return sdpa_vector_1pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal_);
  }
}

struct SDPACacheKey {
  int device_id;
  fe::DataType_t cudnn_type;

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

  bool generate_stats;
  bool causal_mask;
};

auto& sdpa_cache() {
  static LRUBytesKeyCache<SDPACacheKey, std::shared_ptr<fe::graph::Graph>>
      cache(
          /* capacity */ 128);
  return cache;
}

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5

std::shared_ptr<fe::graph::Graph> get_sdpa_forward_graph(
    cu::CommandEncoder& encoder,
    const SDPACacheKey& cache_key) {
  // Check if graph has already been fully built
  if (auto it = sdpa_cache().find(cache_key); it != sdpa_cache().end()) {
    return it->second;
  }

  // Set up new graph
  auto graph = std::make_shared<fe::graph::Graph>();

  graph->set_io_data_type(cache_key.cudnn_type)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto Q = graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("Q")
          .set_uid(Q_UID)
          .set_dim({cache_key.B, cache_key.H, cache_key.qL, cache_key.D})
          .set_stride(
              {cache_key.Q_strides[0],
               cache_key.Q_strides[1],
               cache_key.Q_strides[2],
               1}));

  int h_kv = cache_key.H / cache_key.gqa_factor;
  auto K =
      graph->tensor(fe::graph::Tensor_attributes()
                        .set_name("K")
                        .set_uid(K_UID)
                        .set_dim({cache_key.B, h_kv, cache_key.kL, cache_key.D})
                        .set_stride(
                            {cache_key.K_strides[0],
                             cache_key.K_strides[1],
                             cache_key.V_strides[2],
                             1}));

  auto V =
      graph->tensor(fe::graph::Tensor_attributes()
                        .set_name("V")
                        .set_uid(V_UID)
                        .set_dim({cache_key.B, h_kv, cache_key.kL, cache_key.D})
                        .set_stride(
                            {cache_key.V_strides[0],
                             cache_key.V_strides[1],
                             cache_key.V_strides[2],
                             1}));

  auto sdpa_options = fe::graph::SDPA_attributes()
                          .set_name("flash_attention")
                          .set_is_inference(!cache_key.generate_stats)
                          .set_attn_scale(cache_key.scale);

  if (cache_key.causal_mask && cache_key.qL > 1) {
    sdpa_options.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }

  auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

  O->set_output(true)
      .set_uid(O_UID)
      .set_dim({cache_key.B, cache_key.H, cache_key.qL, cache_key.D})
      .set_stride(
          {cache_key.O_strides[0],
           cache_key.O_strides[1],
           cache_key.O_strides[2],
           1});

  if (cache_key.generate_stats) {
    Stats->set_output(true)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(STATS_UID);
  }

  // Build and Validate cudnn graph

  auto handle = encoder.device().cudnn_handle();

  // cuDNN only supports native CUDA graphs for sdpa in 9.6 or above.
  if (cudnnGetVersion() < 90600) {
    auto build_status = graph->build(handle, {fe::HeurMode_t::A});
    if (!build_status.is_good()) {
      throw std::runtime_error(
          "Unable to build cudnn graph for attention."
          " Failed with message: " +
          build_status.get_message());
    }

  } else {
    auto val_status = graph->validate();
    auto op_status = graph->build_operation_graph(handle);

    auto plan_stauts =
        graph->create_execution_plans({cudnn_frontend::HeurMode_t::A});
    if (!plan_stauts.is_good()) {
      throw std::runtime_error(
          "Unable to create exec plan for cudnn attention."
          " Failed with message: " +
          plan_stauts.get_message());
    }

    graph->select_behavior_notes(
        {cudnn_frontend::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});

    auto support_status = graph->check_support(handle);
    if (!support_status.is_good()) {
      throw std::runtime_error(
          "No cuda graph support for cudnn attention."
          " Failed with message: " +
          support_status.get_message());
    }

    auto build_status = graph->build_plans(handle);
    if (!build_status.is_good()) {
      throw std::runtime_error(
          "Unable to build cudnn graph for attention."
          " Failed with message: " +
          build_status.get_message());
    }
  }

  auto [it, _] = sdpa_cache().emplace(cache_key, graph);

  return it->second;
}

inline fe::DataType_t dtype_to_cudnn_type(Dtype dtype) {
  switch (dtype) {
    case int8:
      return fe::DataType_t::INT8;
    case int32:
      return fe::DataType_t::INT32;
    case uint8:
      return fe::DataType_t::UINT8;
    case float16:
      return fe::DataType_t::HALF;
    case bfloat16:
      return fe::DataType_t::BFLOAT16;
    case float32:
      return fe::DataType_t::FLOAT;
    case float64:
      return fe::DataType_t::DOUBLE;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in SDPA: {}.", dtype_to_string(dtype)));
  }
}

void sdpa_cudnn(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_ = false) {
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(o);

  auto cudnn_type = dtype_to_cudnn_type(q.dtype());

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);

  int qL = q.shape(2);
  int kL = k.shape(2);

  SDPACacheKey cache_key{
      /* int device_id = */ encoder.device().cuda_device(),
      /* fe::DataType_t cudnn_type = */ cudnn_type,

      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},

      /* bool generate_stats = */ false,
      /* bool causal_mask = */ do_causal_};

  auto graph = get_sdpa_forward_graph(encoder, cache_key);

  int64_t workspace_size = 0;
  auto workspace_status = graph->get_workspace_size(workspace_size);
  if (!workspace_status.is_good()) {
    throw std::runtime_error("Unable to get workspace for cudnn attention.");
  }

  array workspace(
      allocator::malloc(workspace_size), {int(workspace_size)}, uint8);
  auto workspace_ptr = workspace.data<void>();

  std::unordered_map<int64_t, void*> variant_pack = {
      {Q_UID, const_cast<void*>(q.data<void>())},
      {K_UID, const_cast<void*>(k.data<void>())},
      {V_UID, const_cast<void*>(v.data<void>())},
      {O_UID, o.data<void>()}};

  auto handle = encoder.device().cudnn_handle();
  cudnnSetStream(handle, encoder.stream());

  // cuDNN only supports native CUDA graphs for sdpa in 9.6 or above.
  if (cudnnGetVersion() < 90600) {
    auto capture = encoder.capture_context();
    auto exec_status = graph->execute(handle, variant_pack, workspace_ptr);

    if (!exec_status.is_good()) {
      capture.discard = true;
      throw std::runtime_error(
          "Unable to execute cudnn attention."
          " Failed with message: " +
          exec_status.get_message());
    }
  } else {
    cudaGraph_t cu_graph;
    cudaGraphCreate(&cu_graph, 0);

    std::unique_ptr<cudaGraph_t, void (*)(cudaGraph_t*)> graph_freer(
        &cu_graph, [](cudaGraph_t* p) { cudaGraphDestroy(*p); });

    auto cu_graph_status = graph->populate_cuda_graph(
        handle, variant_pack, workspace_ptr, cu_graph);

    if (!cu_graph_status.is_good()) {
      throw std::runtime_error(
          "Unable to add cuda graph for cudnn attention."
          " Failed with message: " +
          cu_graph_status.get_message());
    }

    encoder.add_graph_node(cu_graph);
  }

  encoder.add_temporary(workspace);
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
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);

  const bool sdpa_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128);

  return has_arr_mask || !sdpa_supported_head_dim;
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
  copies.reserve(3);
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

  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  // We are in vector mode ie single query
  if (q_pre.shape(2) <= 1) {
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

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      int64_t str_oD = 1;
      int64_t str_oH = o.shape(3);
      int64_t str_oL = o.shape(1) * str_oH;
      int64_t str_oB = o.shape(2) * str_oL;
      size_t data_size = o.shape(0) * str_oB;

      array::Flags flags{
          /* bool contiguous = */ 1,
          /* bool row_contiguous = */ 0,
          /* bool col_contiguous = */ 0,
      };

      o.set_data(
          allocator::malloc(o.nbytes()),
          data_size,
          {str_oB, str_oH, str_oL, str_oD},
          flags);
    }

    return sdpa_vector_fallback(s, encoder, q, k, v, scale_, o, do_causal_);
    // return sdpa_cudnn(s, encoder, q, k, v, scale_, o, do_causal_);
  }

  // Full attention mode
  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    return sdpa_cudnn(s, encoder, q, k, v, scale_, o, do_causal_);
  }
}

} // namespace fast

} // namespace mlx::core