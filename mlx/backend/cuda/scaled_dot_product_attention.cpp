// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

array prepare_sdpa_input(const array& x, Stream s) {
  // SDPA kernel's requirements on inputs:
  // 1. last dim's stride be 1;
  // 2. pointer be aligned.
  if (x.strides(-1) != 1 || get_alignment(x) < 16) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

void malloc_with_same_layout(
    cu::CommandEncoder& encoder,
    array& o,
    const array& q) {
  if (q.flags().row_contiguous) {
    o.set_data(cu::malloc_async(o.nbytes(), encoder));
    return;
  }
  // fill_order = argsort(q.strides())
  Shape fill_order(q.ndim());
  std::iota(fill_order.begin(), fill_order.end(), 0);
  std::stable_sort(
      fill_order.begin(), fill_order.end(), [&q](int idx1, int idx2) {
        auto s1 = q.strides(idx1) > 0 ? q.strides(idx1) : 1;
        auto s2 = q.strides(idx2) > 0 ? q.strides(idx2) : 1;
        return s1 < s2;
      });
  // Generate o_strides with fill_order
  Strides o_strides(q.ndim());
  int64_t stride = 1;
  for (int i : fill_order) {
    o_strides[i] = stride;
    stride *= o.shape(i);
  }
  // o is a transposed contiguous array
  o.set_data(
      cu::malloc_async(o.nbytes(), encoder),
      o.size(),
      o_strides,
      {true, false, false});
}

constexpr int QKV_NDIM = 4;

struct SDPACacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  std::array<int, QKV_NDIM> q_shape;
  std::array<int, QKV_NDIM> k_shape;
  std::array<int, QKV_NDIM> v_shape;
  std::array<int64_t, QKV_NDIM> q_strides;
  std::array<int64_t, QKV_NDIM> k_strides;
  std::array<int64_t, QKV_NDIM> v_strides;
  bool do_causal;
  std::array<int, QKV_NDIM> mask_shape;
  std::array<int64_t, QKV_NDIM> mask_strides;
  bool output_logsumexp;
};

inline BytesKey<SDPACacheKey> build_sdpa_cache_key(
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp = true) {
  BytesKey<SDPACacheKey> cache_key;
  cache_key.pod = {
      encoder.device().cuda_device(),
      dtype_to_cudnn_type(q.dtype()),
      vector_key<QKV_NDIM>(q.shape()),
      vector_key<QKV_NDIM>(k.shape()),
      vector_key<QKV_NDIM>(v.shape()),
      vector_key<QKV_NDIM>(q.strides()),
      vector_key<QKV_NDIM>(k.strides()),
      vector_key<QKV_NDIM>(v.strides()),
      do_causal,
      {},
      {},
      output_logsumexp,
  };
  if (mask_arr) {
    cache_key.pod.mask_shape = vector_key<QKV_NDIM>(mask_arr->shape());
    cache_key.pod.mask_strides = vector_key<QKV_NDIM>(mask_arr->strides());
  }
  return cache_key;
}

auto& sdpa_cache() {
  static LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

auto& sdpa_backward_cache() {
  static LRUBytesKeyCache<SDPACacheKey, DnnGraph> cache(
      "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE", /* default_capacity */ 64);
  return cache;
}

enum UIDS {
  Q,
  K,
  V,
  SCALE,
  BIAS,
  O,
  STATS,
  // Backward graph:
  D_Q,
  D_K,
  D_V,
  D_O,
};

DnnGraph build_sdpa_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp,
    const array& o,
    const array& stats) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);

  auto options = fe::graph::SDPA_attributes()
                     .set_name("sdpa_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32))
                     .set_generate_stats(output_logsumexp);
  if (do_causal) {
    if (q.shape(2) > k.shape(2)) {
      options.set_causal_mask(do_causal);
    } else {
      options.set_causal_mask_bottom_right(do_causal);
    }
  }
  if (mask_arr) {
    options.set_bias(graph.tensor("BIAS", BIAS, *mask_arr));
  }

  auto [o_, stats_] = graph.sdpa(q_, k_, v_, options);
  graph.tensor(o_, O, o)->set_output(true);
  if (output_logsumexp) {
    graph.tensor(stats_, STATS, stats)->set_output(true);
  }

  CHECK_CUDNN_FE_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
}

DnnGraph build_sdpa_backward_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const array& o,
    const array& d_o,
    const array& stats,
    array& d_q,
    array& d_k,
    array& d_v) {
  DnnGraph graph(handle, q.dtype());

  auto q_ = graph.tensor("Q", Q, q);
  auto k_ = graph.tensor("K", K, k);
  auto v_ = graph.tensor("V", V, v);
  auto o_ = graph.tensor("O", O, o);
  auto d_o_ = graph.tensor("D_O", D_O, d_o);
  auto stats_ = graph.tensor("STATS", STATS, stats);

  auto options = fe::graph::SDPA_backward_attributes()
                     .set_name("sdpa_backward_cudnn")
                     .set_attn_scale(graph.scalar("Scale", SCALE, float32));
  if (do_causal) {
    if (q.shape(2) > k.shape(2)) {
      options.set_causal_mask(do_causal);
    } else {
      options.set_causal_mask_bottom_right(do_causal);
    }
  }
  if (mask_arr) {
    options.set_bias(graph.tensor("BIAS", BIAS, *mask_arr));
  }

  auto [d_q_, d_k_, d_v_] =
      graph.sdpa_backward(q_, k_, v_, o_, d_o_, stats_, options);
  graph.tensor(d_q_, D_Q, d_q)->set_output(true);
  graph.tensor(d_k_, D_K, d_k)->set_output(true);
  graph.tensor(d_v_, D_V, d_v)->set_output(true);

  CHECK_CUDNN_FE_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.build());
  return graph;
}

} // namespace

bool supports_sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    Stream s) {
  static bool enabled = env::get_var("MLX_CUDA_USE_CUDNN_SPDA", 1);
  if (!enabled) {
    return false;
  }

  // cuDNN SDPA requires Ampere and later.
  if (cu::device(s.device).compute_capability_major() < 8) {
    return false;
  }

  // Only use cuDNN for prefilling (T_q > 1) and training (T_q == T_kv).
  if ((q.shape(2) == 1) && (q.shape(2) != k.shape(2))) {
    return false;
  }

  // D_qk and D_v must be a multiple of 8 with maximum value 128.
  if ((q.shape(-1) % 8 != 0) || (q.shape(-1) > 128) || (v.shape(-1) % 8 != 0) ||
      (v.shape(-1) > 128)) {
    return false;
  }

  Dtype dtype = q.dtype();
  return dtype == float16 || dtype == bfloat16;
}

void sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    array& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = encoder.device().cudnn_handle();

  malloc_with_same_layout(encoder, o, q);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(o);
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  if (output_logsumexp) {
    stats.set_data(cu::malloc_async(stats.nbytes(), encoder));
    encoder.set_output_array(stats);
  }

  // Search cache.
  auto cache_key = build_sdpa_cache_key(
      encoder, q, k, v, do_causal, mask_arr, output_logsumexp);
  auto it = sdpa_cache().find(cache_key);
  if (it == sdpa_cache().end()) {
    auto graph = build_sdpa_graph(
        handle, q, k, v, do_causal, mask_arr, output_logsumexp, o, stats);
    it = sdpa_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)}};
  if (mask_arr) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_arr);
  }
  if (output_logsumexp) {
    variant_pack[STATS] = gpu_ptr<void>(stats);
  }

  CHECK_CUDNN_FE_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

void sdpa_backward_cudnn(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    const array& o,
    const array& stats,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const array& d_o,
    array& d_q,
    array& d_k,
    array& d_v,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  auto handle = encoder.device().cudnn_handle();

  malloc_with_same_layout(encoder, d_q, q);
  malloc_with_same_layout(encoder, d_k, k);
  malloc_with_same_layout(encoder, d_v, v);

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_input_array(o);
  encoder.set_input_array(stats);
  encoder.set_input_array(d_o);
  encoder.set_output_array(d_q);
  encoder.set_output_array(d_k);
  encoder.set_output_array(d_v);
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }

  // Search cache.
  auto cache_key = build_sdpa_cache_key(encoder, q, k, v, do_causal, mask_arr);
  auto it = sdpa_backward_cache().find(cache_key);
  if (it == sdpa_backward_cache().end()) {
    auto graph = build_sdpa_backward_graph(
        handle, q, k, v, do_causal, mask_arr, o, d_o, stats, d_q, d_k, d_v);
    it = sdpa_backward_cache().emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, gpu_ptr<void>(q)},
      {K, gpu_ptr<void>(k)},
      {V, gpu_ptr<void>(v)},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)},
      {STATS, gpu_ptr<void>(stats)},
      {D_O, gpu_ptr<void>(d_o)},
      {D_Q, gpu_ptr<void>(d_q)},
      {D_K, gpu_ptr<void>(d_k)},
      {D_V, gpu_ptr<void>(d_v)}};
  if (mask_arr) {
    variant_pack[BIAS] = gpu_ptr<void>(*mask_arr);
  }

  CHECK_CUDNN_FE_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

// Defined in scaled_dot_product_attention.cu file.
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

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }

  return !supports_sdpa_vector(
             q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp) &&
      !supports_sdpa_cudnn(q, k, v, do_causal, s);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  auto& out = outputs[0];
  auto& stats = outputs[1];
  bool has_mask = inputs.size() - has_sinks_ > 3;
  bool has_arr_mask = has_mask && !do_causal_;

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }

  if (supports_sdpa_vector(
          q, k, v, has_mask, has_arr_mask, do_causal_, output_logsumexp_)) {
    if (has_sinks_) {
      sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
    } else {
      sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
    }
  } else {
    sdpa_cudnn(
        q,
        k,
        v,
        scale_,
        out,
        stats,
        do_causal_,
        mask_arr,
        output_logsumexp_,
        s);
  }
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // The frontend adds a padding mask when sequence length is not a multiple of
  // tile size.
  if (q.shape(2) % 128 != 0) {
    return true;
  }
  return s.device == Device::cpu;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("ScaledDotProductAttentionVJP::eval_gpu");

  auto& s = stream();

  assert(inputs.size() >= 6);
  int primals_size = inputs.size() - 3;
  bool has_arr_mask = primals_size > 3 + has_sinks_;

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array o = prepare_sdpa_input(inputs[primals_size], s);
  array stats = prepare_sdpa_input(inputs[primals_size + 1], s);
  array d_o = prepare_sdpa_input(inputs[primals_size + 2], s);

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }

  assert(outputs.size() == 3);
  auto& d_q = outputs[0];
  auto& d_k = outputs[1];
  auto& d_v = outputs[2];

  sdpa_backward_cudnn(
      q, k, v, scale_, o, stats, do_causal_, mask_arr, d_o, d_q, d_k, d_v, s);
}

} // namespace fast

} // namespace mlx::core
