// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace fe = cudnn_frontend;

namespace {

#define CHECK_CUDNN_FE_ERROR(cmd)                                    \
  do {                                                               \
    auto error = cmd;                                                \
    if (!error.is_good()) {                                          \
      throw std::runtime_error(                                      \
          fmt::format("{} failed: {}.", #cmd, error.get_message())); \
    }                                                                \
  } while (0)

std::vector<int64_t> normalized_strides(const array& x) {
  std::vector<int64_t> strides(x.strides().begin(), x.strides().end());
  if (std::all_of(
          strides.begin(), strides.end(), [](int64_t s) { return s == 0; })) {
    strides.back() = 1;
    return strides;
  }
  if (!x.flags().row_contiguous || x.ndim() < 2) {
    return strides;
  }
  for (int i = x.ndim() - 2; i >= 0; --i) {
    if (x.shape(i) == 1) {
      strides[i] = x.shape(i + 1) * strides[i + 1];
    }
  }
  return strides;
}

void set_tensor_attrs(
    std::shared_ptr<fe::graph::Tensor_attributes>& tensor,
    int64_t uid,
    const array& x) {
  tensor->set_uid(uid)
      .set_dim({x.shape().begin(), x.shape().end()})
      .set_stride(normalized_strides(x));
}

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
  cudnnDataType_t cudnn_dtype;
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
  static LRUBytesKeyCache<SDPACacheKey, fe::graph::Graph> cache(
      "MLX_CUDA_SDPA_CACHE_SIZE", /* default_capacity */ 16);
  return cache;
}

auto& sdpa_backward_cache() {
  static LRUBytesKeyCache<SDPACacheKey, fe::graph::Graph> cache(
      "MLX_CUDA_SDPA_BACKWARD_CACHE_SIZE", /* default_capacity */ 16);
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

fe::graph::Graph build_sdpa_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const std::optional<array>& mask_arr,
    bool output_logsumexp,
    const array& o,
    const array& stats) {
  auto dtype = fe::DataType_t::HALF;
  if (q.dtype() == bfloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }

  fe::graph::Graph graph;
  graph.set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto q_ = graph.tensor(fe::graph::Tensor_attributes().set_name("Q"));
  auto k_ = graph.tensor(fe::graph::Tensor_attributes().set_name("K"));
  auto v_ = graph.tensor(fe::graph::Tensor_attributes().set_name("V"));
  set_tensor_attrs(q_, Q, q);
  set_tensor_attrs(k_, K, k);
  set_tensor_attrs(v_, V, v);

  auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("Scale")
                                .set_uid(SCALE)
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

  auto options = fe::graph::SDPA_attributes()
                     .set_name("sdpa_cudnn")
                     .set_attn_scale(scale)
                     .set_causal_mask(do_causal)
                     .set_generate_stats(output_logsumexp);
  if (mask_arr) {
    auto bias_ = graph.tensor(fe::graph::Tensor_attributes().set_name("BIAS"));
    set_tensor_attrs(bias_, BIAS, *mask_arr);
    options.set_bias(bias_);
  }

  auto [o_, stats_] = graph.sdpa(q_, k_, v_, options);
  o_->set_output(true);
  set_tensor_attrs(o_, O, o);
  if (output_logsumexp) {
    stats_->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    set_tensor_attrs(stats_, STATS, stats);
  }

  CHECK_CUDNN_FE_ERROR(graph.validate());
  CHECK_CUDNN_FE_ERROR(graph.build_operation_graph(handle));
  CHECK_CUDNN_FE_ERROR(graph.create_execution_plans({fe::HeurMode_t::A}));
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.check_support(handle));
  CHECK_CUDNN_FE_ERROR(graph.build_plans(handle));

  return graph;
}

fe::graph::Graph build_sdpa_backward_graph(
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
  auto dtype = fe::DataType_t::HALF;
  if (q.dtype() == bfloat16) {
    dtype = fe::DataType_t::BFLOAT16;
  }

  fe::graph::Graph graph;
  graph.set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto q_ = graph.tensor(fe::graph::Tensor_attributes().set_name("Q"));
  auto k_ = graph.tensor(fe::graph::Tensor_attributes().set_name("K"));
  auto v_ = graph.tensor(fe::graph::Tensor_attributes().set_name("V"));
  auto o_ = graph.tensor(fe::graph::Tensor_attributes().set_name("O"));
  auto d_o_ = graph.tensor(fe::graph::Tensor_attributes().set_name("D_O"));
  auto stats_ = graph.tensor(fe::graph::Tensor_attributes().set_name("STATS"));
  set_tensor_attrs(q_, Q, q);
  set_tensor_attrs(k_, K, k);
  set_tensor_attrs(v_, V, v);
  set_tensor_attrs(o_, O, o);
  set_tensor_attrs(d_o_, D_O, d_o);
  set_tensor_attrs(stats_, STATS, stats);
  stats_->set_data_type(fe::DataType_t::FLOAT);

  auto scale = graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("Scale")
                                .set_uid(SCALE)
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

  auto options = fe::graph::SDPA_backward_attributes()
                     .set_name("sdpa_backward_cudnn")
                     .set_attn_scale(scale)
                     .set_causal_mask(do_causal);
  if (mask_arr) {
    auto bias_ = graph.tensor(fe::graph::Tensor_attributes().set_name("BIAS"));
    set_tensor_attrs(bias_, BIAS, *mask_arr);
    options.set_bias(bias_);
  }

  auto [d_q_, d_k_, d_v_] =
      graph.sdpa_backward(q_, k_, v_, o_, d_o_, stats_, options);
  d_q_->set_output(true);
  d_k_->set_output(true);
  d_v_->set_output(true);
  set_tensor_attrs(d_q_, D_Q, d_q);
  set_tensor_attrs(d_k_, D_K, d_k);
  set_tensor_attrs(d_v_, D_V, d_v);

  CHECK_CUDNN_FE_ERROR(graph.validate());
  CHECK_CUDNN_FE_ERROR(graph.build_operation_graph(handle));
  CHECK_CUDNN_FE_ERROR(graph.create_execution_plans({fe::HeurMode_t::A}));
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.check_support(handle));
  CHECK_CUDNN_FE_ERROR(graph.build_plans(handle));

  return graph;
}

void execute_graph(
    cu::CommandEncoder& encoder,
    cudnnHandle_t handle,
    fe::graph::Graph& graph,
    std::unordered_map<int64_t, void*>& variant_pack) {
  int64_t workspace_size = 0;
  CHECK_CUDNN_FE_ERROR(graph.get_workspace_size(workspace_size));
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    array workspace(
        cu::malloc_async(workspace_size, encoder),
        {static_cast<int>(workspace_size)},
        uint8);
    encoder.add_temporary(workspace);
    workspace_ptr = gpu_ptr<void>(workspace);
  }

  cudnnSetStream(handle, encoder.stream());

  CudaGraph cuda_graph(encoder.device());
  CHECK_CUDNN_FE_ERROR(graph.populate_cuda_graph(
      handle, variant_pack, workspace_ptr, cuda_graph));
  encoder.add_graph_node(cuda_graph);
}

} // namespace

bool supports_sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    Stream s) {
  static bool enabled = env::get_var("MLX_CUDA_USE_CUDNN_SPDA", 1);
  if (!enabled) {
    return false;
  }

  // cuDNN SDPA requires Ampere and later.
  if (cu::device(s.device).compute_capability_major() < 8) {
    return false;
  }

  // Only use cuDNN for prefilling and training.
  if (q.shape(2) != k.shape(2)) {
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
      {Q, const_cast<void*>(gpu_ptr<void>(q))},
      {K, const_cast<void*>(gpu_ptr<void>(k))},
      {V, const_cast<void*>(gpu_ptr<void>(v))},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)}};
  if (mask_arr) {
    variant_pack[BIAS] = const_cast<void*>(gpu_ptr<void>(*mask_arr));
  }
  if (output_logsumexp) {
    variant_pack[STATS] = gpu_ptr<void>(stats);
  }

  execute_graph(encoder, handle, graph, variant_pack);
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
      {Q, const_cast<void*>(gpu_ptr<void>(q))},
      {K, const_cast<void*>(gpu_ptr<void>(k))},
      {V, const_cast<void*>(gpu_ptr<void>(v))},
      {SCALE, &scale},
      {O, const_cast<void*>(gpu_ptr<void>(o))},
      {STATS, const_cast<void*>(gpu_ptr<void>(stats))},
      {D_O, const_cast<void*>(gpu_ptr<void>(d_o))},
      {D_Q, gpu_ptr<void>(d_q)},
      {D_K, gpu_ptr<void>(d_k)},
      {D_V, gpu_ptr<void>(d_v)}};
  if (mask_arr) {
    variant_pack[BIAS] = const_cast<void*>(gpu_ptr<void>(*mask_arr));
  }

  execute_graph(encoder, handle, graph, variant_pack);
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
      !supports_sdpa_cudnn(q, k, v, s);
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
