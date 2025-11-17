// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"

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
};

auto& sdpa_cache() {
  static LRUBytesKeyCache<SDPACacheKey, fe::graph::Graph> cache(
      "MLX_CUDA_SDPA_CACHE_SIZE", /* default_capacity */ 128);
  return cache;
}

enum UIDS {
  Q,
  K,
  V,
  SCALE,
  O,
};

fe::graph::Graph build_sdpa_graph(
    cudnnHandle_t handle,
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    const array& o) {
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

  auto sdpa_options = fe::graph::SDPA_attributes()
                          .set_name("sdpa_cudnn")
                          .set_attn_scale(scale)
                          .set_causal_mask(do_causal)
                          .set_generate_stats(false);

  auto [o_, _] = graph.sdpa(q_, k_, v_, sdpa_options);
  o_->set_output(true);
  set_tensor_attrs(o_, O, o);

  CHECK_CUDNN_FE_ERROR(graph.validate());
  CHECK_CUDNN_FE_ERROR(graph.build_operation_graph(handle));
  CHECK_CUDNN_FE_ERROR(graph.create_execution_plans({fe::HeurMode_t::A}));
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_FE_ERROR(graph.check_support(handle));
  CHECK_CUDNN_FE_ERROR(graph.build_plans(handle));

  return graph;
}

} // namespace

bool supports_sdpa_cudnn(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
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

  if (has_mask) {
    // TODO: Support array masks.
    if (!do_causal) {
      return false;
    }
    // FIXME: Causal mask generates wrong results when L_Q != L_K.
    if (q.shape(2) != k.shape(2)) {
      return false;
    }
  }

  // Only use cuDNN for prefilling.
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
    bool do_causal,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  // TODO: Handle donation.
  // TODO: Make O use same memory layout with Q.
  o.set_data(cu::malloc_async(o.nbytes(), encoder));

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  encoder.set_output_array(o);

  auto handle = encoder.device().cudnn_handle();
  cudnnSetStream(handle, encoder.stream());

  // Search cache.
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
  };
  auto it = sdpa_cache().find(cache_key);
  if (it == sdpa_cache().end()) {
    it =
        sdpa_cache()
            .emplace(cache_key, build_sdpa_graph(handle, q, k, v, do_causal, o))
            .first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {Q, const_cast<void*>(gpu_ptr<void>(q))},
      {K, const_cast<void*>(gpu_ptr<void>(k))},
      {V, const_cast<void*>(gpu_ptr<void>(v))},
      {SCALE, &scale},
      {O, gpu_ptr<void>(o)}};

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

  CudaGraph cuda_graph(encoder.device());
  CHECK_CUDNN_FE_ERROR(graph.populate_cuda_graph(
      handle, variant_pack, workspace_ptr, cuda_graph));
  encoder.add_graph_node(cuda_graph);
}

// Defined in scaled_dot_product_attention.cu file.
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal);
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
    Stream s) {
  if (detail::in_grad_tracing()) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }

  return !supports_sdpa_vector(q, k, v, has_mask, has_arr_mask, do_causal) &&
      !supports_sdpa_cudnn(q, k, v, has_mask, do_causal, s);
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  bool has_mask = inputs.size() - has_sinks_ > 3;
  bool has_arr_mask = has_mask && !do_causal_;

  if (supports_sdpa_vector(q, k, v, has_mask, has_arr_mask, do_causal_)) {
    if (has_sinks_) {
      sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
    } else {
      sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
    }
  } else {
    sdpa_cudnn(q, k, v, scale_, out, do_causal_, s);
  }
}

} // namespace fast

} // namespace mlx::core
