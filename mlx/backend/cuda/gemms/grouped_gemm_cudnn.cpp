// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <optional>

namespace mlx::core {

namespace {

array prepare_grouped_mm_w(const array& w, Stream s) {
  // todo
  return w;
}

array prepare_grouped_mm_x(const array& x, Stream s) {
  if (x.strides(-1) != 1) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& encoder = cu::get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

constexpr int GMM_NDIM = 3;

struct GatherMMCacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  int mode; // NONE / GATHER / SCATTER
  std::array<int, GMM_NDIM> x_shape;
  std::array<int64_t, GMM_NDIM> x_strides;
  std::array<int, GMM_NDIM> w_shape;
  std::array<int64_t, GMM_NDIM> w_strides;
};

inline BytesKey<GatherMMCacheKey> build_grouped_mm_key(
    cu::CommandEncoder& encoder,
    const array& x,
    const array& w,
    int mode) {
  BytesKey<GatherMMCacheKey> key;
  key.pod.device_id = encoder.device().cuda_device();
  key.pod.cudnn_dtype = dtype_to_cudnn_type(x.dtype());
  key.pod.mode = mode;
  key.pod.x_shape = vector_key<GMM_NDIM>(x.shape());
  key.pod.x_strides = vector_key<GMM_NDIM>(x.strides());
  key.pod.w_shape = vector_key<GMM_NDIM>(w.shape());
  key.pod.w_strides = vector_key<GMM_NDIM>(w.strides());
  return key;
}

enum UIDS {
  X,
  W,
  TOKEN_OFFSETS,
  TOKEN_INDEX,
  TOKEN_KS,
  O,
};

fe::MoeGroupedMatmulMode_t grouped_mm_mode(
    const std::optional<array>& lhs_indices) {
  return lhs_indices.has_value() ? fe::MoeGroupedMatmulMode_t::GATHER
                                 : fe::MoeGroupedMatmulMode_t::NONE;
}

DnnGraph grouped_mm_graph(
    cudnnHandle_t handle,
    const array& x,
    const array& w,
    const array& offsets,
    const std::optional<array>& lhs_indices,
    int top_k,
    const array& output) {
  DnnGraph graph(handle, x.dtype());

  auto mode = grouped_mm_mode(lhs_indices);

  auto x_ = graph.tensor("X", X, x);
  auto w_ = graph.tensor("W", W, w);
  auto offsets_ = graph.tensor("TOKEN_OFFSETS", TOKEN_OFFSETS, offsets);

  auto moe_grouped_matmul_attr = fe::graph::Moe_grouped_matmul_attributes()
                                     .set_name("grouped_matmul")
                                     .set_mode(mode)
                                     .set_top_k(top_k);

  auto out_ = graph.moe_grouped_matmul(
      x_,
      w_,
      offsets_,
      lhs_indices.has_value()
          ? graph.tensor("TOKEN_INDEX", TOKEN_INDEX, *lhs_indices)
          : nullptr,
      nullptr,
      moe_grouped_matmul_attr);
  graph.tensor(out_, O, output)->set_output(true);

  CHECK_CUDNN_ERROR(graph.prepare());
  graph.select_behavior_notes(
      {fe::BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API});
  CHECK_CUDNN_ERROR(graph.build());
  return graph;
}

auto& grouped_mm_cache() {
  static thread_local LRUBytesKeyCache<GatherMMCacheKey, DnnGraph> cache(
      "MLX_CUDA_GMM_CACHE_SIZE", /* default_capacity */ 256);
  return cache;
}

// rhs_indices (per-slot expert id) always an input to calculate the offsets.
// lhs_indices selects the mode: provided -> GATHER (gather rows before matmul),
// absent -> NONE (rows already grouped by expert)
void grouped_mm_cudnn(
    const array& x,
    const array& w,
    const array& rhs_indices,
    const std::optional<array>& lhs_indices,
    int top_k,
    array& out,
    Stream s) {
  nvtx3::scoped_range r("grouped_mm_cudnn");

  auto& encoder = cu::get_command_encoder(s);
  auto handle = get_cudnn_handle(encoder.device());

  array x_c = prepare_grouped_mm_x(x, s);
  array w_c = prepare_grouped_mm_w(w, s);

  int group_count = w_c.shape(0);
  array token_offsets = compute_token_offset(rhs_indices, group_count, encoder);

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  encoder.set_input_array(x_c);
  encoder.set_input_array(w_c);
  encoder.set_input_array(token_offsets);
  if (lhs_indices.has_value()) {
    encoder.set_input_array(*lhs_indices);
  }
  encoder.set_output_array(out);

  int mode = static_cast<int>(grouped_mm_mode(lhs_indices));

  auto cache_key = build_grouped_mm_key(encoder, x_c, w_c, mode);
  auto& cache = grouped_mm_cache();
  auto it = cache.find(cache_key);
  if (it == cache.end()) {
    auto graph = grouped_mm_graph(
        handle, x_c, w_c, token_offsets, lhs_indices, top_k, out);
    it = cache.emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {X, gpu_ptr<void>(x_c)},
      {W, gpu_ptr<void>(w_c)},
      {TOKEN_OFFSETS, gpu_ptr<void>(token_offsets)},
      {O, gpu_ptr<void>(out)}};
  if (lhs_indices.has_value()) {
    variant_pack[TOKEN_INDEX] = gpu_ptr<void>(*lhs_indices);
  }
  CHECK_CUDNN_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

} // namespace

} // namespace mlx::core
