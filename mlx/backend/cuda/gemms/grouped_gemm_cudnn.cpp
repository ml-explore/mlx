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

#if CUDNN_VERSION >= 91800

namespace {

constexpr int GMM_NDIM = 3;

struct GatherMMCacheKey {
  int device_id;
  fe::DataType_t cudnn_dtype;
  int mode; // NONE / GATHER / SCATTER
  std::array<int, GMM_NDIM> x_shape;
  std::array<int64_t, GMM_NDIM> x_strides;
  std::array<int, GMM_NDIM> w_shape;
  std::array<int64_t, GMM_NDIM> w_strides;
  std::array<int, GMM_NDIM> out_shape;
};

inline BytesKey<GatherMMCacheKey> build_grouped_mm_key(
    cu::CommandEncoder& encoder,
    const array& x,
    const array& w,
    const array& out) {
  BytesKey<GatherMMCacheKey> key;
  key.pod.device_id = encoder.device().cuda_device();
  key.pod.cudnn_dtype = dtype_to_cudnn_type(x.dtype());
  key.pod.mode = 0; // NONE
  key.pod.x_shape = vector_key<GMM_NDIM>(x.shape());
  key.pod.x_strides = vector_key<GMM_NDIM>(x.strides());
  key.pod.w_shape = vector_key<GMM_NDIM>(w.shape());
  key.pod.w_strides = vector_key<GMM_NDIM>(w.strides());
  key.pod.out_shape = vector_key<GMM_NDIM>(out.shape());
  return key;
}

enum UIDS { X, W, TOKEN_OFFSETS, TOKEN_INDEX, O };

// cudnn expects specific shape and strides for grouped matmul:
// [1, T, H]
void set_moe_layout(
    std::shared_ptr<fe::graph::Tensor_attributes>& t,
    const array& x) {
  int64_t L = x.shape(0);
  int64_t D = x.shape(-1);
  int64_t sL = x.strides(0);
  int64_t sD = x.strides(-1);
  t->set_dim({1, L, D}).set_stride({L * sL, sL, sD});
}

DnnGraph grouped_mm_graph(
    cudnnHandle_t handle,
    const array& x,
    const array& w,
    const array& token_offsets,
    const array& output) {
  DnnGraph graph(handle, x.dtype());

  auto x_ = graph.tensor("X", X, x);
  set_moe_layout(x_, x);
  auto w_ = graph.tensor("W", W, w);
  auto token_offsets_ =
      graph.tensor("TOKEN_OFFSETS", TOKEN_OFFSETS, token_offsets);

  auto moe_grouped_matmul_attr =
      fe::graph::Moe_grouped_matmul_attributes()
          .set_name("grouped_matmul")
          .set_mode(fe::MoeGroupedMatmulMode_t::NONE);

  std::shared_ptr<fe::graph::Tensor_attributes> token_index = nullptr;
  std::shared_ptr<fe::graph::Tensor_attributes> token_ks = nullptr;

  auto out_ = graph.moe_grouped_matmul(
      x_, w_, token_offsets_, token_index, token_ks, moe_grouped_matmul_attr);
  graph.tensor(out_, O, output);
  set_moe_layout(out_, output);
  out_->set_output(true);

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

} // namespace

void cudnn_grouped_mm(
    const array& x,
    const array& w,
    const array& token_offsets, // precomputed offsets for each expert
    array& out,
    cu::CommandEncoder& encoder) {
  nvtx3::scoped_range r("cudnn_grouped_mm");

  auto handle = get_cudnn_handle(encoder.device());

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(token_offsets);
  encoder.set_output_array(out);

  auto cache_key = build_grouped_mm_key(encoder, x, w, out);
  auto& cache = grouped_mm_cache();
  auto it = cache.find(cache_key);
  if (it == cache.end()) {
    auto graph = grouped_mm_graph(handle, x, w, token_offsets, out);
    it = cache.emplace(cache_key, std::move(graph)).first;
  }
  auto& graph = it->second;

  std::unordered_map<int64_t, void*> variant_pack{
      {X, gpu_ptr<void>(x)},
      {W, gpu_ptr<void>(w)},
      {TOKEN_OFFSETS, gpu_ptr<void>(token_offsets)},
      {O, gpu_ptr<void>(out)}};
  CHECK_CUDNN_ERROR(graph.encode_graph(encoder, std::move(variant_pack)));
}

#endif

} // namespace mlx::core
