// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

struct RMSNormCacheKey {
  int device_id;
  cudnnDataType_t cudnn_dtype;
  std::array<int, MAX_NDIM> x_shape;
  std::array<int64_t, MAX_NDIM> x_strides;
  std::array<int, MAX_NDIM> scale_shape;
  std::array<int64_t, MAX_NDIM> scale_strides;
  uint8_t x_alignment;
  uint8_t scale_alignment;
};

auto& rms_norm_cache() {
  static LRUBytesKeyCache<RMSNormCacheKey, cudnn_frontend::ExecutionPlan> cache(
      /* capacity */ 32);
  return cache;
}

auto build_norm_op_graph(
    cudnnHandle_t handle,
    cudnnBackendDescriptorType_t backend_type,
    cudnnBackendNormMode_t norm_mode,
    cudnnBackendNormFwdPhase_t norm_phase,
    const array& x,
    const array& scale,
    array& y) {
  auto op = cudnn_frontend::OperationBuilder(backend_type)
                .setNormalizationMode(norm_mode)
                .setNormFwdPhase(norm_phase)
                .setxDesc(build_cudnn_tensor_4d_nchw('x', x))
                .setScale(build_cudnn_tensor_4d_nchw('s', scale))
                .setyDesc(build_cudnn_tensor_4d_nchw('y', y))
                .setEpsilonTensor(build_cudnn_scalar_4d('e', float32))
                .build();

  std::array<cudnn_frontend::Operation const*, 1> ops = {&op};
  return cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(ops.size(), ops.data())
      .build();
}

} // namespace

namespace fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNorm::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  const array& x = inputs[0];
  array scale = inputs[1];

  // cuDNN does not accept scalar as scale.
  if (scale.ndim() == 0) {
    array scale_copy({1, 1, 1, x.shape(-1)}, scale.dtype(), nullptr, {});
    fill_gpu(scale, scale_copy, s);
    encoder.add_temporary(scale_copy);
    scale = scale_copy;
  }

  // TODO: Handle donations.
  assert(outputs.size() == 1);
  array& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(x);
  encoder.set_input_array(scale);
  encoder.set_output_array(out);

  // Search cache.
  RMSNormCacheKey cache_key{
      encoder.device().cuda_device(),
      dtype_to_cudnn_type(out.dtype()),
      vector_key(x.shape()),
      vector_key(x.strides()),
      vector_key(scale.shape()),
      vector_key(scale.strides()),
      get_alignment(x),
      get_alignment(scale)};
  if (auto it = rms_norm_cache().find(cache_key);
      it != rms_norm_cache().end()) {
    auto& plan = it->second;
    if (!encode_cudnn_plan(
            encoder, plan, {'x', 's', 'y', 'e'}, x, scale, out, eps_)) {
      throw std::runtime_error("[norm] Cached plan failed to execute.");
    }
    return;
  }

  // Try to build op graph.
  auto handle = encoder.device().cudnn_handle();
  auto backend_type = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
  auto norm_mode = CUDNN_RMS_NORM;
  auto norm_phase = CUDNN_NORM_FWD_INFERENCE;
  auto op_graph = build_norm_op_graph(
      handle, backend_type, norm_mode, norm_phase, x, scale, out);

  // Find a plan for the graph and execute it.
  auto plan = find_cudnn_plan_from_op_graph(
      handle, backend_type, out.dtype(), op_graph);
  if (!plan) {
    throw std::runtime_error("[norm] Unable to find an execution plan.");
  }
  if (!encode_cudnn_plan(
          encoder, *plan, {'x', 's', 'y', 'e'}, x, scale, out, eps_)) {
    throw std::runtime_error("[conv] Failed to run execution plan.");
  }
  rms_norm_cache().emplace(cache_key, std::move(*plan));
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("RMSNormVJP::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  throw std::runtime_error("NYI");
}

} // namespace fast

} // namespace mlx::core
