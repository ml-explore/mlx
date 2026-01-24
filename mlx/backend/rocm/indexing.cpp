// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <fmt/format.h>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace {

constexpr const char* g_scatter_ops[] = {"Max", "Min", "Sum", "Prod", "Assign"};

} // namespace

// Note: Gather, Scatter, GatherAxis, ScatterAxis implementations require
// JIT compilation support. For now, we provide stub implementations that
// throw errors, similar to how CUDA handles unsupported operations.

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("Gather::eval_gpu not yet implemented for ROCm.");
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("Scatter::eval_gpu not yet implemented for ROCm.");
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("GatherAxis::eval_gpu not yet implemented for ROCm.");
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("ScatterAxis::eval_gpu not yet implemented for ROCm.");
}

} // namespace mlx::core
