// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

void AllReduce::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllReduce::eval_gpu] has no GPU implementation.");
}

void AllGather::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[AllGather::eval_gpu] has no GPU implementation.");
}

void Send::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Send::eval_gpu] has no GPU implementation.");
}

void Recv::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("[Recv::eval_gpu] has no GPU implementation.");
}

void ReduceScatter::eval_gpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error(
      "[ReduceScatter::eval_gpu] has no GPU implementation.");
}

} // namespace mlx::core::distributed
