// Copyright © 2024 Apple Inc.

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>

#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

namespace {

Group to_group(std::optional<Group> group) {
  if (group.has_value()) {
    return group.value();
  } else {
    return distributed::init();
  }
}

// Auto mode: resolve to CPU (Phase 5 policy removed for PR1 simplification)
MoeBackend resolve_auto_backend(
    int /* N */,
    int /* top_k */,
    int /* D */,
    int /* elem_size */) {
  return MoeBackend::Cpu;
}

MoeBackend resolve_backend_str(const std::string& backend) {
  if (backend == "auto")
    return MoeBackend::Auto;
  if (backend == "cpu")
    return MoeBackend::Cpu;
  if (backend == "metal")
    return MoeBackend::Metal;
  throw std::invalid_argument(
      "[moe] invalid backend '" + backend + "', expected auto/cpu/metal");
}

// GPU stream selection with graceful fallback to CPU
Stream resolve_moe_stream(MoeBackend& backend, StreamOrDevice s) {
  if (backend == MoeBackend::Metal) {
    try {
      return to_stream(s, Device::gpu);
    } catch (...) {
      static std::once_flag warn_once;
      std::call_once(warn_once, []() {
        std::cerr << "[MoE EP] GPU stream unavailable. Falling back to CPU.\n";
      });
      backend = MoeBackend::Cpu;
    }
  }
  return to_stream(s, Device::cpu);
}

} // namespace

array all_sum(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Sum),
      {x});
}

array all_max(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Max),
      {x});
}

array all_min(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(stream, group, AllReduce::Min),
      {x});
}

array all_gather(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }
  auto stream = detail::communication_stream(group, s);

  auto result_shape = x.shape();
  if (result_shape.size() == 0) {
    result_shape.push_back(group.size());
  } else {
    result_shape[0] *= group.size();
  }
  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<AllGather>(stream, group),
      {x});
}

array send(
    const array& x,
    int dst,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot send to a singleton group");
  }
  auto stream = detail::communication_stream(group, s);

  if (dst < 0 || dst >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid destination=" << dst << " for a group of size "
        << group.size();
    throw std::invalid_argument(msg.str());
  }

  return array(
      x.shape(), x.dtype(), std::make_shared<Send>(stream, group, dst), {x});
}

array recv(
    Shape shape,
    Dtype dtype,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot recv from a singleton group");
  }
  auto stream = detail::communication_stream(group, s);

  if (src < 0 || src >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid source=" << src << " for a group of size " << group.size();
    throw std::invalid_argument(msg.str());
  }

  return array(
      std::move(shape),
      std::move(dtype),
      std::make_shared<Recv>(stream, group, src),
      std::vector<array>{});
}

array recv_like(
    const array& x,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return recv(x.shape(), x.dtype(), src, group_, s);
}

array sum_scatter(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  if (group.size() == 1) {
    return x;
  }
  if (x.shape()[0] % group.size() != 0) {
    std::ostringstream msg;
    msg << "[sum_scatter] Invalid shape=" << x.shape()
        << " for a group of size " << group.size()
        << ". The first dimension (axis 0) must be divisible by the group size.";
    throw std::invalid_argument(msg.str());
  }

  auto result_shape = x.shape();
  result_shape[0] /= group.size();
  auto stream = detail::communication_stream(group, s);

  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<ReduceScatter>(stream, group, ReduceScatter::Sum),
      {x});
}

array all_to_all(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  if (group.size() == 1) {
    return x;
  }
  if (x.ndim() < 1) {
    throw std::invalid_argument("[all_to_all] Input must be at least 1-D.");
  }
  if (x.shape(0) % group.size() != 0) {
    std::ostringstream msg;
    msg << "[all_to_all] Invalid shape=" << x.shape() << " for a group of size "
        << group.size()
        << ". The first dimension (axis 0) must be divisible by the group size.";
    throw std::invalid_argument(msg.str());
  }
  auto stream = detail::communication_stream(group, s);
  return array(
      x.shape(), x.dtype(), std::make_shared<AllToAll>(stream, group), {x});
}

std::pair<array, array> moe_dispatch_exchange(
    const array& tokens,
    const array& expert_indices,
    int num_experts,
    int capacity,
    std::optional<Group> group_,
    bool deterministic,
    const std::string& backend,
    StreamOrDevice s) {
  auto group = to_group(group_);

  // Validate inputs
  if (tokens.ndim() != 2) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] tokens must be 2-D [N, D].");
  }
  if (expert_indices.ndim() != 2) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] expert_indices must be 2-D [N, top_k].");
  }
  if (tokens.shape(0) != expert_indices.shape(0)) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] tokens and expert_indices must have same N.");
  }
  if (expert_indices.dtype() != int32) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] expert_indices must have dtype int32.");
  }
  if (num_experts % group.size() != 0) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] num_experts must be divisible by group size.");
  }
  if (capacity <= 0) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] capacity must be positive.");
  }

  int world_size = group.size();
  int experts_per_device = num_experts / world_size;

  if (experts_per_device > 65535 || capacity > 65535) {
    throw std::invalid_argument(
        "[moe_dispatch_exchange] meta32 overflow: "
        "experts_per_device=" +
        std::to_string(experts_per_device) + " capacity=" +
        std::to_string(capacity) + " — both must be <= 65535 for v3 protocol");
  }

  int cap_total = world_size * capacity;
  int D = tokens.shape(1);
  int N = tokens.shape(0);
  int top_k = expert_indices.shape(1);

  // Output shapes:
  // dispatched: [experts_per_device, cap_total, D]
  // route_indices: [N, top_k] int32
  auto dispatched_shape = Shape{experts_per_device, cap_total, D};
  auto route_indices_shape = Shape{N, top_k};

  auto moe_backend = resolve_backend_str(backend);

  // Resolve Auto → CPU
  if (moe_backend == MoeBackend::Auto) {
    moe_backend = MoeBackend::Cpu;
  }

  // ws > 2: Metal not yet optimized, fall back to CPU
  if (moe_backend == MoeBackend::Metal && world_size > 2) {
    static std::once_flag warned;
    std::call_once(warned, []() {
      std::cerr
          << "[MoE EP] Metal backend not yet optimized for world_size > 2, "
          << "falling back to CPU path." << std::endl;
    });
    moe_backend = MoeBackend::Cpu;
  }

  auto stream = resolve_moe_stream(moe_backend, s);

  auto outputs = array::make_arrays(
      {std::move(dispatched_shape), std::move(route_indices_shape)},
      {tokens.dtype(), int32},
      std::make_shared<MoeDispatchExchange>(
          stream, group, num_experts, capacity, deterministic, moe_backend),
      {tokens, expert_indices});

  return {outputs[0], outputs[1]};
}

array moe_combine_exchange(
    const array& expert_outputs,
    const array& route_indices,
    const array& weights,
    const array& original_tokens,
    int num_experts,
    int capacity,
    std::optional<Group> group_,
    bool deterministic,
    const std::string& backend,
    StreamOrDevice s) {
  auto group = to_group(group_);

  if (expert_outputs.ndim() != 3) {
    throw std::invalid_argument(
        "[moe_combine_exchange] expert_outputs must be 3-D [E_local, cap_total, D].");
  }
  if (route_indices.ndim() != 2) {
    throw std::invalid_argument(
        "[moe_combine_exchange] route_indices must be 2-D [N, top_k].");
  }
  if (weights.ndim() != 2) {
    throw std::invalid_argument(
        "[moe_combine_exchange] weights must be 2-D [N, top_k].");
  }
  if (original_tokens.ndim() != 2) {
    throw std::invalid_argument(
        "[moe_combine_exchange] original_tokens must be 2-D [N, D].");
  }
  if (route_indices.dtype() != int32) {
    throw std::invalid_argument(
        "[moe_combine_exchange] route_indices must have dtype int32.");
  }
  if (weights.dtype() != float32) {
    throw std::invalid_argument(
        "[moe_combine_exchange] weights must have dtype float32.");
  }

  // Shape compatibility checks
  if (route_indices.shape(0) != weights.shape(0) ||
      route_indices.shape(0) != original_tokens.shape(0)) {
    std::ostringstream msg;
    msg << "[moe_combine_exchange] N dimension mismatch: "
        << "route_indices.shape(0)=" << route_indices.shape(0)
        << " weights.shape(0)=" << weights.shape(0)
        << " original_tokens.shape(0)=" << original_tokens.shape(0);
    throw std::invalid_argument(msg.str());
  }
  if (route_indices.shape(1) != weights.shape(1)) {
    std::ostringstream msg;
    msg << "[moe_combine_exchange] top_k dimension mismatch: "
        << "route_indices.shape(1)=" << route_indices.shape(1)
        << " weights.shape(1)=" << weights.shape(1);
    throw std::invalid_argument(msg.str());
  }
  if (original_tokens.shape(1) != expert_outputs.shape(2)) {
    std::ostringstream msg;
    msg << "[moe_combine_exchange] hidden dim D mismatch: "
        << "original_tokens.shape(1)=" << original_tokens.shape(1)
        << " expert_outputs.shape(2)=" << expert_outputs.shape(2);
    throw std::invalid_argument(msg.str());
  }
  if (original_tokens.dtype() != expert_outputs.dtype()) {
    std::ostringstream msg;
    msg << "[moe_combine_exchange] dtype mismatch: "
        << "original_tokens.dtype=" << original_tokens.dtype()
        << " expert_outputs.dtype=" << expert_outputs.dtype();
    throw std::invalid_argument(msg.str());
  }

  int world_size = group.size();
  if (expert_outputs.shape(1) != world_size * capacity) {
    std::ostringstream msg;
    msg << "[moe_combine_exchange] expert_outputs.shape(1)="
        << expert_outputs.shape(1)
        << " must equal world_size * capacity = " << world_size << " * "
        << capacity << " = " << (world_size * capacity) << ".";
    throw std::invalid_argument(msg.str());
  }
  if (capacity <= 0) {
    throw std::invalid_argument(
        "[moe_combine_exchange] capacity must be positive.");
  }

  int N = original_tokens.shape(0);
  int D = original_tokens.shape(1);
  auto combined_shape = Shape{N, D};

  auto moe_backend = resolve_backend_str(backend);

  // Resolve Auto → CPU
  if (moe_backend == MoeBackend::Auto) {
    moe_backend = MoeBackend::Cpu;
  }

  // ws > 2: Metal not yet optimized, fall back to CPU
  if (moe_backend == MoeBackend::Metal && world_size > 2) {
    static std::once_flag warned;
    std::call_once(warned, []() {
      std::cerr
          << "[MoE EP] Metal backend not yet optimized for world_size > 2, "
          << "falling back to CPU path." << std::endl;
    });
    moe_backend = MoeBackend::Cpu;
  }

  auto stream = resolve_moe_stream(moe_backend, s);

  return array(
      std::move(combined_shape),
      expert_outputs.dtype(),
      std::make_shared<MoeCombineExchange>(
          stream, group, num_experts, capacity, deterministic, moe_backend),
      {expert_outputs, route_indices, weights, original_tokens});
}

} // namespace mlx::core::distributed
