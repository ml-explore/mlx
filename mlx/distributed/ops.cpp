// Copyright © 2024 Apple Inc.

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

array all_reduce(
    const array& x,
    const std::string& op /* = "sum" */,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  
  if (group.size() == 1) {
    return x;
  }
  
  // Dispatch to appropriate all_reduce operation based on op
  if (op == "sum" || op.empty()) {
    return all_sum(x, group, s);
  } else if (op == "max") {
    return all_max(x, group, s);
  } else if (op == "min") {
    return all_min(x, group, s);
  } else {
    throw std::invalid_argument("Unknown all_reduce operation: " + op);
  }
}

array all_reduce_opt(
    const array& x,
    const std::string& op /* = "sum" */,
    std::optional<Group> group_ /* = std::nullopt */,
    CollectiveAlgorithm algo /* = CollectiveAlgorithm::DEFAULT */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  
  if (group.size() == 1) {
    return x;
  }
  
  auto stream = detail::communication_stream(group, s);
  
  // Algorithm selection based on data size and group characteristics
  if (algo == CollectiveAlgorithm::DEFAULT) {
    // Automatic algorithm selection based on data size
    auto num_elements = x.size();
    if (num_elements < 1024 || group.size() <= 2) {
      algo = CollectiveAlgorithm::LINEAR;
    } else if (group.size() <= 8) {
      algo = CollectiveAlgorithm::RECURSIVE_DOUBLING;
    } else if (num_elements > 1024 * 1024) {
      algo = CollectiveAlgorithm::TREE;
    } else {
      algo = CollectiveAlgorithm::RING;
    }
  }
  
  // Create optimized all-reduce based on algorithm
  // For now, use the existing implementation with hint for future optimization
  if (op == "sum" || op.empty()) {
    return all_sum(x, group, s);
  } else if (op == "max") {
    return all_max(x, group, s);
  } else if (op == "min") {
    return all_min(x, group, s);
  } else if (op == "prod") {
    // TODO: Implement prod all-reduce
    throw std::invalid_argument("Prod operation not yet supported in optimized all_reduce");
  } else {
    throw std::invalid_argument("Unknown all_reduce operation: " + op);
  }
}

array all_gather_opt(
    const array& x,
    std::optional<Group> group_ /* = std::nullopt */,
    CollectiveAlgorithm algo /* = CollectiveAlgorithm::DEFAULT */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  
  if (group.size() == 1) {
    return x;
  }
  
  // For now, use the standard all_gather
  // Algorithm selection can be implemented for better performance
  if (algo == CollectiveAlgorithm::DEFAULT) {
    algo = CollectiveAlgorithm::RING; // Default to ring for gather
  }
  
  return all_gather(x, group, s);
}

array reduce_scatter_opt(
    const array& x,
    const std::string& op /* = "sum" */,
    std::optional<Group> group_ /* = std::nullopt */,
    CollectiveAlgorithm algo /* = CollectiveAlgorithm::DEFAULT */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);
  
  if (group.size() == 1) {
    return x;
  }
  
  // Validate shape
  if (x.shape()[0] % group.size() != 0) {
    std::ostringstream msg;
    msg << "[reduce_scatter_opt] Invalid shape=" << x.shape()
        << " for a group of size " << group.size()
        << ". The first dimension (axis 0) must be divisible by the group size.";
    throw std::invalid_argument(msg.str());
  }
  
  // For now, use the standard sum_scatter
  if (op == "sum" || op.empty()) {
    return sum_scatter(x, group, s);
  } else if (op == "max") {
    // TODO: Implement max reduce-scatter
    throw std::invalid_argument("Max operation not yet supported in optimized reduce_scatter");
  } else if (op == "min") {
    // TODO: Implement min reduce-scatter
    throw std::invalid_argument("Min operation not yet supported in optimized reduce_scatter");
  } else {
    throw std::invalid_argument("Unknown reduce_scatter operation: " + op);
  }
}

array execute_pipeline(
    const std::vector<PipelineStage>& stages,
    const array& input,
    std::optional<Group> group_ /* = std::nullopt */) {
  auto group = to_group(group_);
  
  if (stages.empty()) {
    return input;
  }
  
  auto result = input;
  for (const auto& stage : stages) {
    // In a real implementation, this would:
    // 1. Overlap computation across stages
    // 2. Use communication-computation overlap
    // 3. Handle pipeline bubbles and stalls
    
    // For now, execute each stage sequentially
    result = stage.compute_fn(result);
    
    // If we have a distributed group, sync after each stage
    if (group.size() > 1) {
      // Add a synchronization point
      // In real implementation, this would use events or barriers
    }
  }
  
  return result;
}
} // namespace mlx::core::distributed
