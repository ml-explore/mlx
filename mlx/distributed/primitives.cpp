// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"
#include "mlx/ops.h"

namespace mlx::core::distributed {

std::pair<std::vector<array>, std::vector<int>> AllReduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  switch (reduce_type_) {
    case Sum:
      return {{all_sum(inputs[0], group(), stream())}, axes};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::vector<array> AllReduce::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  switch (reduce_type_) {
    case Sum:
      return {all_sum(tangents[0], group(), stream())};
    default:
      throw std::runtime_error("Only all reduce sum is supported for now");
  }
}

std::vector<array> AllReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return cotangents;
}

std::pair<std::vector<array>, std::vector<int>> AllGather::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{all_gather(inputs[0], group(), stream())}, axes};
}

std::vector<array> AllGather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {all_gather(tangents[0], group(), stream())};
}

std::vector<array> AllGather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto g = group();
  Shape starts(primals[0].ndim(), 0);
  auto stops = primals[0].shape();
  starts[0] = g.rank() * stops[0];
  stops[0] += starts[0];
  return {slice(cotangents[0], starts, stops)};
}

std::pair<std::vector<array>, std::vector<int>> Send::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{send(inputs[0], dst_, group(), stream())}, axes};
}

} // namespace mlx::core::distributed
