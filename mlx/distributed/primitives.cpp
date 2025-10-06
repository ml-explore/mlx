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
    case Max:
      return {{all_max(inputs[0], group(), stream())}, axes};
    case Min:
      return {{all_min(inputs[0], group(), stream())}, axes};
    default:

      throw std::runtime_error(
          "Only all reduce sum, max and min are supported for now");
  }
}

std::vector<array> AllReduce::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  switch (reduce_type_) {
    case Sum:
      return {all_sum(tangents[0], group(), stream())};
    case Max:
      return {all_max(tangents[0], group(), stream())};
    case Min:
      return {all_min(tangents[0], group(), stream())};
    default:
      throw std::runtime_error(
          "Only all reduce sum, max and min are supported for now");
  }
}

std::vector<array> AllReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
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
    const std::vector<int>&) {
  return {all_gather(tangents[0], group(), stream())};
}

std::vector<array> AllGather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  auto g = group();
  auto ndim = primals[0].ndim();
  Shape starts(primals[0].ndim(), 0);
  auto stops = primals[0].shape();
  if (ndim == 0) {
    starts.push_back(0);
    stops.push_back(1);
  }
  starts[0] = g.rank() * stops[0];
  stops[0] += starts[0];
  auto out = slice(cotangents[0], starts, stops);
  if (ndim == 0) {
    out = squeeze(out, 0);
  }
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Send::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{send(inputs[0], dst_, group(), stream())}, axes};
}

} // namespace mlx::core::distributed
