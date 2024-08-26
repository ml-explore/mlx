// Copyright © 2024 Apple Inc.

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

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(to_stream(s), group, AllReduce::Sum),
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

  auto result_shape = x.shape();
  if (result_shape.size() == 0) {
    result_shape.push_back(group.size());
  } else {
    result_shape[0] *= group.size();
  }
  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<AllGather>(to_stream(s), group),
      {x});
}

} // namespace mlx::core::distributed
