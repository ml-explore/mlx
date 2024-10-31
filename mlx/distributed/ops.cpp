// Copyright © 2024 Apple Inc.

#include <sstream>

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

array send(
    const array& x,
    int dst,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot send to a singleton group");
  }

  if (dst < 0 || dst >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid destination=" << dst << " for a group of size "
        << group.size();
    throw std::invalid_argument(msg.str());
  }

  return array(
      {0}, int32, std::make_shared<Send>(to_stream(s), group, dst), {x});
}

array recv(
    std::vector<int> shape,
    Dtype dtype,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    throw std::invalid_argument("Cannot recv from a singleton group");
  }

  if (src < 0 || src >= group.size()) {
    std::ostringstream msg;
    msg << "Invalid source=" << src << " for a group of size " << group.size();
    throw std::invalid_argument(msg.str());
  }

  return array(
      std::move(shape),
      std::move(dtype),
      std::make_shared<Recv>(to_stream(s), group, src),
      std::vector<array>{});
}

array recv_like(
    const array& x,
    int src,
    std::optional<Group> group_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return recv(x.shape(), x.dtype(), src, group_, s);
}

} // namespace mlx::core::distributed
