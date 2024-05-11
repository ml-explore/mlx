// Copyright © 2024 Apple Inc.

#include "mlx/distributed/ops.h"

namespace mlx::core::distributed {

array all_reduce_sum(const array& x, std::shared_ptr<Group> group) {
  if (group == nullptr) {
    group = distributed::init();
  }

  if (group->size() == 1) {
    return x;
  }

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(group, AllReduce::Sum),
      {x});
}

array all_gather(const array& x, std::shared_ptr<Group> group) {
  if (group == nullptr) {
    group = distributed::init();
  }

  if (group->size() == 1) {
    return x;
  }

  auto result_shape = x.shape();
  if (result_shape.size() == 0) {
    result_shape.push_back(group->size());
  } else {
    result_shape[0] *= group->size();
  }
  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<AllGather>(group),
      {x});
}

} // namespace mlx::core::distributed
