// Copyright © 2024 Apple Inc.

#include "mlx/distributed/ops.h"

namespace mlx::core::distributed {

array all_reduce_sum(const array& x, std::shared_ptr<Group> group) {
  if (group == nullptr) {
    group = distributed::init();
  }

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(group, AllReduce::Sum),
      {x});
}

} // namespace mlx::core::distributed
