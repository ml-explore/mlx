// Copyright © 2024 Apple Inc.

#include "mlx/dist/ops.h"

namespace mlx::core::dist {

array all_reduce_sum(const array& x, std::shared_ptr<Group> group) {
  if (group == nullptr) {
    group = dist::init();
  }

  return array(
      x.shape(),
      x.dtype(),
      std::make_shared<AllReduce>(group, AllReduce::Sum),
      {x});
}

} // namespace mlx::core::dist
