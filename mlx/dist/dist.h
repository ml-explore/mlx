// Copyright © 2024 Apple Inc.

#pragma once

#include <memory>

#include "mlx/array.h"

namespace mlx::core::dist {

/* Check if a communication backend is available */
bool is_available();

/**
 * A dist::Group represents a group of independent mlx processes that can
 * communicate. We must also be able to create sub-groups from a group in order
 * to define more granular communication.
 */
struct Group {
  virtual int rank() = 0;
  virtual int size() = 0;
  virtual std::shared_ptr<Group> split(int n) = 0;
};

/**
 * Initialize the distributed backend and return the group containing all
 * discoverable processes.
 */
std::shared_ptr<Group> init();

/* Perform an all reduce sum operation */
void all_reduce_sum(
    std::shared_ptr<Group> group,
    const array& input,
    array& output);

} // namespace mlx::core::dist
