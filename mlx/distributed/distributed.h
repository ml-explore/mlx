// Copyright © 2024 Apple Inc.

#pragma once

#include <memory>

#include "mlx/array.h"

namespace mlx::core::distributed {

/* Check if a communication backend is available */
bool is_available();

/**
 * A distributed::Group represents a group of independent mlx processes that
 * can communicate. We must also be able to create sub-groups from a group in
 * order to define more granular communication.
 */
struct Group {
  Group(std::shared_ptr<void> group) : group_(group) {}

  int rank();
  int size();

  /**
   * Split the group according to the provided color. Namely processes that use
   * the same color will go to the same group.
   *
   * The key defines the rank of the processes in the new group. The smaller
   * the key the smaller the rank. If the provided key is negative, then the
   * rank in the current group is used.
   */
  Group split(int color, int key = -1);

  void barrier();

  const std::shared_ptr<void>& raw_group() {
    return group_;
  }

 private:
  std::shared_ptr<void> group_{nullptr};
};

/**
 * Initialize the distributed backend and return the group containing all
 * discoverable processes.
 *
 * If strict is true then throw an error if we couldn't initialize the
 * distributed subsystem. Otherwise simply return a singleton group which will
 * render communication operations as no-op.
 */
Group init(bool strict = false);

} // namespace mlx::core::distributed
