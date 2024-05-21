// Copyright © 2024 Apple Inc.

#pragma once

#include <memory>

#include "mlx/array.h"

namespace mlx::core::distributed {

namespace detail {

/* Return the communication stream. */
Stream communication_stream();

} // namespace detail

/* Check if a communication backend is available */
bool is_available();

/**
 * A distributed::Group represents a group of independent mlx processes that
 * can communicate. We must also be able to create sub-groups from a group in
 * order to define more granular communication.
 */
struct Group {
  virtual int rank() = 0;
  virtual int size() = 0;
  /**
   * Split the group according to the provided color. Namely processes that use
   * the same color will go to the same group.
   *
   * The key defines the rank of the processes in the new group. The smaller
   * the key the smaller the rank. If the provided key is negative, then the
   * rank in the current group is used.
   */
  virtual std::shared_ptr<Group> split(int color, int key = -1) = 0;
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

/* Perform an all reduce sum operation */
void all_gather(
    std::shared_ptr<Group> group,
    const array& input,
    array& output);

} // namespace mlx::core::distributed
