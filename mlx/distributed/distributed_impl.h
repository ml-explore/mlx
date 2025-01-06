// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

/**
 * Abstract base class of a distributed group implementation.
 */
class GroupImpl {
 public:
  virtual int rank() = 0;
  virtual int size() = 0;
  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) = 0;

  virtual void all_sum(const array& input, array& output) = 0;
  virtual void all_gather(const array& input, array& output) = 0;
  virtual void send(const array& input, int dst) = 0;
  virtual void recv(array& out, int src) = 0;
};

/* Return the communication stream. */
Stream communication_stream();

/* Perform an all reduce sum operation */
void all_sum(Group group, const array& input, array& output);

/* Perform an all gather operation */
void all_gather(Group group, const array& input, array& output);

/** Send an array to the dst rank */
void send(Group group, const array& input, int dst);

/** Recv an array from the src rank */
void recv(Group group, array& out, int src);

} // namespace mlx::core::distributed::detail
