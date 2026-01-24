// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

/**
 * Abstract base class of a distributed group implementation.
 */
class GroupImpl {
 public:
  virtual ~GroupImpl() {}

  // Choose the stream this communication group can operate on
  virtual Stream communication_stream(StreamOrDevice s = {}) = 0;

  // Group operations
  virtual int rank() = 0;
  virtual int size() = 0;
  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) = 0;

  // Actual communication operations
  virtual void all_sum(const array& input, array& output, Stream stream) = 0;
  virtual void all_gather(const array& input, array& output, Stream stream) = 0;
  virtual void send(const array& input, int dst, Stream stream) = 0;
  virtual void recv(array& out, int src, Stream stream) = 0;
  virtual void all_max(const array& input, array& output, Stream stream) = 0;
  virtual void all_min(const array& input, array& output, Stream stream) = 0;
};

/* Define the MLX stream that the communication should happen in. */
Stream communication_stream(Group group, StreamOrDevice s = {});

/* Perform an all reduce sum operation */
void all_sum(Group group, const array& input, array& output, Stream stream);

/* Perform an all gather operation */
void all_gather(Group group, const array& input, array& output, Stream stream);

/** Send an array to the dst rank */
void send(Group group, const array& input, int dst, Stream stream);

/** Recv an array from the src rank */
void recv(Group group, array& out, int src, Stream stream);

/** Max reduction */
void all_max(Group group, const array& input, array& output, Stream stream);

/** Min reduction */
void all_min(Group group, const array& input, array& output, Stream stream);

} // namespace mlx::core::distributed::detail
