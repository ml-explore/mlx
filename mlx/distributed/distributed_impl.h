// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::detail {

enum class ExchangeTag : uint16_t {
  MoeDispatchCount = 100,
  MoeDispatchPayload = 101,
  MoeCombineReqCount = 110,
  MoeCombineReqPayload = 111,
  MoeCombineResCount = 120,
  MoeCombineResPayload = 121,
};

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
  virtual void
  sum_scatter(const array& input, array& output, Stream stream) = 0;
  virtual void all_to_all(const array& input, array& output, Stream stream) = 0;

  // Blocking (synchronous) communication — runs directly on the calling
  // thread without going through the encoder/stream machinery.
  virtual void blocking_send(const array& input, int dst) {
    throw std::runtime_error(
        "[GroupImpl] blocking_send not supported by this backend");
  }
  virtual void blocking_recv(array& output, int src) {
    throw std::runtime_error(
        "[GroupImpl] blocking_recv not supported by this backend");
  }
  virtual void blocking_all_to_all(const array& input, array& output) {
    throw std::runtime_error(
        "[GroupImpl] blocking_all_to_all not supported by this backend");
  }

  // Tagged bidirectional blocking sendrecv — sends send_nbytes from
  // send_buf and receives recv_nbytes into recv_buf in a single call.
  // Supports asymmetric sizes. If send_nbytes==0 or recv_nbytes==0 the
  // corresponding direction is skipped.
  virtual void blocking_sendrecv(
      const array& send_buf,
      size_t send_nbytes,
      array& recv_buf,
      size_t recv_nbytes,
      int peer,
      ExchangeTag tag) {
    throw std::runtime_error(
        "[GroupImpl] blocking_sendrecv not supported by this backend");
  }

  // Non-virtual concrete helper: variable-length row exchange with a peer.
  // Performs two blocking_sendrecv calls: (1) count exchange, (2) payload.
  // count_send/count_recv must be pre-allocated int32 arrays of size >= 1.
  // Returns: number of rows received from peer (peer_count).
  int blocking_exchange_v(
      const array& send_rows_buf,
      int send_rows,
      array& recv_rows_buf,
      int recv_cap_rows,
      int row_stride_bytes,
      int peer,
      ExchangeTag count_tag,
      ExchangeTag payload_tag,
      array& count_send,
      array& count_recv);
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

/** Reduce scatter with average operation */
void sum_scatter(Group group, const array& input, array& output, Stream stream);

/** All-to-all exchange */
void all_to_all(Group group, const array& input, array& output, Stream stream);

} // namespace mlx::core::distributed::detail
