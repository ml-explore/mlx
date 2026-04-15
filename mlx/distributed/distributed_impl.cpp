// Copyright © 2024 Apple Inc.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mlx/distributed/distributed_impl.h"

namespace mlx::core::distributed::detail {

int GroupImpl::blocking_exchange_v(
    const array& send_rows_buf,
    int send_rows,
    array& recv_rows_buf,
    int recv_cap_rows,
    int row_stride_bytes,
    int peer,
    ExchangeTag count_tag,
    ExchangeTag payload_tag,
    array& count_send,
    array& count_recv) {
  if (send_rows < 0 || recv_cap_rows < 0 || row_stride_bytes <= 0) {
    throw std::invalid_argument(
        "[blocking_exchange_v] invalid args: send_rows=" +
        std::to_string(send_rows) +
        " recv_cap_rows=" + std::to_string(recv_cap_rows) +
        " row_stride_bytes=" + std::to_string(row_stride_bytes));
  }

  // Overflow check: send_rows * row_stride_bytes
  if (send_rows > 0 &&
      static_cast<size_t>(row_stride_bytes) >
          SIZE_MAX / static_cast<size_t>(send_rows)) {
    throw std::overflow_error(
        "[blocking_exchange_v] send_rows * row_stride_bytes overflow");
  }
  size_t send_payload_bytes = static_cast<size_t>(send_rows) * row_stride_bytes;
  if (send_payload_bytes > send_rows_buf.nbytes()) {
    throw std::out_of_range(
        "[blocking_exchange_v] send payload exceeds send buffer");
  }

  // Phase 1: Exchange counts
  count_send.data<int32_t>()[0] = static_cast<int32_t>(send_rows);
  count_recv.data<int32_t>()[0] = 0;

  blocking_sendrecv(
      count_send,
      sizeof(int32_t),
      count_recv,
      sizeof(int32_t),
      peer,
      count_tag);

  int peer_count = count_recv.data<int32_t>()[0];

  if (peer_count < 0) {
    throw std::runtime_error(
        "[blocking_exchange_v] negative peer_count=" +
        std::to_string(peer_count));
  }
  if (peer_count > recv_cap_rows) {
    throw std::out_of_range(
        "[blocking_exchange_v] peer_count=" + std::to_string(peer_count) +
        " exceeds recv_cap_rows=" + std::to_string(recv_cap_rows));
  }

  // Overflow check: peer_count * row_stride_bytes
  if (peer_count > 0 &&
      static_cast<size_t>(row_stride_bytes) >
          SIZE_MAX / static_cast<size_t>(peer_count)) {
    throw std::overflow_error(
        "[blocking_exchange_v] peer_count * row_stride_bytes overflow");
  }
  size_t recv_payload_bytes =
      static_cast<size_t>(peer_count) * row_stride_bytes;
  if (recv_payload_bytes > recv_rows_buf.nbytes()) {
    throw std::out_of_range(
        "[blocking_exchange_v] peer payload exceeds recv buffer");
  }

  // Early return if no payload
  if (send_rows == 0 && peer_count == 0) {
    return 0;
  }

  // Phase 2: Exchange payload
  blocking_sendrecv(
      send_rows_buf,
      send_payload_bytes,
      recv_rows_buf,
      recv_payload_bytes,
      peer,
      payload_tag);

  return peer_count;
}

} // namespace mlx::core::distributed::detail
