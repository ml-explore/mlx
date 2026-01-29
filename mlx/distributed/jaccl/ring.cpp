// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/ring.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

namespace mlx::core::distributed::jaccl {

RingGroup::RingGroup(
    int rank,
    int size,
    const std::vector<std::string>& left_devices,
    const std::vector<std::string>& right_devices,
    const char* coordinator_addr)
    : rank_(rank),
      size_(size),
      side_channel_(rank_, size_, coordinator_addr),
      left_(create_connections(left_devices)),
      right_(create_connections(right_devices)) {}

void RingGroup::initialize() {
  // Create the queue pairs
  for (auto& conn : left_) {
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }
  for (auto& conn : right_) {
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  // Allocate the buffers

  // Initialize the conections
  for (auto& conn : left_) {
    conn.queue_pair_init();
  }
  for (auto& conn : right_) {
    conn.queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier so
  // that all peers have initialized their connections before attempting to
  // transition to RTS.
  std::vector<Destination> left_info;
  for (auto& conn : left_) {
    left_info.emplace_back(conn.info());
  }
  std::vector<Destination> right_info;
  for (auto& conn : right_) {
    right_info.emplace_back(conn.info());
  }
  auto all_left_infos = side_channel_.all_gather(left_info);
  auto all_right_infos = side_channel_.all_gather(right_info);

  // Transition queue pairs to RTS
  int left_peer = (rank_ + size_ - 1) % size_;
  for (int i = 0; i < left_.size(); i++) {
    auto peer_info = all_right_infos[left_peer][i];
    left_[i].queue_pair_rtr(peer_info);
    left_[i].queue_pair_rts();
  }
  int right_peer = (rank_ + 1) % size_;
  for (int i = 0; i < right_.size(); i++) {
    auto peer_info = all_left_infos[right_peer][i];
    right_[i].queue_pair_rtr(peer_info);
    right_[i].queue_pair_rts();
  }
}

void RingGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{});
  });
}

void RingGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
  });
}

void RingGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{});
  });
}

void RingGroup::all_gather(const array& input, array& output, Stream stream) {}

void RingGroup::send(const array& input, int dst, Stream stream) {}

void RingGroup::recv(array& out, int src, Stream stream) {}

template <typename T, typename ReduceOp>
void RingGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {}

} // namespace mlx::core::distributed::jaccl
