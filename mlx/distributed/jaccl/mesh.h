// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/jaccl/utils.h"

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

namespace mlx::core::distributed::jaccl {

/**
 * The JACCL communication group for a fully connected mesh. We expect one
 * connection per peer and it should be the lowest latency communication group
 * for small to medium size messages.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class MeshGroup : public GroupImpl {
 public:
  MeshGroup(
      int rank,
      const std::vector<std::string>& device_names,
      const char* coordinator_addr);

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input, array& output, Stream stream) override;
  void all_max(const array& input, array& output, Stream stream) override;
  void all_min(const array& input, array& output, Stream stream) override;
  void all_gather(const array& input, array& output, Stream stream) override;
  void all_to_all(const array& input, array& output, Stream stream) override;
  void send(const array& input, int dst, Stream stream) override;
  void recv(array& out, int src, Stream stream) override;

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] sum_scatter not supported.");
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const array& input,
      array& output,
      Stream stream,
      ReduceOp reduce_op);

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state and all buffers
   * should have been allocated.
   */
  void initialize();

  /**
   * Allocate all the buffers that we will use in the communication group.
   */
  void allocate_buffers();

  void send_to(int sz, int rank, int buff) {
    connections_[rank].post_send(
        send_buffer(sz, buff), SEND_WR << 16 | buff << 8 | rank);
  }

  void recv_from(int sz, int rank, int buff) {
    connections_[rank].post_recv(
        recv_buffer(sz, buff, rank), RECV_WR << 16 | buff << 8 | rank);
  }

  SharedBuffer& send_buffer(int sz, int buff) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank_];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int rank) {
    return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank];
  }

  void post_send_all(int sz, int buff) {
    auto& b = send_buffer(sz, buff);
    int wr_id = SEND_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_send(b, wr_id | i);
    }
  }

  void post_recv_all(int sz, int buff) {
    int b = sz * NUM_BUFFERS * size_ + buff * size_;
    int wr_id = RECV_WR << 16 | buff << 8;
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      connections_[i].post_recv(buffers_[b + i], wr_id | i);
    }
  }

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
};

} // namespace mlx::core::distributed::jaccl
