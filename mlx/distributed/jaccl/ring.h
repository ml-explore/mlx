// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/jaccl/utils.h"

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

constexpr int MAX_CONNS = 4;

namespace mlx::core::distributed::jaccl {

/**
 * The JACCL communication group for a ring where each node is connected to its
 * two neighboring nodes. It should be the highest bandwidth communication
 * group for large messages when many connections per peer are used.
 *
 * Like all JACCL groups it uses a side channel to exchange the necessary
 * information and then configure the connections to be ready for RDMA
 * operations.
 */
class RingGroup : public GroupImpl {
 public:
  RingGroup(
      int rank,
      int size,
      const std::vector<std::string>& left_devices,
      const std::vector<std::string>& right_devices,
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
  void send(const array& input, int dst, Stream stream) override;
  void recv(array& out, int src, Stream stream) override;

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] sum_scatter not supported.");
  }

  void all_to_all(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] all_to_all not supported.");
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

  template <int MAX_DIR, typename T, typename ReduceOp>
  void all_reduce_impl(
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      int n_wires,
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

  void send_to(int sz, int buff, int left_right, int wire) {
    if (left_right) {
      left_[wire].post_send(
          send_buffer_left(sz, buff, wire),
          SEND_WR << 16 | buff << 8 | (MAX_CONNS + wire));
    } else {
      right_[wire].post_send(
          send_buffer_right(sz, buff, wire), SEND_WR << 16 | buff << 8 | wire);
    }
  }

  void recv_from(int sz, int buff, int left_right, int wire) {
    if (left_right) {
      right_[wire].post_recv(
          recv_buffer_right(sz, buff, wire),
          RECV_WR << 16 | buff << 8 | (MAX_CONNS + wire));
    } else {
      left_[wire].post_recv(
          recv_buffer_left(sz, buff, wire), RECV_WR << 16 | buff << 8 | wire);
    }
  }

  SharedBuffer& send_buffer_right(int sz, int buff, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 + wire];
  }

  SharedBuffer& send_buffer_left(int sz, int buff, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 + MAX_CONNS +
         wire];
  }

  SharedBuffer& send_buffer(int sz, int buff, int left_right, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 +
         left_right * MAX_CONNS + wire];
  }

  SharedBuffer& recv_buffer_left(int sz, int buff, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 + wire];
  }

  SharedBuffer& recv_buffer_right(int sz, int buff, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 + MAX_CONNS +
         wire];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int left_right, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * MAX_CONNS * 2 + buff * MAX_CONNS * 2 +
         left_right * MAX_CONNS + wire];
  }

  template <int MAX_DIR>
  void post_recv_all(int sz, int buff, int n_wires) {
    for (int lr = 0; lr < MAX_DIR; lr++) {
      for (int lw = 0; lw < n_wires; lw++) {
        recv_from(sz, buff, lr, lw);
      }
    }
  }

  void post_recv_all(int sz, int buff) {
    post_recv_all<2>(sz, buff, left_.size());
  }

  template <int MAX_DIR>
  void post_send_all(int sz, int buff, int n_wires) {
    for (int lr = 0; lr < MAX_DIR; lr++) {
      for (int lw = 0; lw < n_wires; lw++) {
        send_to(sz, buff, lr, lw);
      }
    }
  }

  void post_send_all(int sz, int buff) {
    post_send_all<2>(sz, buff, left_.size());
  }

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::vector<Connection> left_;
  std::vector<Connection> right_;
  std::vector<SharedBuffer> send_buffers_;
  std::vector<SharedBuffer> recv_buffers_;
};

} // namespace mlx::core::distributed::jaccl
