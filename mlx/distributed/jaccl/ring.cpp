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
      right_(create_connections(right_devices)) {
  if (left_.size() > MAX_CONNS || right_.size() > MAX_CONNS) {
    std::ostringstream msg;
    msg << "[jaccl] Up to " << MAX_CONNS << " per direction supported but "
        << left_.size() << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);
}

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
  allocate_buffers();

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

void RingGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  send_buffers_.clear();
  recv_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < MAX_CONNS * 2; j++) {
        send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  // Register the buffers with the corresponding connections
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < MAX_CONNS * 2; j++) {
        int wire = j % MAX_CONNS;
        int lr = j / MAX_CONNS;
        if (wire >= left_.size()) {
          continue;
        }
        if (lr) {
          send_buffers_[k * NUM_BUFFERS * MAX_CONNS * 2 + i * MAX_CONNS * 2 + j]
              .register_to_protection_domain(left_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * MAX_CONNS * 2 + i * MAX_CONNS * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
        } else {
          send_buffers_[k * NUM_BUFFERS * MAX_CONNS * 2 + i * MAX_CONNS * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * MAX_CONNS * 2 + i * MAX_CONNS * 2 + j]
              .register_to_protection_domain(left_[wire].protection_domain);
        }
      }
    }
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

void RingGroup::all_gather(const array& input, array& output, Stream stream) {
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  size_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MAX_CONNS * 2 * 2;
    int n_wires = left_.size();
    size_t n_bytes_per_wire = (n_bytes + (2 * n_wires) - 1) / (2 * n_wires);
    size_t out_bytes = n_bytes * size_;
    auto [sz, N] = buffer_size_from_message(n_bytes_per_wire);
    int n_steps = (n_bytes_per_wire + N - 1) / N;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    size_t send_offset[2];
    size_t recv_offset[2];
    int64_t limits[2];
    int send_count[2 * MAX_CONNS] = {0};
    int recv_count[2 * MAX_CONNS] = {0};
    send_offset[0] = send_offset[1] = rank_ * n_bytes;
    recv_offset[0] = ((rank_ + size_ - 1) % size_) * n_bytes;
    recv_offset[1] = ((rank_ + 1) % size_) * n_bytes;
    limits[0] = n_wires * n_bytes_per_wire;
    limits[1] = n_bytes;

    // Possible perf improvement by not syncing at every step but running ahead
    // as needed.
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        post_recv_all(sz, buff);
        for (int lr = 0; lr < 2; lr++) {
          for (int lw = 0; lw < n_wires; lw++) {
            int offset = lw * N +
                send_count[lr * MAX_CONNS + lw] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] + std::min(offset + N, limits[lr]),
                send_buffer(sz, buff, lr, lw).begin<char>());
            send_count[lr * MAX_CONNS + lw]++;
          }
        }
        post_send_all(sz, buff);

        buff++;
        in_flight += 2 * 2 * n_wires;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll(left_, right_, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int wire = wc[i].wr_id & 0xff;
          int lr = wire / MAX_CONNS;
          int lw = wire % MAX_CONNS;

          in_flight--;

          if (work_type == SEND_WR && send_count[wire] < n_steps) {
            int offset = lw * N + send_count[wire] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] + std::min(offset + N, limits[lr]),
                send_buffer(sz, buff, lr, lw).begin<char>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[wire]++;
          }

          else if (work_type == RECV_WR) {
            int offset = lw * N + recv_count[wire] * n_wires * N +
                lr * n_wires * n_bytes_per_wire;
            std::copy(
                recv_buffer(sz, buff, lr, lw).begin<char>(),
                recv_buffer(sz, buff, lr, lw).begin<char>() +
                    std::min(N, limits[lr] - offset),
                out_ptr + recv_offset[lr] + offset);
            recv_count[wire]++;
            if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
              recv_from(sz, buff, lr, lw);
              in_flight++;
            }
          }
        }
      }

      send_offset[0] = (send_offset[0] + out_bytes - n_bytes) % out_bytes;
      recv_offset[0] = (recv_offset[0] + out_bytes - n_bytes) % out_bytes;
      send_offset[1] = (send_offset[1] + n_bytes) % out_bytes;
      recv_offset[1] = (recv_offset[1] + n_bytes) % out_bytes;
      for (int i = 0; i < 2 * MAX_CONNS; i++) {
        send_count[i] = recv_count[i] = 0;
      }
    }
  });
}

void RingGroup::send(const array& input, int dst, Stream stream) {}

void RingGroup::recv(array& out, int src, Stream stream) {}

template <typename T, typename ReduceOp>
void RingGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {}

} // namespace mlx::core::distributed::jaccl
