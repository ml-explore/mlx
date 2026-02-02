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
    ReduceOp reduce_op) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, size = input.size(), this, reduce_op]() {
    if (size < size_) {
      // TODO: Maybe allocate dynamically so we don't have the constraint
      // below?
      if (sizeof(T) * size_ > 1024) {
        std::ostringstream msg;
        msg << "Can't perform the ring all reduce of " << size
            << " elements with a ring of size " << size_;
        throw std::runtime_error(msg.str());
      }

      size_t nbytes = size * sizeof(T);
      char buffer[1024];
      std::memset(buffer, 0, size_ * sizeof(T));
      std::memcpy(buffer, in_ptr, nbytes);
      all_reduce_impl<1, T, ReduceOp>(
          reinterpret_cast<T*>(buffer),
          reinterpret_cast<T*>(buffer),
          size_,
          1,
          reduce_op);
      std::memcpy(out_ptr, buffer, nbytes);
      return;
    }

    if (size < 2 * size_) {
      all_reduce_impl<1, T, ReduceOp>(in_ptr, out_ptr, size, 1, reduce_op);
      return;
    }

    all_reduce_impl<2, T, ReduceOp>(
        in_ptr, out_ptr, size, left_.size(), reduce_op);
  });
}

template <int MAX_DIR, typename T, typename ReduceOp>
void RingGroup::all_reduce_impl(
    const T* in_ptr,
    T* out_ptr,
    int64_t size,
    int n_wires,
    ReduceOp reduce_op) {
  // If not inplace all reduce then copy the input to the output first
  if (in_ptr != out_ptr) {
    std::memcpy(out_ptr, in_ptr, size * sizeof(T));
  }

  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE * MAX_CONNS * 2 * MAX_DIR;
  int64_t chunk_size = (size + size_ - 1) / size_;
  int64_t size_per_wire =
      (chunk_size + (MAX_DIR * n_wires) - 1) / (MAX_DIR * n_wires);
  auto [sz, N] = buffer_size_from_message(size_per_wire * sizeof(T));
  N /= sizeof(T);
  int64_t n_steps = (size_per_wire + N - 1) / N;

  // Counters to maintain the state of transfers
  int in_flight = 0;
  int64_t chunk_multiple_size = size_ * chunk_size;
  int64_t send_offset[MAX_DIR];
  int64_t recv_offset[MAX_DIR];
  int64_t send_limits[MAX_DIR];
  int64_t recv_limits[MAX_DIR];
  int send_count[MAX_DIR * MAX_CONNS] = {0};
  int recv_count[MAX_DIR * MAX_CONNS] = {0};
  send_offset[0] = rank_ * chunk_size;
  recv_offset[0] = ((rank_ + size_ - 1) % size_) * chunk_size;
  if constexpr (MAX_DIR == 2) {
    send_offset[1] = rank_ * chunk_size;
    recv_offset[1] = ((rank_ + 1) % size_) * chunk_size;
    send_limits[0] = std::min(
        n_wires * size_per_wire, std::max<int64_t>(0, size - send_offset[0]));
    send_limits[1] =
        std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
    recv_limits[0] = std::min(
        n_wires * size_per_wire, std::max<int64_t>(0, size - recv_offset[0]));
    recv_limits[1] =
        std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
  } else {
    send_limits[0] =
        std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
    recv_limits[0] =
        std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
  }

  // First reduce scatter
  //
  // Possible perf improvement by not syncing at every step but running ahead
  // as needed.
  for (int k = 0; k < size_ - 1; k++) {
    // Prefill the pipeline
    int buff = 0;
    while (buff < n_steps && buff < PIPELINE) {
      post_recv_all<MAX_DIR>(sz, buff, n_wires);
      for (int lr = 0; lr < MAX_DIR; lr++) {
        for (int lw = 0; lw < n_wires; lw++) {
          int64_t offset = lw * N +
              send_count[lr * MAX_CONNS + lw] * n_wires * N +
              lr * n_wires * size_per_wire;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_count[lr * MAX_CONNS + lw]++;
        }
      }
      post_send_all<MAX_DIR>(sz, buff, n_wires);

      buff++;
      in_flight += 2 * MAX_DIR * n_wires;
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
          int64_t offset = lw * N + send_count[wire] * n_wires * N +
              lr * n_wires * size_per_wire;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_to(sz, buff, lr, lw);
          in_flight++;
          send_count[wire]++;
        }

        else if (work_type == RECV_WR) {
          int64_t offset = lw * N + recv_count[wire] * n_wires * N +
              lr * n_wires * size_per_wire;
          reduce_op(
              recv_buffer(sz, buff, lr, lw).begin<T>(),
              out_ptr + recv_offset[lr] + offset,
              std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)));
          recv_count[wire]++;
          if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
            recv_from(sz, buff, lr, lw);
            in_flight++;
          }
        }
      }
    }

    send_offset[0] = (send_offset[0] + chunk_multiple_size - chunk_size) %
        chunk_multiple_size;
    recv_offset[0] = (recv_offset[0] + chunk_multiple_size - chunk_size) %
        chunk_multiple_size;
    if constexpr (MAX_DIR == 2) {
      send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
      recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
      send_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - send_offset[0]));
      send_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
      recv_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - recv_offset[0]));
      recv_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
    } else {
      send_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
      recv_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
    }
    for (int i = 0; i < MAX_DIR * MAX_CONNS; i++) {
      send_count[i] = recv_count[i] = 0;
    }
  }

  // Secondly all gather
  //
  // The offsets are correct from the scatter reduce
  for (int k = 0; k < size_ - 1; k++) {
    // Prefill the pipeline
    int buff = 0;
    while (buff < n_steps && buff < PIPELINE) {
      post_recv_all<MAX_DIR>(sz, buff, n_wires);
      for (int lr = 0; lr < MAX_DIR; lr++) {
        for (int lw = 0; lw < n_wires; lw++) {
          int64_t offset = lw * N +
              send_count[lr * MAX_CONNS + lw] * n_wires * N +
              lr * n_wires * size_per_wire;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_count[lr * MAX_CONNS + lw]++;
        }
      }
      post_send_all<MAX_DIR>(sz, buff, n_wires);

      buff++;
      in_flight += 2 * MAX_DIR * n_wires;
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
          int64_t offset = lw * N + send_count[wire] * n_wires * N +
              lr * n_wires * size_per_wire;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_to(sz, buff, lr, lw);
          in_flight++;
          send_count[wire]++;
        }

        else if (work_type == RECV_WR) {
          int64_t offset = lw * N + recv_count[wire] * n_wires * N +
              lr * n_wires * size_per_wire;
          std::copy(
              recv_buffer(sz, buff, lr, lw).begin<T>(),
              recv_buffer(sz, buff, lr, lw).begin<T>() +
                  std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)),
              out_ptr + recv_offset[lr] + offset);
          recv_count[wire]++;
          if (recv_count[wire] + (PIPELINE - 1) < n_steps) {
            recv_from(sz, buff, lr, lw);
            in_flight++;
          }
        }
      }
    }

    send_offset[0] = (send_offset[0] + chunk_multiple_size - chunk_size) %
        chunk_multiple_size;
    recv_offset[0] = (recv_offset[0] + chunk_multiple_size - chunk_size) %
        chunk_multiple_size;
    if constexpr (MAX_DIR == 2) {
      send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
      recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
      send_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - send_offset[0]));
      send_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[1]));
      recv_limits[0] = std::min(
          n_wires * size_per_wire, std::max<int64_t>(0, size - recv_offset[0]));
      recv_limits[1] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[1]));
    } else {
      send_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - send_offset[0]));
      recv_limits[0] =
          std::min(chunk_size, std::max<int64_t>(0, size - recv_offset[0]));
    }
    for (int i = 0; i < MAX_DIR * MAX_CONNS; i++) {
      send_count[i] = recv_count[i] = 0;
    }
  }
}

} // namespace mlx::core::distributed::jaccl
