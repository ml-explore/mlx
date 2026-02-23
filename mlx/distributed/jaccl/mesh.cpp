// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

constexpr int MAX_PEERS = 8;

namespace mlx::core::distributed::jaccl {

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const char* coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      side_channel_(rank_, size_, coordinator_addr),
      connections_(create_connections(device_names)) {
  if (size_ > MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);
}

void MeshGroup::initialize() {
  // Create the queue pairs
  for (auto& conn : connections_) {
    if (conn.ctx == nullptr) {
      continue;
    }
    conn.allocate_protection_domain();
    conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
    conn.create_queue_pair();
  }

  allocate_buffers();

  // First init all connections
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    connections_[peer].queue_pair_init();
  }

  // Gather the information to be exchanged, this also serves as a barrier so
  // that all peers have initialized their connections before attempting to
  // transition to RTS.
  std::vector<Destination> info;
  for (auto& conn : connections_) {
    info.emplace_back(conn.info());
  }
  auto all_infos = side_channel_.all_gather(info);

  // Transition queue pairs to RTS
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_) {
      continue;
    }
    auto peer_info = all_infos[peer][rank_];
    connections_[peer].queue_pair_rtr(peer_info);
    connections_[peer].queue_pair_rts();
  }
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        // This is our send buffer so register it with all pds so we can send
        // it to all connected devices.
        if (j == rank_) {
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .register_to_protection_domain(conn.protection_domain);
            }
          }
        }

        // This is the recv buffer from rank j so register it to rank j's
        // protection domain.
        else {
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
        }
      }
    }
  }
}

void MeshGroup::all_sum(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::SumOp<T>{});
  });
}

void MeshGroup::all_max(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
  });
}

void MeshGroup::all_min(const array& input, array& output, Stream stream) {
  dispatch_all_types(output.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    all_reduce<T>(input, output, stream, detail::MinOp<T>{});
  });
}

void MeshGroup::all_gather(const array& input, array& output, Stream stream) {
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  size_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    // Fully connected all gather
    char* data = out_ptr;
    char* our_data = out_ptr + rank_ * n_bytes;
    auto [sz, N] = buffer_size_from_message(n_bytes);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(n_bytes);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int write_offset[MAX_PEERS] = {0};

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);
      std::copy(
          our_data + read_offset,
          our_data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<char>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        // Send completed. If all sends completed then send the next chunk.
        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                our_data + read_offset,
                our_data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<char>());
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        // Recv completed. If we have more chunks then post another recv.
        else if (work_type == RECV_WR) {
          std::copy(
              recv_buffer(sz, buff, rank).begin<char>(),
              recv_buffer(sz, buff, rank).begin<char>() +
                  std::min(N, total - write_offset[rank]),
              data + rank * n_bytes + write_offset[rank]);
          write_offset[rank] += N;
          if (write_offset[rank] + N * (PIPELINE - 1) < total) {
            recv_from(sz, rank, buff);
            in_flight++;
          }
        }
      }
    }
  });
}

void MeshGroup::send(const array& input, int dst, Stream stream) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([data, n_bytes, dst, this]() {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < n_bytes && buff < PIPELINE) {
      std::copy(
          data + read_offset,
          data + std::min(read_offset + N, n_bytes),
          send_buffer(sz, buff).begin<char>());
      send_to(sz, dst, buff);

      buff++;
      read_offset += N;
      in_flight++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed and we have more data to send then go ahead
      // and send them.
      ibv_wc wc[WC_NUM];
      int n = connections_[dst].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (read_offset < n_bytes) {
          std::copy(
              data + read_offset,
              data + std::min(read_offset + N, n_bytes),
              send_buffer(sz, buff).begin<char>());
          send_to(sz, dst, buff);

          read_offset += N;
          in_flight++;
        }
      }
    }
  });
}

void MeshGroup::recv(array& out, int src, Stream stream) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([data, n_bytes, src, this]() {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;
    auto [sz, N] = buffer_size_from_message(n_bytes);

    int in_flight = 0;
    int64_t write_offset = 0;

    // Prefill the pipeline
    int buff = 0;
    while (N * buff < n_bytes && buff < PIPELINE) {
      recv_from(sz, src, buff);

      in_flight++;
      buff++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
      ibv_wc wc[WC_NUM];
      int n = connections_[src].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        std::copy(
            recv_buffer(sz, buff, src).begin<char>(),
            recv_buffer(sz, buff, src).begin<char>() +
                std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
            data + write_offset);
        write_offset += N;

        if (write_offset + (PIPELINE - 1) * N < n_bytes) {
          recv_from(sz, src, buff);

          in_flight++;
        }
      }
    }
  });
}

void MeshGroup::blocking_send(const array& input, int dst) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE;
  auto [sz, N] = buffer_size_from_message(n_bytes);
  int in_flight = 0;
  int64_t read_offset = 0;
  int buff = 0;
  while (read_offset < n_bytes && buff < PIPELINE) {
    std::copy(
        data + read_offset,
        data + std::min(read_offset + N, n_bytes),
        send_buffer(sz, buff).begin<char>());
    send_to(sz, dst, buff);
    buff++;
    read_offset += N;
    in_flight++;
  }
  auto deadline_send =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (in_flight > 0) {
    ibv_wc wc[WC_NUM];
    int n = connections_[dst].poll(WC_NUM, wc);
    if (n < 0) {
      throw std::runtime_error(
          "[jaccl] blocking_send: poll() returned " + std::to_string(n));
    }
    if (n == 0 && std::chrono::steady_clock::now() > deadline_send) {
      throw std::runtime_error(
          "[jaccl] blocking_send: timeout waiting for CQ completion (in_flight=" +
          std::to_string(in_flight) + ")");
    }
    for (int i = 0; i < n; i++) {
      int work_type = wc[i].wr_id >> 16;
      int b = (wc[i].wr_id >> 8) & 0xff;
      if (wc[i].status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            "[jaccl] blocking_send: WC error status=" +
            std::to_string(wc[i].status) +
            " wr_id=" + std::to_string(wc[i].wr_id));
      }
      if (work_type != SEND_WR) {
        throw std::runtime_error(
            "[jaccl] blocking_send: unexpected work_type=" +
            std::to_string(work_type));
      }
      in_flight--;
      if (read_offset < n_bytes) {
        std::copy(
            data + read_offset,
            data + std::min(read_offset + N, n_bytes),
            send_buffer(sz, b).begin<char>());
        send_to(sz, dst, b);
        read_offset += N;
        in_flight++;
      }
    }
  }
}

void MeshGroup::blocking_recv(array& out, int src) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE;
  auto [sz, N] = buffer_size_from_message(n_bytes);
  int in_flight = 0;
  int64_t write_offset = 0;
  int buff = 0;
  while (N * buff < n_bytes && buff < PIPELINE) {
    recv_from(sz, src, buff);
    in_flight++;
    buff++;
  }
  auto deadline_recv =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (in_flight > 0) {
    ibv_wc wc[WC_NUM];
    int n = connections_[src].poll(WC_NUM, wc);
    if (n < 0) {
      throw std::runtime_error(
          "[jaccl] blocking_recv: poll() returned " + std::to_string(n));
    }
    if (n == 0 && std::chrono::steady_clock::now() > deadline_recv) {
      throw std::runtime_error(
          "[jaccl] blocking_recv: timeout waiting for CQ completion (in_flight=" +
          std::to_string(in_flight) + ")");
    }
    for (int i = 0; i < n; i++) {
      int work_type = wc[i].wr_id >> 16;
      int b = (wc[i].wr_id >> 8) & 0xff;
      if (wc[i].status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            "[jaccl] blocking_recv: WC error status=" +
            std::to_string(wc[i].status) +
            " wr_id=" + std::to_string(wc[i].wr_id));
      }
      if (work_type != RECV_WR) {
        throw std::runtime_error(
            "[jaccl] blocking_recv: unexpected work_type=" +
            std::to_string(work_type));
      }
      in_flight--;
      std::copy(
          recv_buffer(sz, b, src).begin<char>(),
          recv_buffer(sz, b, src).begin<char>() +
              std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
          data + write_offset);
      write_offset += N;
      if (write_offset + (PIPELINE - 1) * N < n_bytes) {
        recv_from(sz, src, b);
        in_flight++;
      }
    }
  }
}

void MeshGroup::blocking_all_to_all(const array& input, array& output) {
  if (size_ != 2) {
    throw std::runtime_error(
        "[jaccl] blocking_all_to_all currently supports size == 2, got " +
        std::to_string(size_) + ".");
  }
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  if (in_ptr == out_ptr) {
    throw std::runtime_error(
        "[jaccl] in-place blocking_all_to_all is not supported.");
  }
  int64_t n_bytes = static_cast<int64_t>(input.nbytes());
  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE * 2;
  int peer = 1 - rank_;
  int64_t per_peer_bytes = n_bytes / size_;
  std::memcpy(
      out_ptr + rank_ * per_peer_bytes,
      in_ptr + rank_ * per_peer_bytes,
      per_peer_bytes);
  if (per_peer_bytes == 0)
    return;
  char* send_src = const_cast<char*>(in_ptr) + peer * per_peer_bytes;
  char* recv_dst = out_ptr + peer * per_peer_bytes;
  auto [sz, N] = buffer_size_from_message(per_peer_bytes);
  int in_flight = 0;
  int64_t read_offset = 0;
  int64_t write_offset = 0;
  int buff = 0;
  while (read_offset < per_peer_bytes && buff < PIPELINE) {
    recv_from(sz, peer, buff);
    in_flight++;
    std::copy(
        send_src + read_offset,
        send_src +
            std::min(read_offset + static_cast<int64_t>(N), per_peer_bytes),
        send_buffer(sz, buff).begin<char>());
    send_to(sz, peer, buff);
    in_flight++;
    read_offset += N;
    buff++;
  }
  auto deadline_a2a =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (in_flight > 0) {
    ibv_wc wc[WC_NUM];
    int n = connections_[peer].poll(WC_NUM, wc);
    if (n < 0) {
      throw std::runtime_error(
          "[jaccl] blocking_all_to_all: poll() returned " + std::to_string(n));
    }
    if (n == 0 && std::chrono::steady_clock::now() > deadline_a2a) {
      throw std::runtime_error(
          "[jaccl] blocking_all_to_all: timeout waiting for CQ completion (in_flight=" +
          std::to_string(in_flight) + ")");
    }
    for (int i = 0; i < n; i++) {
      int work_type = wc[i].wr_id >> 16;
      int b = (wc[i].wr_id >> 8) & 0xff;
      if (wc[i].status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            "[jaccl] blocking_all_to_all: WC error status=" +
            std::to_string(wc[i].status) +
            " wr_id=" + std::to_string(wc[i].wr_id));
      }
      in_flight--;
      if (work_type == SEND_WR) {
        if (read_offset < per_peer_bytes) {
          std::copy(
              send_src + read_offset,
              send_src +
                  std::min(
                      read_offset + static_cast<int64_t>(N), per_peer_bytes),
              send_buffer(sz, b).begin<char>());
          send_to(sz, peer, b);
          in_flight++;
          read_offset += N;
        }
      } else if (work_type == RECV_WR) {
        std::copy(
            recv_buffer(sz, b, peer).begin<char>(),
            recv_buffer(sz, b, peer).begin<char>() +
                std::min(
                    static_cast<int64_t>(N), per_peer_bytes - write_offset),
            recv_dst + write_offset);
        write_offset += N;
        if (write_offset + (PIPELINE - 1) * N < per_peer_bytes) {
          recv_from(sz, peer, b);
          in_flight++;
        }
      }
    }
  }
}

void MeshGroup::blocking_sendrecv(
    const array& send_buf,
    size_t send_nbytes,
    array& recv_buf,
    size_t recv_nbytes,
    int peer,
    detail::ExchangeTag tag) {
  // Skip no-op
  if (send_nbytes == 0 && recv_nbytes == 0)
    return;

  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE * 2;

  // We use the existing send_buffer/recv_buffer infrastructure.
  // For the tagged sendrecv, we encode tag in the upper 16 bits of wr_id:
  //   wr_id = (tag << 48) | (work_type << 16) | (buff << 8) | rank
  // But since existing code uses wr_id as: work_type(16-23) | buff(8-15) |
  // rank(0-7) and poll() only looks at bits 0-23, we can safely put tag in bits
  // 48-63. However, ibv_wc.wr_id is uint64_t, so this is safe.
  //
  // Actually, to keep it simple and avoid any potential issues with existing
  // code, we'll just use the standard wr_id format (SEND_WR/RECV_WR, buff,
  // rank) and NOT encode the tag in wr_id. The tag is implicit since
  // blocking_sendrecv uses its own poll loop and won't see other CQ entries
  // from other operations (we're in a blocking context inside enc.dispatch
  // lambda).

  auto send_ptr = static_cast<const char*>(send_buf.data<void>());
  auto recv_ptr = static_cast<char*>(recv_buf.data<void>());

  // Determine buffer size for RDMA frames
  size_t max_bytes = std::max(send_nbytes, recv_nbytes);
  if (max_bytes == 0)
    max_bytes = 1; // avoid 0-size
  auto [sz, N] = buffer_size_from_message(static_cast<int64_t>(max_bytes));

  int in_flight = 0;
  int64_t send_offset = 0;
  int64_t recv_write_offset = 0;
  int64_t send_total = static_cast<int64_t>(send_nbytes);
  int64_t recv_total = static_cast<int64_t>(recv_nbytes);

  // Prefill: post recvs first, then sends
  int buff = 0;
  int recv_buffs_posted = 0;
  int send_buffs_posted = 0;

  // Post recv buffers first (if we have data to receive)
  if (recv_total > 0) {
    buff = 0;
    while (static_cast<int64_t>(N) * buff < recv_total && buff < PIPELINE) {
      recv_from(sz, peer, buff);
      in_flight++;
      recv_buffs_posted++;
      buff++;
    }
  }

  // Post sends
  if (send_total > 0) {
    buff = 0;
    while (send_offset < send_total && buff < PIPELINE) {
      auto chunk = std::min(send_total - send_offset, static_cast<int64_t>(N));
      std::copy(
          send_ptr + send_offset,
          send_ptr + send_offset + chunk,
          send_buffer(sz, buff).begin<char>());
      send_to(sz, peer, buff);
      in_flight++;
      send_offset += N;
      send_buffs_posted++;
      buff++;
    }
  }

  // Poll loop
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (in_flight > 0) {
    ibv_wc wc[WC_NUM];
    int n = connections_[peer].poll(WC_NUM, wc);
    if (n < 0) {
      throw std::runtime_error(
          "[jaccl] blocking_sendrecv: poll() returned " + std::to_string(n));
    }
    if (n == 0 && std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error(
          "[jaccl] blocking_sendrecv: timeout (in_flight=" +
          std::to_string(in_flight) +
          " tag=" + std::to_string(static_cast<int>(tag)) + ")");
    }
    if (n > 0) {
      deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    }

    for (int i = 0; i < n; i++) {
      int work_type = wc[i].wr_id >> 16;
      int b = (wc[i].wr_id >> 8) & 0xff;

      if (wc[i].status != IBV_WC_SUCCESS) {
        throw std::runtime_error(
            "[jaccl] blocking_sendrecv: WC error status=" +
            std::to_string(wc[i].status) +
            " wr_id=" + std::to_string(wc[i].wr_id) +
            " tag=" + std::to_string(static_cast<int>(tag)));
      }

      in_flight--;

      if (work_type == SEND_WR) {
        // Send completed — post next chunk if available
        if (send_offset < send_total) {
          auto chunk =
              std::min(send_total - send_offset, static_cast<int64_t>(N));
          std::copy(
              send_ptr + send_offset,
              send_ptr + send_offset + chunk,
              send_buffer(sz, b).begin<char>());
          send_to(sz, peer, b);
          in_flight++;
          send_offset += N;
        }
      } else if (work_type == RECV_WR) {
        // Recv completed — copy data and post next recv if needed
        auto chunk =
            std::min(static_cast<int64_t>(N), recv_total - recv_write_offset);
        if (chunk > 0) {
          std::copy(
              recv_buffer(sz, b, peer).begin<char>(),
              recv_buffer(sz, b, peer).begin<char>() + chunk,
              recv_ptr + recv_write_offset);
          recv_write_offset += N;
        }
        if (recv_write_offset + (PIPELINE - 1) * static_cast<int64_t>(N) <
            recv_total) {
          recv_from(sz, peer, b);
          in_flight++;
        }
      }
    }
  }
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
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
    // If not inplace all reduce then copy the input to the output first
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }

    // Fully connected all reduce
    T* data = out_ptr;
    auto [sz, buffer_size] = buffer_size_from_message(size * sizeof(T));
    int64_t N = buffer_size / sizeof(T);
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * MAX_PEERS * 2;
    int64_t total = static_cast<int64_t>(size);
    int num_peers = size_ - 1;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t read_offset = 0;
    int completed_send_count[PIPELINE] = {0};
    int completed_recv_begin[MAX_PEERS] = {0};
    int completed_recv_end[MAX_PEERS] = {0};

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < total && buff < PIPELINE) {
      post_recv_all(sz, buff);
      std::copy(
          data + read_offset,
          data + std::min(read_offset + N, total),
          send_buffer(sz, buff).begin<T>());
      post_send_all(sz, buff);

      buff++;
      in_flight += 2 * num_peers;
      read_offset += N;
    }

    // Main loop
    //
    // Keep going until we have no longer data in flight.
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a send was completed mark how many completions we have received
      // for that buffer. If we have sent the buffer to all peers we can
      // reuse the buffer so copy the next chunk of data and send it to all.
      //
      // If a receive is completed then advance the pointer of completed
      // receives.
      ibv_wc wc[WC_NUM];
      int n = poll(connections_, WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int buff = (wc[i].wr_id >> 8) & 0xff;
        int rank = wc[i].wr_id & 0xff;

        in_flight--;

        if (work_type == SEND_WR && read_offset < total) {
          completed_send_count[buff]++;
          if (completed_send_count[buff] == num_peers) {
            std::copy(
                data + read_offset,
                data + std::min(read_offset + N, total),
                send_buffer(sz, buff).begin<T>());
            post_send_all(sz, buff);

            completed_send_count[buff] = 0;
            in_flight += num_peers;
            read_offset += N;
          }
        }

        else if (work_type == RECV_WR) {
          completed_recv_end[rank]++;
        }
      }

      // Process the completed recv
      //
      // For each rank we have a range of completed recv defined by a begin
      // and end inclusive and exlusive in standard C++ fashion.
      //
      // When there is an unprocessed receive we first check if we have
      // finished sending the write location. If so then we reduce in-place
      // and then check if there is more to be received and post a recv.
      for (int r = 0; r < size_; r++) {
        int s = completed_recv_begin[r];
        int e = completed_recv_end[r];
        int w = s * N;
        while (w < read_offset && e - s > 0) {
          int buff = s % PIPELINE;
          reduce_op(
              recv_buffer(sz, buff, r).begin<T>(),
              data + w,
              std::min(N, total - w));
          w += N;
          s++;
          if (w + (PIPELINE - 1) * N < total) {
            recv_from(sz, r, buff);
            in_flight++;
          }
        }
        completed_recv_begin[r] = s;
      }
    }
  });
}

void MeshGroup::all_to_all(const array& input, array& output, Stream stream) {
  if (size_ != 2) {
    throw std::runtime_error(
        "[jaccl] all_to_all currently supports size == 2, got " +
        std::to_string(size_) + ".");
  }
  auto in_ptr = input.data<char>();
  auto out_ptr = output.data<char>();
  if (in_ptr == out_ptr) {
    throw std::runtime_error(
        "[jaccl] in-place all_to_all is not supported (input/output alias).");
  }
  int64_t n_bytes = static_cast<int64_t>(input.nbytes());

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * 2;

    int peer = 1 - rank_;
    int64_t per_peer_bytes = n_bytes / size_;

    // Local chunk: input[rank] -> output[rank]
    std::memcpy(
        out_ptr + rank_ * per_peer_bytes,
        in_ptr + rank_ * per_peer_bytes,
        per_peer_bytes);

    if (per_peer_bytes == 0)
      return;

    char* send_src = const_cast<char*>(in_ptr) + peer * per_peer_bytes;
    char* recv_dst = out_ptr + peer * per_peer_bytes;

    auto [sz, N] = buffer_size_from_message(per_peer_bytes);

    int in_flight = 0;
    int64_t read_offset = 0;
    int64_t write_offset = 0;

    // Prefill: recv-first (deadlock prevention)
    int buff = 0;
    while (read_offset < per_peer_bytes && buff < PIPELINE) {
      recv_from(sz, peer, buff);
      in_flight++;

      std::copy(
          send_src + read_offset,
          send_src +
              std::min(read_offset + static_cast<int64_t>(N), per_peer_bytes),
          send_buffer(sz, buff).begin<char>());
      send_to(sz, peer, buff);
      in_flight++;

      read_offset += N;
      buff++;
    }

    // Single poll loop
    while (in_flight > 0) {
      ibv_wc wc[WC_NUM];
      int n = connections_[peer].poll(WC_NUM, wc);

      for (int i = 0; i < n; i++) {
        int work_type = wc[i].wr_id >> 16;
        int b = (wc[i].wr_id >> 8) & 0xff;

        in_flight--;

        if (work_type == SEND_WR) {
          if (read_offset < per_peer_bytes) {
            std::copy(
                send_src + read_offset,
                send_src +
                    std::min(
                        read_offset + static_cast<int64_t>(N), per_peer_bytes),
                send_buffer(sz, b).begin<char>());
            send_to(sz, peer, b);
            in_flight++;
            read_offset += N;
          }
        } else if (work_type == RECV_WR) {
          std::copy(
              recv_buffer(sz, b, peer).begin<char>(),
              recv_buffer(sz, b, peer).begin<char>() +
                  std::min(
                      static_cast<int64_t>(N), per_peer_bytes - write_offset),
              recv_dst + write_offset);
          write_offset += N;

          if (write_offset + (PIPELINE - 1) * N < per_peer_bytes) {
            recv_from(sz, peer, b);
            in_flight++;
          }
        }
      }
    }
  });
}

} // namespace mlx::core::distributed::jaccl
