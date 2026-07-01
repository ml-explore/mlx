// Copyright © 2026 Apple Inc.

#pragma once

#include <span>

#include "jaccl/rdma.h"

constexpr int RING_MAX_CONNS = 4;

namespace jaccl {

class RingImpl {
 public:
  RingImpl(
      int rank,
      int size,
      std::vector<Connection>& left,
      std::vector<Connection>& right,
      std::vector<SharedBuffer>& send_buffers,
      std::vector<SharedBuffer>& recv_buffers)
      : rank_(rank),
        size_(size),
        n_conns_(left.size()),
        left_(left),
        right_(right),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers) {}

  RingImpl(
      int rank,
      int size,
      Connection* left_begin,
      Connection* right_begin,
      size_t n_conns,
      std::vector<SharedBuffer>& send_buffers,
      std::vector<SharedBuffer>& recv_buffers)
      : rank_(rank),
        size_(size),
        n_conns_(n_conns),
        left_(left_begin, n_conns),
        right_(right_begin, n_conns),
        send_buffers_(send_buffers),
        recv_buffers_(recv_buffers) {}

  RingImpl() : rank_(0), size_(1), n_conns_(0) {}

  template <int MAX_DIR, typename T, typename ReduceOp>
  void all_reduce(
      const T* in_ptr,
      T* out_ptr,
      int64_t size,
      int n_wires,
      ReduceOp reduce_op) {
    // If not inplace all reduce then copy the input to the output first
    if (in_ptr != out_ptr) {
      std::memcpy(out_ptr, in_ptr, size * sizeof(T));
    }

    int64_t chunk_size = (size + size_ - 1) / size_;
    int64_t size_per_wire =
        (chunk_size + (MAX_DIR * n_wires) - 1) / (MAX_DIR * n_wires);

    // Split the reduce scatter + all gather across the available wires. Each
    // wire handles a contiguous slice of each chunk in every direction.
    //
    // TODO: These calls are independent so they should be dispatched to a
    // threadpool and run concurrently instead of sequentially.
    for (int lw = 0; lw < n_wires; lw++) {
      all_reduce_wire<MAX_DIR>(
          out_ptr, size, chunk_size, size_per_wire, n_wires, lw, reduce_op);
    }
  }

  // Perform the ring all reduce (reduce scatter followed by all gather) for a
  // single wire `lw`.
  //
  // Every chunk of `chunk_size` elements is divided into `MAX_DIR` directional
  // regions and each region is further split into `n_wires` contiguous slices
  // of `size_per_wire` elements. This function is responsible for slice `lw` in
  // every direction and only touches `left_[lw]` / `right_[lw]` and the buffers
  // that belong to wire `lw`, so several wires can run concurrently.
  template <int MAX_DIR, typename T, typename ReduceOp>
  void all_reduce_wire(
      T* out_ptr,
      int64_t size,
      int64_t chunk_size,
      int64_t size_per_wire,
      int n_wires,
      int lw,
      ReduceOp reduce_op) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * 2 * MAX_DIR;
    auto [sz, N] = buffer_size_from_message(size_per_wire * sizeof(T));
    N /= sizeof(T);
    int64_t n_steps = (size_per_wire + N - 1) / N;

    // The element offset (within a chunk) of this wire's slice in each
    // direction and the end of each direction's region. Wire slices are
    // contiguous rather than interleaved. Direction `lr` owns the chunk region
    // [lr * n_wires * size_per_wire, (lr + 1) * n_wires * size_per_wire) (the
    // last region is clamped to chunk_size).
    int64_t wire_offset[MAX_DIR];
    int64_t region_end[MAX_DIR];
    for (int lr = 0; lr < MAX_DIR; lr++) {
      wire_offset[lr] = lr * n_wires * size_per_wire +
          static_cast<int64_t>(lw) * size_per_wire;
      region_end[lr] = std::min(chunk_size, (lr + 1) * n_wires * size_per_wire);
    }

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t chunk_multiple_size = size_ * chunk_size;
    int64_t send_offset[MAX_DIR];
    int64_t recv_offset[MAX_DIR];
    int64_t send_limits[MAX_DIR];
    int64_t recv_limits[MAX_DIR];
    int send_count[MAX_DIR] = {0};
    int recv_count[MAX_DIR] = {0};
    send_offset[0] = rank_ * chunk_size;
    recv_offset[0] = ((rank_ + size_ - 1) % size_) * chunk_size;
    send_limits[0] =
        std::min(region_end[0], std::max<int64_t>(0, size - send_offset[0]));
    recv_limits[0] =
        std::min(region_end[0], std::max<int64_t>(0, size - recv_offset[0]));
    if constexpr (MAX_DIR == 2) {
      send_offset[1] = rank_ * chunk_size;
      recv_offset[1] = ((rank_ + 1) % size_) * chunk_size;
      send_limits[1] =
          std::min(region_end[1], std::max<int64_t>(0, size - send_offset[1]));
      recv_limits[1] =
          std::min(region_end[1], std::max<int64_t>(0, size - recv_offset[1]));
    }

    // First reduce scatter
    //
    // Possible perf improvement by not syncing at every step but running ahead
    // as needed.
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        for (int lr = 0; lr < MAX_DIR; lr++) {
          recv_from(sz, buff, lr, lw);
        }
        for (int lr = 0; lr < MAX_DIR; lr++) {
          int64_t offset = wire_offset[lr] + send_count[lr] * N;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_count[lr]++;
          send_to(sz, buff, lr, lw);
        }

        buff++;
        in_flight += 2 * MAX_DIR;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll_wire(lw, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int lr = wc[i].wr_id & 0xff;

          in_flight--;

          if (work_type == SEND_WR && send_count[lr] < n_steps) {
            int64_t offset = wire_offset[lr] + send_count[lr] * N;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[lr]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = wire_offset[lr] + recv_count[lr] * N;
            reduce_op(
                recv_buffer(sz, buff, lr, lw).begin<T>(),
                out_ptr + recv_offset[lr] + offset,
                std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)));
            recv_count[lr]++;
            if (recv_count[lr] + (PIPELINE - 1) < n_steps) {
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
      send_limits[0] =
          std::min(region_end[0], std::max<int64_t>(0, size - send_offset[0]));
      recv_limits[0] =
          std::min(region_end[0], std::max<int64_t>(0, size - recv_offset[0]));
      if constexpr (MAX_DIR == 2) {
        send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
        recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
        send_limits[1] = std::min(
            region_end[1], std::max<int64_t>(0, size - send_offset[1]));
        recv_limits[1] = std::min(
            region_end[1], std::max<int64_t>(0, size - recv_offset[1]));
      }
      for (int lr = 0; lr < MAX_DIR; lr++) {
        send_count[lr] = recv_count[lr] = 0;
      }
    }

    // Secondly all gather
    //
    // The offsets are correct from the scatter reduce
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        for (int lr = 0; lr < MAX_DIR; lr++) {
          recv_from(sz, buff, lr, lw);
        }
        for (int lr = 0; lr < MAX_DIR; lr++) {
          int64_t offset = wire_offset[lr] + send_count[lr] * N;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, send_limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<T>());
          send_count[lr]++;
          send_to(sz, buff, lr, lw);
        }

        buff++;
        in_flight += 2 * MAX_DIR;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll_wire(lw, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int lr = wc[i].wr_id & 0xff;

          in_flight--;

          if (work_type == SEND_WR && send_count[lr] < n_steps) {
            int64_t offset = wire_offset[lr] + send_count[lr] * N;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, send_limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<T>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[lr]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = wire_offset[lr] + recv_count[lr] * N;
            std::copy(
                recv_buffer(sz, buff, lr, lw).begin<T>(),
                recv_buffer(sz, buff, lr, lw).begin<T>() +
                    std::max<int64_t>(0, std::min(N, recv_limits[lr] - offset)),
                out_ptr + recv_offset[lr] + offset);
            recv_count[lr]++;
            if (recv_count[lr] + (PIPELINE - 1) < n_steps) {
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
      send_limits[0] =
          std::min(region_end[0], std::max<int64_t>(0, size - send_offset[0]));
      recv_limits[0] =
          std::min(region_end[0], std::max<int64_t>(0, size - recv_offset[0]));
      if constexpr (MAX_DIR == 2) {
        send_offset[1] = (send_offset[1] + chunk_size) % chunk_multiple_size;
        recv_offset[1] = (recv_offset[1] + chunk_size) % chunk_multiple_size;
        send_limits[1] = std::min(
            region_end[1], std::max<int64_t>(0, size - send_offset[1]));
        recv_limits[1] = std::min(
            region_end[1], std::max<int64_t>(0, size - recv_offset[1]));
      }
      for (int lr = 0; lr < MAX_DIR; lr++) {
        send_count[lr] = recv_count[lr] = 0;
      }
    }
  }

  void
  all_gather(const char* in_ptr, char* out_ptr, int64_t n_bytes, int n_wires) {
    // Copy our data to the appropriate place
    std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

    // Split the all gather across the available wires. Each wire handles a
    // contiguous slice of every rank's data in both directions.
    //
    // TODO: These calls are independent so they should be dispatched to a
    // threadpool and run concurrently instead of sequentially.
    size_t n_bytes_per_wire = (n_bytes + (2 * n_wires) - 1) / (2 * n_wires);
    for (int lw = 0; lw < n_wires; lw++) {
      all_gather_wire(out_ptr, n_bytes, n_bytes_per_wire, lw);
    }
  }

  // Perform the ring all gather for a single wire `lw`.
  //
  // The wire is responsible for the contiguous slice
  //   [lw * n_bytes_per_wire, (lw + 1) * n_bytes_per_wire)
  // of every rank's `n_bytes` region, sent in the left direction (lr == 0),
  // and the mirrored slice sent in the right direction (lr == 1).
  //
  // This function only ever touches `left_[lw]` / `right_[lw]` and the buffers
  // that belong to wire `lw` so several wires can run concurrently.
  void all_gather_wire(
      char* out_ptr,
      int64_t n_bytes,
      size_t n_bytes_per_wire,
      int lw) {
    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE * 2;
    size_t out_bytes = n_bytes * size_;
    auto [sz, N] = buffer_size_from_message(n_bytes_per_wire);
    int n_steps = (n_bytes_per_wire + N - 1) / N;

    // The byte offset (within a rank's region) that this wire is responsible
    // for. Wire slices are contiguous rather than interleaved.
    int64_t wire_offset = static_cast<int64_t>(lw) * n_bytes_per_wire;

    // Counters to maintain the state of transfers
    int in_flight = 0;
    int64_t send_offset[2];
    int64_t recv_offset[2];
    int64_t limits[2];
    int send_count[2] = {0};
    int recv_count[2] = {0};
    send_offset[0] = send_offset[1] = rank_ * n_bytes;
    recv_offset[0] = ((rank_ + size_ - 1) % size_) * n_bytes;
    recv_offset[1] = ((rank_ + 1) % size_) * n_bytes;
    limits[0] = limits[1] = n_bytes;

    // Possible perf improvement by not syncing at every step but running ahead
    // as needed.
    for (int k = 0; k < size_ - 1; k++) {
      // Prefill the pipeline
      int buff = 0;
      while (buff < n_steps && buff < PIPELINE) {
        for (int lr = 0; lr < 2; lr++) {
          recv_from(sz, buff, lr, lw);
        }
        for (int lr = 0; lr < 2; lr++) {
          int64_t offset = wire_offset + send_count[lr] * N;
          std::copy(
              out_ptr + send_offset[lr] + offset,
              out_ptr + send_offset[lr] +
                  std::max(offset, std::min(offset + N, limits[lr])),
              send_buffer(sz, buff, lr, lw).begin<char>());
          send_count[lr]++;
          send_to(sz, buff, lr, lw);
        }

        buff++;
        in_flight += 2 * 2;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = poll_wire(lw, WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int work_type = wc[i].wr_id >> 16;
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int lr = wc[i].wr_id & 0xff;

          in_flight--;

          if (work_type == SEND_WR && send_count[lr] < n_steps) {
            int64_t offset = wire_offset + send_count[lr] * N;
            std::copy(
                out_ptr + send_offset[lr] + offset,
                out_ptr + send_offset[lr] +
                    std::max(offset, std::min(offset + N, limits[lr])),
                send_buffer(sz, buff, lr, lw).begin<char>());
            send_to(sz, buff, lr, lw);
            in_flight++;
            send_count[lr]++;
          }

          else if (work_type == RECV_WR) {
            int64_t offset = wire_offset + recv_count[lr] * N;
            std::copy(
                recv_buffer(sz, buff, lr, lw).begin<char>(),
                recv_buffer(sz, buff, lr, lw).begin<char>() +
                    std::max<int64_t>(0, std::min(N, limits[lr] - offset)),
                out_ptr + recv_offset[lr] + offset);
            recv_count[lr]++;
            if (recv_count[lr] + (PIPELINE - 1) < n_steps) {
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
      send_count[0] = send_count[1] = 0;
      recv_count[0] = recv_count[1] = 0;
    }
  }

  void send(const char* in_ptr, int64_t n_bytes, int dst, int n_wires) {
    int left = (rank_ + size_ - 1) % size_;

    // In the case that size_ == 2 then left == right so we bias send towards
    // left and recv towards right so that the selections will be correct for
    // the 2 node case.
    int dir = dst == left;

    int64_t bytes_per_wire = (n_bytes + n_wires - 1) / n_wires;

    // Split the send across the available wires. Each wire handles the
    // contiguous slice [lw * bytes_per_wire, (lw + 1) * bytes_per_wire).
    //
    // TODO: These calls are independent so they should be dispatched to a
    // threadpool and run concurrently instead of sequentially.
    for (int lw = 0; lw < n_wires; lw++) {
      send_wire(in_ptr, n_bytes, dir, bytes_per_wire, lw);
    }
  }

  // Perform a point-to-point send for a single wire `lw`.
  //
  // Only touches the connection / buffers of wire `lw` so several wires can run
  // concurrently.
  void send_wire(
      const char* in_ptr,
      int64_t n_bytes,
      int dir,
      int64_t bytes_per_wire,
      int lw) {
    auto& conns = dir ? left_ : right_;

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;

    auto [sz, N] = buffer_size_from_message(bytes_per_wire);

    int in_flight = 0;
    int64_t read_offset = std::min(lw * bytes_per_wire, n_bytes);
    int64_t limit = std::min((lw + 1) * bytes_per_wire, n_bytes);

    // Prefill the pipeline
    int buff = 0;
    while (read_offset < limit && buff < PIPELINE) {
      std::copy(
          in_ptr + read_offset,
          in_ptr + std::min(read_offset + N, limit),
          send_buffer(sz, buff, dir, lw).begin<char>());
      send_to(sz, buff, dir, lw);

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
      int n = conns[lw].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;

        in_flight--;

        if (read_offset < limit) {
          std::copy(
              in_ptr + read_offset,
              in_ptr + std::min(read_offset + N, limit),
              send_buffer(sz, buff, dir, lw).begin<char>());
          send_to(sz, buff, dir, lw);

          read_offset += N;
          in_flight++;
        }
      }
    }
  }

  void recv(char* out_ptr, int64_t n_bytes, int src, int n_wires) {
    int right = (rank_ + 1) % size_;

    // In the case that size_ == 2 then left == right so we bias send towards
    // left and recv towards right so that the selections will be correct for
    // the 2 node case.
    int dir = src == right;

    int64_t bytes_per_wire = (n_bytes + n_wires - 1) / n_wires;

    // Split the recv across the available wires. Each wire handles the
    // contiguous slice [lw * bytes_per_wire, (lw + 1) * bytes_per_wire).
    //
    // TODO: These calls are independent so they should be dispatched to a
    // threadpool and run concurrently instead of sequentially.
    for (int lw = 0; lw < n_wires; lw++) {
      recv_wire(out_ptr, n_bytes, dir, bytes_per_wire, lw);
    }
  }

  // Perform a point-to-point recv for a single wire `lw`.
  //
  // Only touches the connection / buffers of wire `lw` so several wires can run
  // concurrently.
  void recv_wire(
      char* out_ptr,
      int64_t n_bytes,
      int dir,
      int64_t bytes_per_wire,
      int lw) {
    auto& conns = dir ? right_ : left_;

    constexpr int PIPELINE = 2;
    constexpr int WC_NUM = PIPELINE;

    auto [sz, N] = buffer_size_from_message(bytes_per_wire);

    int in_flight = 0;
    int64_t write_offset = std::min(lw * bytes_per_wire, n_bytes);
    int64_t limit = std::min((lw + 1) * bytes_per_wire, n_bytes);

    // Prefill the pipeline
    int buff = 0;
    while (write_offset + N * buff < limit && buff < PIPELINE) {
      recv_from(sz, buff, dir, lw);

      buff++;
      in_flight++;
    }

    // Main loop
    while (in_flight > 0) {
      // Poll the hardware for completions.
      //
      // If a recv was completed copy it to the output and if we have more
      // data to fetch post another recv.
      ibv_wc wc[WC_NUM];
      int n = conns[lw].poll(WC_NUM, wc);
      for (int i = 0; i < n; i++) {
        int buff = (wc[i].wr_id >> 8) & 0xff;

        in_flight--;

        std::copy(
            recv_buffer(sz, buff, dir, lw).begin<char>(),
            recv_buffer(sz, buff, dir, lw).begin<char>() +
                std::max<int64_t>(
                    0, std::min<int64_t>(limit - write_offset, N)),
            out_ptr + write_offset);
        write_offset += N;

        if (write_offset + (PIPELINE - 1) * N < limit) {
          recv_from(sz, buff, dir, lw);

          in_flight++;
        }
      }
    }
  }

 private:
  void send_to(int sz, int buff, int left_right, int wire) {
    auto& conns = left_right ? left_ : right_;
    conns[wire].post_send(
        send_buffer(sz, buff, left_right, wire),
        SEND_WR << 16 | buff << 8 | left_right);
  }

  void recv_from(int sz, int buff, int left_right, int wire) {
    auto& conns = left_right ? right_ : left_;
    conns[wire].post_recv(
        recv_buffer(sz, buff, left_right, wire),
        RECV_WR << 16 | buff << 8 | left_right);
  }

  SharedBuffer& send_buffer(int sz, int buff, int left_right, int wire) {
    return send_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 +
         left_right * n_conns_ + wire];
  }

  SharedBuffer& recv_buffer(int sz, int buff, int left_right, int wire) {
    return recv_buffers_
        [sz * NUM_BUFFERS * n_conns_ * 2 + buff * n_conns_ * 2 +
         left_right * n_conns_ + wire];
  }

  // Poll the completion queues that belong to a single wire.
  //
  // A wire always uses both its left and right connections (direction 0 sends
  // right / receives left, direction 1 sends left / receives right) so both are
  // polled here. Restricting polling to a single wire's connections is what
  // allows several wires to run concurrently.
  int poll_wire(int wire, int num_completions, ibv_wc* work_completions) {
    return poll(
        std::span<Connection>(&left_[wire], 1),
        std::span<Connection>(&right_[wire], 1),
        num_completions,
        work_completions);
  }

  int rank_;
  int size_;
  int n_conns_;
  std::span<Connection> left_;
  std::span<Connection> right_;
  std::span<SharedBuffer> send_buffers_;
  std::span<SharedBuffer> recv_buffers_;
};

} // namespace jaccl
