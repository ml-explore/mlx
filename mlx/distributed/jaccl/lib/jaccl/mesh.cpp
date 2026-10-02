// Copyright © 2026 Apple Inc.

#include "jaccl/mesh.h"
#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

namespace jaccl {

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::vector<std::string>>& device_names,
    SideChannel sc)
    : rank_(rank),
      size_(device_names[0].size()),
      n_wires_(device_names.size()),
      side_channel_(std::move(sc)),
      pool_(n_wires_ - 1) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  for (auto& names : device_names) {
    connections_.push_back(create_connections(names));
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.barrier();

  // Create the mesh implementation objects
  for (int w = 0; w < n_wires_; w++) {
    meshes_.emplace_back(
        rank_, size_, connections_[w], buffers_[w], scatter_buffers_[w]);
  }
}

void MeshGroup::initialize() {
  // Create the queue pairs
  for (auto& wire : connections_) {
    for (auto& conn : wire) {
      if (conn.ctx == nullptr) {
        continue;
      }
      conn.allocate_protection_domain();
      conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
      conn.create_queue_pair();
    }
  }

  allocate_buffers();

  // First init all connections
  for (auto& wire : connections_) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      wire[peer].queue_pair_init();
    }
  }

  // Gather the information to be exchanged, this also serves as a barrier
  // so that all peers have initialized their connections before attempting
  // to transition to RTS.
  std::vector<Destination> info;
  for (auto& wire : connections_) {
    for (auto& conn : wire) {
      info.emplace_back(conn.info());
    }
  }
  auto all_infos = side_channel_.all_gather(info);

  // Transition queue pairs to RTS
  for (int w = 0; w < n_wires_; w++) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_) {
        continue;
      }
      auto peer_info = all_infos[peer][w * size_ + rank_];
      connections_[w][peer].queue_pair_rtr(peer_info);
      connections_[w][peer].queue_pair_rts();
    }
  }
}

void MeshGroup::allocate_buffers() {
  // Deregister any buffers and free the memory
  buffers_.clear();
  scatter_buffers_.clear();
  buffers_.resize(n_wires_);
  scatter_buffers_.resize(n_wires_);

  for (int w = 0; w < n_wires_; w++) {
    auto& conns = connections_[w];
    auto& buffers = buffers_[w];
    auto& scatter_buffers = scatter_buffers_[w];

    // Allocate the memory
    for (int k = 0; k < BUFFER_SIZES; k++) {
      for (int i = 0; i < NUM_BUFFERS; i++) {
        // Mesh buffers
        for (int j = 0; j < size_; j++) {
          buffers.emplace_back(FRAME_SIZE * (1 << k));
        }
        // Scatter buffers (size_ send slots followed by size_ recv slots)
        for (int j = 0; j < 2 * size_; j++) {
          scatter_buffers.emplace_back(FRAME_SIZE * (1 << k));
        }
      }
    }

    for (int k = 0; k < BUFFER_SIZES; k++) {
      for (int i = 0; i < NUM_BUFFERS; i++) {
        // Mesh buffers
        for (int j = 0; j < size_; j++) {
          if (j == rank_) {
            // This is our send buffer so register it with all pds so we can
            // send it to all connected devices.
            for (auto& conn : conns) {
              if (conn.ctx != nullptr) {
                buffers[k * NUM_BUFFERS * size_ + i * size_ + j]
                    .register_to_protection_domain(conn.protection_domain);
              }
            }
          } else {
            // This is the recv buffer from rank j so register it to rank j's
            // protection domain.
            buffers[k * NUM_BUFFERS * size_ + i * size_ + j]
                .register_to_protection_domain(conns[j].protection_domain);
          }
        }

        // Scatter buffers. Slot p (send to peer p) and slot size_ + p (recv
        // from peer p) are both registered to peer p's protection domain. The
        // slots for our own rank are unused but kept for uniform indexing.
        int scatter_base = k * NUM_BUFFERS * 2 * size_ + i * 2 * size_;
        for (int j = 0; j < size_; j++) {
          if (j == rank_) {
            continue;
          }
          scatter_buffers[scatter_base + j].register_to_protection_domain(
              conns[j].protection_domain);
          scatter_buffers[scatter_base + size_ + j]
              .register_to_protection_domain(conns[j].protection_domain);
        }
      }
    }
  }
}

void MeshGroup::all_sum(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, SumOp<T>{});
  });
}

void MeshGroup::all_max(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MaxOp<T>{});
  });
}

void MeshGroup::all_min(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    all_reduce<T>(input, output, n_bytes, MinOp<T>{});
  });
}

void MeshGroup::all_gather(const void* input, void* output, size_t n_bytes) {
  auto in_ptr = static_cast<const char*>(input);
  auto out_ptr = static_cast<char*>(output);
  split_wires(n_bytes, n_bytes, [&](int w, int64_t offset, int64_t n) {
    meshes_[w].all_gather(in_ptr + offset, out_ptr + offset, n, n_bytes);
  });
}

void MeshGroup::sum_scatter(
    const void* input,
    void* output,
    size_t n_bytes,
    int dtype) {
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = JACCL_GET_TYPE(type_tag);
    reduce_scatter<T>(input, output, n_bytes, SumOp<T>{});
  });
}

void MeshGroup::send(const void* input, size_t n_bytes, int dst) {
  auto in_ptr = static_cast<const char*>(input);
  split_wires(n_bytes, n_bytes, [&](int w, int64_t offset, int64_t n) {
    meshes_[w].send(in_ptr + offset, n, dst);
  });
}

void MeshGroup::recv(void* output, size_t n_bytes, int src) {
  auto out_ptr = static_cast<char*>(output);
  split_wires(n_bytes, n_bytes, [&](int w, int64_t offset, int64_t n) {
    meshes_[w].recv(out_ptr + offset, n, src);
  });
}

void MeshGroup::barrier() {
  uint8_t b = 0;
  all_sum(&b, &b, sizeof(b), Dtype::UInt8);
}

template <typename Fn>
void MeshGroup::split_wires(size_t n_bytes, int64_t total, Fn&& fn) {
  int n_wires = n_bytes >= MESH_MULTI_WIRE_MIN_BYTES ? n_wires_ : 1;
  int64_t per_wire = (total + n_wires - 1) / n_wires;
  dispatch_wires(&pool_, n_wires, [&](int w) {
    int64_t offset = std::min(total, w * per_wire);
    fn(w, offset, std::min(per_wire, total - offset));
  });
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  auto in_ptr = static_cast<const T*>(input);
  auto out_ptr = static_cast<T*>(output);
  int64_t count = n_bytes / sizeof(T);
  split_wires(n_bytes, count, [&](int w, int64_t offset, int64_t n) {
    if (size_ > 2 && n * sizeof(T) > 32 * 1024) {
      // Large messages are bandwidth bound so use the reduce scatter + all
      // gather path which moves size_x less data per link than the fully
      // connected all_reduce.
      meshes_[w].all_reduce_scatter_gather(
          in_ptr + offset, out_ptr + offset, n, reduce_op);
    } else {
      // Small messages are latency bound so use the single phase fully
      // connected all_reduce cause it is a bit better.
      meshes_[w].all_reduce(in_ptr + offset, out_ptr + offset, n, reduce_op);
    }
  });
}

template <typename T, typename ReduceOp>
void MeshGroup::reduce_scatter(
    const void* input,
    void* output,
    size_t n_bytes,
    ReduceOp reduce_op) {
  // n_bytes is the size of the output (one chunk). The input holds size_ such
  // chunks laid out contiguously.
  auto in_ptr = static_cast<const T*>(input);
  auto out_ptr = static_cast<T*>(output);
  int64_t count = n_bytes / sizeof(T);
  split_wires(n_bytes, count, [&](int w, int64_t offset, int64_t n) {
    meshes_[w].sum_scatter(
        in_ptr + offset, out_ptr + offset, n, count, reduce_op);
  });
}

} // namespace jaccl
