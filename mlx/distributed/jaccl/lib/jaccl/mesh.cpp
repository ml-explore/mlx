// Copyright © 2026 Apple Inc.

#include "jaccl/mesh.h"
#include "jaccl/reduction_ops.h"
#include "jaccl/types.h"

#include <algorithm>
#include <sstream>

namespace jaccl {

std::shared_ptr<Group> MeshGroup::split(int color, int key) {
  // Everybody learns everybody's colour and key, so each rank can work out the
  // membership of its own child with no further agreement.
  auto colors = side_channel_.all_gather<int>(color);
  auto keys = side_channel_.all_gather<int>(key);

  std::vector<int> members;
  for (int r = 0; r < size_; r++) {
    if (color >= 0 && colors[r] == color) {
      members.push_back(r);
    }
  }
  // MPI's ordering: by key, ties broken by rank in the parent.
  std::stable_sort(members.begin(), members.end(), [&](int a, int b) {
    return keys[a] != keys[b] ? keys[a] < keys[b] : a < b;
  });

  // A negative colour means take part but join no child. The two all_gathers
  // above have already run, which is the point: this rank has done its share
  // of the collective and can leave.
  if (color < 0) {
    return nullptr;
  }

  int child_rank =
      std::find(members.begin(), members.end(), rank_) - members.begin();

  // A child can only be built from members this rank can already reach. The
  // hostfile is the only description of connectivity the runtime ever sees, so
  // a pair missing there cannot be recovered here.
  std::vector<std::string> child_devices(members.size());
  for (size_t i = 0; i < members.size(); i++) {
    int m = members[i];
    if (m == rank_) {
      continue;
    }
    if (device_names_[m].empty()) {
      std::ostringstream msg;
      msg << "[jaccl] Cannot split: rank " << rank_ << " has no device to rank "
          << m
          << ", which shares its colour. A child group can only contain "
             "directly connected members.";
      throw std::runtime_error(msg.str());
    }
    child_devices[i] = device_names_[m];
  }

  // The child bootstraps over the parent, so the parent has to outlive it, and
  // holding a shared_ptr in the closure is what guarantees that.
  //
  // Every rank of the parent runs this on every child round, including ranks
  // in a different child or in none, because the all_gather below is
  // parent-wide. That is what keeps differently sized children in step: the
  // length negotiation inside the parent's container all_gather covers the
  // whole parent at once, so two children can never disagree about how many
  // bytes are in flight.
  auto self = std::static_pointer_cast<MeshGroup>(shared_from_this());
  AllGatherFn child_all_gather =
      [self, members](const char* src, char* dst, size_t n_bytes) {
        auto all = self->side_channel_.all_gather(
            std::vector<char>(src, src + n_bytes));
        for (size_t i = 0; i < members.size(); i++) {
          std::copy(
              all[members[i]].begin(),
              all[members[i]].begin() + n_bytes,
              dst + i * n_bytes);
        }
      };

  return std::make_shared<MeshGroup>(
      child_rank,
      child_devices,
      SideChannel(child_rank, members.size(), std::move(child_all_gather)));
}

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    SideChannel sc)
    : rank_(rank),
      size_(device_names.size()),
      device_names_(device_names),
      side_channel_(std::move(sc)),
      connections_(create_connections(device_names)) {
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.barrier();

  // Create the mesh implementation object
  mesh_ = MeshImpl(rank_, size_, connections_, buffers_, scatter_buffers_);
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

  // Gather the information to be exchanged, this also serves as a barrier
  // so that all peers have initialized their connections before attempting
  // to transition to RTS.
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
  scatter_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      // Scatter buffers (size_ send slots followed by size_ recv slots)
      for (int j = 0; j < 2 * size_; j++) {
        scatter_buffers_.emplace_back(FRAME_SIZE * (1 << k));
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
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .register_to_protection_domain(conn.protection_domain);
            }
          }
        } else {
          // This is the recv buffer from rank j so register it to rank j's
          // protection domain.
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .register_to_protection_domain(connections_[j].protection_domain);
        }
      }

      // Scatter buffers. Slot p (send to peer p) and slot size_ + p (recv from
      // peer p) are both registered to peer p's protection domain. The slots
      // for our own rank are unused but kept for uniform indexing.
      int scatter_base = k * NUM_BUFFERS * 2 * size_ + i * 2 * size_;
      for (int j = 0; j < size_; j++) {
        if (j == rank_) {
          continue;
        }
        scatter_buffers_[scatter_base + j].register_to_protection_domain(
            connections_[j].protection_domain);
        scatter_buffers_[scatter_base + size_ + j]
            .register_to_protection_domain(connections_[j].protection_domain);
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
  mesh_.all_gather(
      static_cast<const char*>(input), static_cast<char*>(output), n_bytes);
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
  mesh_.send(static_cast<const char*>(input), n_bytes, dst);
}

void MeshGroup::recv(void* output, size_t n_bytes, int src) {
  mesh_.recv(static_cast<char*>(output), n_bytes, src);
}

void MeshGroup::barrier() {
  uint8_t b = 0;
  all_sum(&b, &b, sizeof(b), Dtype::UInt8);
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
  if (size_ > 2 && n_bytes > 32 * 1024) {
    // Large messages are bandwidth bound so use the reduce scatter + all gather
    // path which moves size_x less data per link than the fully connected
    // all_reduce.
    mesh_.all_reduce_scatter_gather(in_ptr, out_ptr, count, reduce_op);
  } else {
    // Small messages are latency bound so use the single phase fully
    // connected all_reduce cause it is a bit better.
    mesh_.all_reduce(in_ptr, out_ptr, count, reduce_op);
  }
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
  mesh_.sum_scatter(in_ptr, out_ptr, count, reduce_op);
}

} // namespace jaccl
