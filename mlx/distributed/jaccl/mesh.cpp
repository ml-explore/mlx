// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

namespace mlx::core::distributed::jaccl {

// Lightweight group for size-1 sub-groups (no RDMA needed).
class LocalGroup : public GroupImpl {
 public:
  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }
  int rank() override {
    return 0;
  }
  int size() override {
    return 1;
  }
  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    return std::make_shared<LocalGroup>();
  }
  void all_sum(const array& input, array& output, Stream stream) override {
    copy_input(input, output, stream);
  }
  void all_max(const array& input, array& output, Stream stream) override {
    copy_input(input, output, stream);
  }
  void all_min(const array& input, array& output, Stream stream) override {
    copy_input(input, output, stream);
  }
  void all_gather(const array& input, array& output, Stream stream) override {
    copy_input(input, output, stream);
  }
  void send(const array&, int, Stream) override {
    throw std::runtime_error("[jaccl] Cannot send in a size-1 group.");
  }
  void recv(array&, int, Stream) override {
    throw std::runtime_error("[jaccl] Cannot recv in a size-1 group.");
  }
  void sum_scatter(const array& input, array& output, Stream stream) override {
    copy_input(input, output, stream);
  }

 private:
  void copy_input(const array& input, array& output, Stream stream) {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch(
        [in = input.data<char>(),
         out = output.data<char>(),
         n = input.nbytes()]() {
          if (in != out)
            std::memcpy(out, in, n);
        });
  }
};

MeshGroup::MeshGroup(
    int rank,
    const std::vector<std::string>& device_names,
    const char* coordinator_addr)
    : rank_(rank),
      size_(device_names.size()),
      side_channel_(rank_, size_, coordinator_addr),
      connections_(create_connections(device_names)),
      device_names_(device_names) {
  // Parse and store coordinator host:port for split() sub-group derivation
  std::string addr_str(coordinator_addr);
  auto colon = addr_str.rfind(':');
  if (colon != std::string::npos) {
    coordinator_host_ = addr_str.substr(0, colon);
    coordinator_port_ = std::atoi(addr_str.substr(colon + 1).c_str());
  } else {
    coordinator_host_ = addr_str;
    coordinator_port_ = 29500;
  }
  if (size_ > MESH_MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The JACCL mesh supports up to " << MESH_MAX_PEERS
        << " peers but " << size_ << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);

  // Create the mesh implementation object
  mesh_ = MeshImpl(rank_, size_, connections_, buffers_);
  ring_ = RingImpl(
      rank_,
      size_,
      &connections_[(rank_ + size_ - 1) % size_],
      &connections_[(rank_ + 1) % size_],
      1,
      ring_send_buffers_,
      ring_recv_buffers_);
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
  ring_send_buffers_.clear();
  ring_recv_buffers_.clear();

  // Allocate the memory
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
      // Ring buffers (1 for each direction)
      for (int j = 0; j < 2; j++) {
        ring_send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        ring_recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      // Mesh buffers
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

      // Ring buffers (see ring group for the logic below)
      // We register send buffers to both the right and the left.
      int left = (rank_ + size_ - 1) % size_;
      int right = (rank_ + 1) % size_;
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[right].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 0]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_send_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[left].protection_domain);
      ring_recv_buffers_[k * NUM_BUFFERS * 2 + i * 2 + 1]
          .register_to_protection_domain(connections_[right].protection_domain);
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
    mesh_.all_gather(in_ptr, out_ptr, n_bytes);
  });
}

void MeshGroup::send(const array& input, int dst, Stream stream) {
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch(
      [data, n_bytes, dst, this]() { mesh_.send(data, n_bytes, dst); });
}

void MeshGroup::recv(array& out, int src, Stream stream) {
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch(
      [data, n_bytes, src, this]() { mesh_.recv(data, n_bytes, src); });
}

template <typename T, typename ReduceOp>
void MeshGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  int64_t size = input.size();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, size, this, reduce_op]() {
    if (size_ > 2 &&
        ((std::is_same_v<T, bfloat16_t> && size > 65536) ||
         size >= 8 * 1024 * 1024 / sizeof(T))) {
      ring_.all_reduce<2>(in_ptr, out_ptr, size, 1, reduce_op);
    } else {
      mesh_.all_reduce(in_ptr, out_ptr, size, reduce_op);
    }
  });
}

std::shared_ptr<GroupImpl> MeshGroup::split(int color, int key) {
  key = (key < 0) ? rank_ : key;

  // Step 1: Exchange split info via parent's side channel.
  // IMPORTANT: All SideChannel collectives must be called by ALL ranks in
  // the same order. No early returns between collectives!
  struct SplitInfo {
    int color;
    int key;
  };
  SplitInfo my_info{color, key};
  auto all_info = side_channel_.all_gather(my_info);

  // Step 2: Find all ranks with the same color, sorted by (key, parent_rank).
  struct Member {
    int sort_key;
    int parent_rank;
  };
  std::vector<Member> members;
  for (int i = 0; i < size_; i++) {
    if (all_info[i].color == color) {
      members.push_back({all_info[i].key * size_ + i, i});
    }
  }
  std::sort(members.begin(), members.end(), [](const Member& a, const Member& b) {
    return a.sort_key < b.sort_key;
  });

  int new_size = static_cast<int>(members.size());

  // Step 3: Determine this rank's position in the sub-group.
  int new_rank = -1;
  for (int i = 0; i < new_size; i++) {
    if (members[i].parent_rank == rank_) {
      new_rank = i;
      break;
    }
  }

  // Step 4: Build device names for the sub-group.
  // device_names_ is indexed by parent rank; extract the subset for sub-group
  // peers, with empty string for self.
  std::vector<std::string> sub_devices(new_size);
  for (int i = 0; i < new_size; i++) {
    if (i == new_rank) {
      sub_devices[i] = ""; // self
    } else {
      sub_devices[i] = device_names_[members[i].parent_rank];
    }
  }

  // Step 5: Coordinate the sub-group's coordinator address.
  // ALL ranks participate, even size-1 sub-groups, to keep SideChannel in sync.
  const char* my_ip_env = std::getenv("MLX_JACCL_MY_IP");
  std::string my_ip = my_ip_env ? std::string(my_ip_env) : coordinator_host_;
  auto all_ips = side_channel_.all_gather(my_ip);

  // The sub-group coordinator is the rank that becomes new_rank 0.
  int sub_rank0_parent = members[0].parent_rank;
  std::string coord_host = all_ips[sub_rank0_parent];
  int sub_port = coordinator_port_ + 1000 + color;
  std::string coord_addr = coord_host + ":" + std::to_string(sub_port);

  // Step 6: Final barrier — all ranks must reach here before any proceeds
  // to create sub-group connections.
  side_channel_.all_gather<int>(0);

  // Step 7: Create the appropriate sub-group (local decision, no collectives).
  if (new_size == 1) {
    return std::make_shared<LocalGroup>();
  }
  return std::make_shared<MeshGroup>(
      new_rank, sub_devices, coord_addr.c_str());
}

} // namespace mlx::core::distributed::jaccl
