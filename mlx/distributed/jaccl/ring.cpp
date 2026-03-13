// Copyright © 2026 Apple Inc.

#include "mlx/distributed/jaccl/ring.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/dtype_utils.h"

#include <algorithm>
#include <fstream>
#include <json.hpp>

using json = nlohmann::json;

namespace mlx::core::distributed::jaccl {

// Lightweight group for size-1 sub-groups (no RDMA needed).
class RingLocalGroup : public GroupImpl {
 public:
  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }
  int rank() override { return 0; }
  int size() override { return 1; }
  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    return std::make_shared<RingLocalGroup>();
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

// TCP-based group for sub-groups where opening new RDMA connections would
// deadlock (Apple's TB5 driver doesn't support multiple ibv_context on the
// same physical device). Uses SideChannel TCP star topology for all
// collective operations. Slower than RDMA but avoids device conflicts.
class TCPGroup : public GroupImpl {
 public:
  TCPGroup(int rank, int size, const char* coordinator_addr)
      : rank_(rank),
        size_(size),
        side_channel_(rank, size, coordinator_addr) {
    // Parse coordinator for recursive split()
    std::string addr_str(coordinator_addr);
    auto colon = addr_str.rfind(':');
    if (colon != std::string::npos) {
      coordinator_host_ = addr_str.substr(0, colon);
      coordinator_port_ = std::atoi(addr_str.substr(colon + 1).c_str());
    } else {
      coordinator_host_ = addr_str;
      coordinator_port_ = 29500;
    }
    // Barrier — all ranks must connect before proceeding
    side_channel_.all_gather<int>(0);
  }

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }
  int rank() override { return rank_; }
  int size() override { return size_; }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    key = (key < 0) ? rank_ : key;
    struct SplitInfo { int color; int key; };
    auto all_info = side_channel_.all_gather(SplitInfo{color, key});

    struct Member { int sort_key; int parent_rank; };
    std::vector<Member> members;
    for (int i = 0; i < size_; i++) {
      if (all_info[i].color == color) {
        members.push_back({all_info[i].key * size_ + i, i});
      }
    }
    std::sort(members.begin(), members.end(),
              [](const Member& a, const Member& b) {
                return a.sort_key < b.sort_key;
              });
    int new_size = (int)members.size();
    int new_rank = -1;
    for (int i = 0; i < new_size; i++) {
      if (members[i].parent_rank == rank_) { new_rank = i; break; }
    }
    if (new_size == 1) {
      side_channel_.all_gather<int>(0);
      return std::make_shared<RingLocalGroup>();
    }
    // Derive sub-group coordinator
    const char* my_ip_env = std::getenv("MLX_JACCL_MY_IP");
    std::string my_ip = my_ip_env ? std::string(my_ip_env) : coordinator_host_;
    auto all_ips = side_channel_.all_gather(my_ip);
    std::string coord_host = all_ips[members[0].parent_rank];
    int sub_port = coordinator_port_ + 1000 + color;
    std::string coord_addr = coord_host + ":" + std::to_string(sub_port);
    side_channel_.all_gather<int>(0);
    return std::make_shared<TCPGroup>(new_rank, new_size, coord_addr.c_str());
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      tcp_all_reduce<T>(input, output, stream, detail::SumOp<T>{});
    });
  }
  void all_max(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      tcp_all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
    });
  }
  void all_min(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      tcp_all_reduce<T>(input, output, stream, detail::MinOp<T>{});
    });
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    size_t n_bytes = input.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
      std::vector<char> my_data(in_ptr, in_ptr + n_bytes);
      auto all_data = side_channel_.all_gather(my_data);
      for (int i = 0; i < size_; i++) {
        std::memcpy(out_ptr + i * n_bytes, all_data[i].data(), n_bytes);
      }
    });
  }

  void send(const array&, int, Stream) override {
    throw std::runtime_error("[jaccl] TCPGroup does not support send.");
  }
  void recv(array&, int, Stream) override {
    throw std::runtime_error("[jaccl] TCPGroup does not support recv.");
  }
  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] TCPGroup does not support sum_scatter.");
  }

 private:
  template <typename T, typename ReduceOp>
  void tcp_all_reduce(
      const array& input, array& output, Stream stream, ReduceOp reduce_op) {
    auto in_ptr = input.data<T>();
    auto out_ptr = output.data<T>();
    int64_t count = input.size();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, count, this, reduce_op]() {
      size_t n_bytes = count * sizeof(T);
      std::vector<char> my_data(
          reinterpret_cast<const char*>(in_ptr),
          reinterpret_cast<const char*>(in_ptr) + n_bytes);
      auto all_data = side_channel_.all_gather(my_data);
      // Initialize output with first rank's data
      std::memcpy(out_ptr, all_data[0].data(), n_bytes);
      // Accumulate remaining ranks
      for (int i = 1; i < size_; i++) {
        reduce_op(
            reinterpret_cast<const T*>(all_data[i].data()), out_ptr, count);
      }
    });
  }

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::string coordinator_host_;
  int coordinator_port_;
};

RingGroup::RingGroup(
    int rank,
    int size,
    const std::vector<std::string>& left_devices,
    const std::vector<std::string>& right_devices,
    const char* coordinator_addr)
    : rank_(rank),
      size_(size),
      n_conns_(left_devices.size()),
      side_channel_(rank_, size_, coordinator_addr),
      left_(create_connections(left_devices)),
      right_(create_connections(right_devices)) {
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

  // Store full device matrix for split() — re-read from MLX_IBV_DEVICES
  const char* dev_file = std::getenv("MLX_IBV_DEVICES");
  if (dev_file) {
    std::ifstream f(dev_file);
    if (f.good()) {
      json devices = json::parse(f);
      all_devices_.resize(devices.size());
      for (size_t i = 0; i < devices.size(); i++) {
        all_devices_[i].resize(devices[i].size());
        for (size_t j = 0; j < devices[i].size(); j++) {
          if (devices[i][j].is_string()) {
            all_devices_[i][j].push_back(devices[i][j]);
          } else if (devices[i][j].is_array()) {
            for (auto& name : devices[i][j]) {
              all_devices_[i][j].push_back(name);
            }
          }
        }
      }
    }
  }

  if (left_.size() > RING_MAX_CONNS || right_.size() > RING_MAX_CONNS) {
    std::ostringstream msg;
    msg << "[jaccl] Up to " << RING_MAX_CONNS << " per direction supported but "
        << left_.size() << " were provided.";
    throw std::runtime_error(msg.str());
  }

  // Initialize all the connections and allocate buffers
  initialize();

  // Make sure every node has reached here before continuing
  side_channel_.all_gather<int>(0);

  // Create the ring implementation object
  ring_ = RingImpl(rank_, size_, left_, right_, send_buffers_, recv_buffers_);
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
      for (int j = 0; j < n_conns_ * 2; j++) {
        send_buffers_.emplace_back(FRAME_SIZE * (1 << k));
        recv_buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  // Register the buffers with the corresponding connections
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < n_conns_ * 2; j++) {
        int wire = j % n_conns_;
        int lr = j / n_conns_;
        if (lr) {
          send_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(left_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
        } else {
          send_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
              .register_to_protection_domain(right_[wire].protection_domain);
          recv_buffers_[k * NUM_BUFFERS * n_conns_ * 2 + i * n_conns_ * 2 + j]
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
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, n_bytes, this]() {
    ring_.all_gather(in_ptr, out_ptr, n_bytes, n_conns_);
  });
}

void RingGroup::send(const array& input, int dst, Stream stream) {
  int right = (rank_ + 1) % size_;
  int left = (rank_ + size_ - 1) % size_;
  if (dst != right && dst != left) {
    std::ostringstream msg;
    msg << "[jaccl] In ring mode send is only supported to direct neighbors "
        << "but tried to send to " << dst << " from " << rank_ << std::endl;
    throw std::runtime_error(msg.str());
  }
  auto data = input.data<char>();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.dispatch([data, n_bytes, dst, this]() {
    ring_.send(data, n_bytes, dst, n_conns_);
  });
}

void RingGroup::recv(array& out, int src, Stream stream) {
  int right = (rank_ + 1) % size_;
  int left = (rank_ + size_ - 1) % size_;
  if (src != right && src != left) {
    std::ostringstream msg;
    msg << "[jaccl] In ring mode recv is only supported to direct neighbors "
        << "but tried to recv from " << src << " to " << rank_ << std::endl;
    throw std::runtime_error(msg.str());
  }
  auto data = out.data<char>();
  int64_t n_bytes = out.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([data, n_bytes, src, this]() {
    ring_.recv(data, n_bytes, src, n_conns_);
  });
}

template <typename T, typename ReduceOp>
void RingGroup::all_reduce(
    const array& input,
    array& output,
    Stream stream,
    ReduceOp reduce_op) {
  auto in_ptr = input.data<T>();
  auto out_ptr = output.data<T>();
  int64_t size = input.size();
  int64_t n_bytes = input.nbytes();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(input);
  encoder.set_output_array(output);
  encoder.dispatch([in_ptr, out_ptr, size, n_bytes, this, reduce_op]() {
    if (size < size_ * 2 * n_conns_) {
      ring_.all_reduce<1, T, ReduceOp>(in_ptr, out_ptr, size, 1, reduce_op);
      return;
    }

    if (n_bytes <= 65536) {
      ring_.all_reduce<2, T, ReduceOp>(in_ptr, out_ptr, size, 1, reduce_op);
      return;
    }

    ring_.all_reduce<2, T, ReduceOp>(
        in_ptr, out_ptr, size, n_conns_, reduce_op);
  });
}

std::shared_ptr<GroupImpl> RingGroup::split(int color, int key) {
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
  std::sort(
      members.begin(), members.end(), [](const Member& a, const Member& b) {
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

  // Step 4: Coordinate the sub-group's coordinator address.
  // ALL ranks participate in this all_gather, even size-1 sub-groups.
  // This keeps the parent SideChannel synchronized across all ranks.
  const char* my_ip_env = std::getenv("MLX_JACCL_MY_IP");
  std::string my_ip =
      my_ip_env ? std::string(my_ip_env) : coordinator_host_;
  auto all_ips = side_channel_.all_gather(my_ip);

  // Step 5: Final barrier — all ranks must reach here before any proceeds
  // to create sub-group connections (TCPGroup opens new TCP sockets).
  side_channel_.all_gather<int>(0);

  // Step 6: Create the appropriate sub-group (local decision, no collectives).
  if (new_size == 1) {
    return std::make_shared<RingLocalGroup>();
  }

  // TCPGroup uses TCP (SideChannel) instead of RDMA — avoids the deadlock
  // caused by opening a second ibv_context on an RDMA device already held
  // by the parent RingGroup.
  int sub_rank0_parent = members[0].parent_rank;
  std::string coord_host = all_ips[sub_rank0_parent];
  int sub_port = coordinator_port_ + 1000 + color;
  std::string coord_addr =
      coord_host + ":" + std::to_string(sub_port);

  return std::make_shared<TCPGroup>(
      new_rank, new_size, coord_addr.c_str());
}

} // namespace mlx::core::distributed::jaccl
