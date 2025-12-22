// Copyright © 2025 Apple Inc.

#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <unistd.h>
#include <fstream>
#include <iostream>

#include <json.hpp>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/distributed/utils.h"
#include "mlx/dtype_utils.h"

#define LOAD_SYMBOL(symbol, variable)                               \
  {                                                                 \
    variable = (decltype(variable))dlsym(librdma_handle_, #symbol); \
    char* error = dlerror();                                        \
    if (error != nullptr) {                                         \
      std::cerr << IBV_TAG << " " << error << std::endl;            \
      librdma_handle_ = nullptr;                                    \
      return;                                                       \
    }                                                               \
  }

constexpr const char* IBV_TAG = "[jaccl]";
constexpr int NUM_BUFFERS = 2;
constexpr int BUFFER_SIZE = 4096;
constexpr int MAX_SEND_WR = 32;
constexpr int MAX_RECV_WR = 32;
constexpr int SEND_WR = 1;
constexpr int RECV_WR = 2;
constexpr int MAX_PEERS = 8;

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
using json = nlohmann::json;
namespace detail = mlx::core::distributed::detail;
namespace allocator = mlx::core::allocator;

struct IBVWrapper {
  IBVWrapper() {
    librdma_handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (librdma_handle_ == nullptr) {
      return;
    }

    LOAD_SYMBOL(ibv_get_device_list, get_device_list);
    LOAD_SYMBOL(ibv_get_device_name, get_device_name);
    LOAD_SYMBOL(ibv_open_device, open_device);
    LOAD_SYMBOL(ibv_free_device_list, free_device_list);
    LOAD_SYMBOL(ibv_close_device, close_device);

    LOAD_SYMBOL(ibv_alloc_pd, alloc_pd);
    LOAD_SYMBOL(ibv_create_qp, create_qp);
    LOAD_SYMBOL(ibv_create_cq, create_cq);
    LOAD_SYMBOL(ibv_destroy_cq, destroy_cq);
    LOAD_SYMBOL(ibv_destroy_qp, destroy_qp);
    LOAD_SYMBOL(ibv_dealloc_pd, dealloc_pd);

    LOAD_SYMBOL(ibv_query_port, query_port);
    LOAD_SYMBOL(ibv_query_gid, query_gid);
    LOAD_SYMBOL(ibv_modify_qp, modify_qp);
    LOAD_SYMBOL(ibv_reg_mr, reg_mr);
    LOAD_SYMBOL(ibv_dereg_mr, dereg_mr);

    // Not really symbols but leaving them here in case they become symbols in
    // the future.
    //
    // LOAD_SYMBOL(ibv_post_send, post_send);
    // LOAD_SYMBOL(ibv_post_recv, post_recv);
    // LOAD_SYMBOL(ibv_poll_cq, poll_cq);
  }

  bool is_available() {
    return librdma_handle_ != nullptr;
  }

  void* librdma_handle_;

  // API
  ibv_device** (*get_device_list)(int*);
  const char* (*get_device_name)(ibv_device*);
  ibv_context* (*open_device)(ibv_device*);
  void (*free_device_list)(ibv_device**);
  int (*close_device)(ibv_context*);

  ibv_pd* (*alloc_pd)(ibv_context*);
  ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*);
  ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int);
  int (*destroy_cq)(ibv_cq*);
  int (*destroy_qp)(ibv_qp*);
  int (*dealloc_pd)(ibv_pd*);

  int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*);
  int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*);
  int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int);
  ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int);
  int (*dereg_mr)(ibv_mr*);
};

IBVWrapper& ibv() {
  static IBVWrapper wrapper;
  return wrapper;
}

template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<
    T,
    std::void_t<typename T::value_type, typename T::iterator>>
    : std::true_type {};

std::ostream& operator<<(std::ostream& os, const ibv_gid& gid) {
  os << std::hex << std::setfill('0');
  for (int i = 0; i < 16; i += 2) {
    uint16_t part = (gid.raw[i] << 8) | gid.raw[i + 1];
    os << std::setw(4) << part;
    if (i < 14)
      os << ":";
  }
  os << std::dec;
  return os;
}

void* page_aligned_alloc(size_t num_bytes) {
  static size_t page_size = sysconf(_SC_PAGESIZE);
  void* buf;
  if (posix_memalign(&buf, page_size, num_bytes)) {
    return nullptr;
  }
  return buf;
}

/**
 * Contains the information that defines a destination to a remote device.
 * Basically we can compute our own destination and share it with remote hosts
 * over the side channel.
 */
struct Destination {
  int local_id;
  int queue_pair_number;
  int packet_sequence_number;
  ibv_gid global_identifier;
};

std::ostream& operator<<(std::ostream& os, const Destination& dst) {
  os << dst.local_id << " " << dst.queue_pair_number << " "
     << dst.packet_sequence_number << " " << dst.global_identifier;
  return os;
}

/**
 * A buffer that can be registered to a number of protection domains.
 */
class SharedBuffer {
 public:
  SharedBuffer(size_t num_bytes)
      : data_(page_aligned_alloc(num_bytes)), num_bytes_(num_bytes) {}
  ~SharedBuffer() {
    for (auto& [pd, mr] : memory_regions_) {
      ibv().dereg_mr(mr);
    }
    if (data_ != nullptr) {
      std::free(data_);
    }
  }

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;
  SharedBuffer(SharedBuffer&& b) : data_(nullptr), num_bytes_(0) {
    std::swap(data_, b.data_);
    std::swap(num_bytes_, b.num_bytes_);
    std::swap(memory_regions_, b.memory_regions_);
  }

  void register_to_protection_domain(ibv_pd* protection_domain) {
    auto [it, inserted] = memory_regions_.insert({protection_domain, nullptr});
    if (!inserted) {
      throw std::runtime_error(
          "[jaccl] Buffer can be registered once per protection domain");
    }

    it->second = ibv().reg_mr(
        protection_domain,
        data_,
        num_bytes_,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_REMOTE_WRITE);
    if (!it->second) {
      throw std::runtime_error("[jaccl] Register memory region failed");
    }
  }

  size_t size() const {
    return num_bytes_;
  }

  uint32_t local_key(ibv_pd* protection_domain) const {
    return memory_regions_.at(protection_domain)->lkey;
  }

  ibv_sge to_scatter_gather_entry(ibv_pd* protection_domain) const {
    ibv_sge entry;
    entry.addr = reinterpret_cast<uintptr_t>(data_);
    entry.length = size();
    entry.lkey = local_key(protection_domain);
    return entry;
  }

  template <typename T>
  T* data() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* begin() {
    return static_cast<T*>(data_);
  }

  template <typename T>
  T* end() {
    return static_cast<T*>(data_) + size() / sizeof(T);
  }

 private:
  void* data_;
  size_t num_bytes_;
  std::unordered_map<ibv_pd*, ibv_mr*> memory_regions_;
};

/**
 * Manipulates an RDMA connection. Enables (among other things)
 *
 *   - Creating a queue pair
 *   - Sending and receiving
 *   - Checking completion
 */
struct Connection {
  ibv_context* ctx;
  ibv_pd* protection_domain;
  ibv_cq* completion_queue;
  ibv_qp* queue_pair;
  Destination src; // holds the local information

  Connection(ibv_context* ctx_)
      : ctx(ctx_),
        protection_domain(nullptr),
        completion_queue(nullptr),
        queue_pair(nullptr) {
    src.local_id = -1;
  }

  Connection(Connection&& c) : Connection(nullptr) {
    std::swap(ctx, c.ctx);
    std::swap(protection_domain, c.protection_domain);
    std::swap(completion_queue, c.completion_queue);
    std::swap(queue_pair, c.queue_pair);
    std::swap(src, c.src);
  }

  Connection(const Connection&) = delete;
  Connection& operator=(Connection&) = delete;

  ~Connection() {
    if (queue_pair != nullptr) {
      ibv().destroy_qp(queue_pair);
    }
    if (completion_queue != nullptr) {
      ibv().destroy_cq(completion_queue);
    }
    if (protection_domain != nullptr) {
      ibv().dealloc_pd(protection_domain);
    }
    if (ctx != nullptr) {
      ibv().close_device(ctx);
    }
  }

  void allocate_protection_domain() {
    protection_domain = ibv().alloc_pd(ctx);
    if (protection_domain == nullptr) {
      throw std::runtime_error("[jaccl] Couldn't allocate protection domain");
    }
  }

  void create_completion_queue(int num_entries) {
    completion_queue = ibv().create_cq(ctx, num_entries, nullptr, nullptr, 0);
    if (completion_queue == nullptr) {
      throw std::runtime_error("[jaccl] Couldn't create completion queue");
    }
  }

  void create_queue_pair() {
    ibv_qp_init_attr init_attr;
    init_attr.qp_context = ctx;
    init_attr.qp_context = ctx;
    init_attr.send_cq = completion_queue;
    init_attr.recv_cq = completion_queue;
    init_attr.srq = nullptr;
    init_attr.cap.max_send_wr = MAX_SEND_WR;
    init_attr.cap.max_recv_wr = MAX_RECV_WR;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = 0;
    init_attr.qp_type = IBV_QPT_UC;
    init_attr.sq_sig_all = 0;

    queue_pair = ibv().create_qp(protection_domain, &init_attr);

    if (queue_pair == nullptr) {
      throw std::runtime_error("[jaccl] Couldn't create queue pair");
    }
  }

  const Destination& info() {
    if (queue_pair == nullptr || src.local_id >= 0) {
      return src;
    }

    ibv_port_attr port_attr;
    ibv().query_port(ctx, 1, &port_attr);
    ibv_gid gid;
    ibv().query_gid(ctx, 1, 1, &gid);

    src.local_id = port_attr.lid;
    src.queue_pair_number = queue_pair->qp_num;
    src.packet_sequence_number = 7; // TODO: Change to sth random
    src.global_identifier = gid;

    return src;
  }

  void queue_pair_init() {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Changing queue pair to INIT failed with errno " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  void queue_pair_rtr(const Destination& dst) {
    ibv_qp_attr attr = {};
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.rq_psn = dst.packet_sequence_number;
    attr.dest_qp_num = dst.queue_pair_number;
    attr.ah_attr.dlid = dst.local_id;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 0;

    if (dst.global_identifier.global.interface_id) {
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.hop_limit = 1;
      attr.ah_attr.grh.dgid = dst.global_identifier;
      attr.ah_attr.grh.sgid_index = 1;
    }

    int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN;

    if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Changing queue pair to RTR failed with errno " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  void queue_pair_rts() {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = src.packet_sequence_number;

    int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;

    if (int status = ibv().modify_qp(queue_pair, &attr, mask); status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Changing queue pair to RTS failed with errno " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  void post_send(const SharedBuffer& buff, uint64_t work_request_id) {
    ibv_send_wr work_request, *bad_work_request;

    auto entry = buff.to_scatter_gather_entry(protection_domain);
    work_request.wr_id = work_request_id;
    work_request.sg_list = &entry;
    work_request.num_sge = 1;
    work_request.opcode = IBV_WR_SEND;
    work_request.send_flags = IBV_SEND_SIGNALED;
    work_request.next = nullptr;

    if (int status =
            ibv_post_send(queue_pair, &work_request, &bad_work_request);
        status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Send failed with error code " << status;
      throw std::invalid_argument(msg.str());
    }
  }

  void post_recv(const SharedBuffer& buff, uint64_t work_request_id) {
    ibv_recv_wr work_request, *bad_work_request;

    auto entry = buff.to_scatter_gather_entry(protection_domain);
    work_request.wr_id = work_request_id;
    work_request.sg_list = &entry;
    work_request.num_sge = 1;
    work_request.next = nullptr;

    if (int status =
            ibv_post_recv(queue_pair, &work_request, &bad_work_request);
        status != 0) {
      std::ostringstream msg;
      msg << "[jaccl] Recv failed with error code " << status;
      throw std::invalid_argument(msg.str());
    }
  }
};

/**
 * Implement a TCP side channel to exchange information about the RDMA
 * connections.
 *
 * Implements a simple all gather where every node sends to rank 0 and rank 0
 * broadcasts to every node.
 */
class SideChannel {
 public:
  SideChannel(int rank, int size, const char* addr) : rank_(rank), size_(size) {
    auto address = detail::parse_address(addr);

    if (rank_ == 0) {
      detail::TCPSocket server(IBV_TAG);
      server.listen(IBV_TAG, address);

      for (int i = 0; i < size - 1; i++) {
        sockets_.push_back(server.accept(IBV_TAG));
      }

      std::vector<int> ranks(size - 1);
      for (int i = 0; i < size - 1; i++) {
        sockets_[i].recv(
            IBV_TAG, reinterpret_cast<char*>(&ranks[i]), sizeof(int));
        ranks[i]--;
      }
      for (int i = 0; i < size - 1; i++) {
        while (i != ranks[i]) {
          std::swap(sockets_[i], sockets_[ranks[i]]);
          std::swap(ranks[i], ranks[ranks[i]]);
        }
      }
    } else {
      sockets_.push_back(detail::TCPSocket::connect(
          IBV_TAG, address, 4, 1000, [](int attempt, int wait) {
            std::cerr << IBV_TAG << " Connection attempt " << attempt
                      << " waiting " << wait << " ms" << std::endl;
          }));
      sockets_[0].send(IBV_TAG, reinterpret_cast<char*>(&rank_), sizeof(int));
    }
  }

  SideChannel(const SideChannel&) = delete;
  SideChannel& operator=(const SideChannel&) = delete;

  SideChannel(SideChannel&& sc)
      : rank_(sc.rank_), size_(sc.size_), sockets_(std::move(sc.sockets_)) {
    sc.rank_ = -1;
    sc.size_ = -1;
  }

  template <typename T>
  std::vector<T> all_gather(const T& v) {
    std::vector<T> result(size_);

    // T is a container of stuff like std::vector or std::string
    if constexpr (is_container<T>::value) {
      using U = typename T::value_type;

      // Share the lengths first and set the communication size to be the
      // maximum length of the containers.
      auto lengths = all_gather<int>(v.size());
      auto max_len = *std::max_element(lengths.begin(), lengths.end());
      for (auto& s : result) {
        s.resize(max_len);
      }

      // All gather of length max_len
      if (rank_ == 0) {
        std::copy(v.begin(), v.end(), result[rank_].begin());
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].recv(IBV_TAG, result[i].data(), sizeof(U) * max_len);
        }
        for (int i = 1; i < size_; i++) {
          for (int j = 0; j < size_; j++) {
            sockets_[i - 1].send(
                IBV_TAG, result[j].data(), sizeof(U) * max_len);
          }
        }
      } else {
        std::copy(v.begin(), v.end(), result[rank_].begin());
        sockets_[0].send(IBV_TAG, result[rank_].data(), sizeof(U) * max_len);
        for (int i = 0; i < size_; i++) {
          sockets_[0].recv(IBV_TAG, result[i].data(), sizeof(U) * max_len);
        }
      }

      // Resize the outputs back to the original length
      for (int i = 0; i < size_; i++) {
        result[i].resize(lengths[i]);
      }
    }

    // T is a scalar
    else {
      if (rank_ == 0) {
        result[rank_] = v;
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].recv(IBV_TAG, &result[i], sizeof(T));
        }
        for (int i = 1; i < size_; i++) {
          sockets_[i - 1].send(IBV_TAG, result.data(), size_ * sizeof(T));
        }
      } else {
        sockets_[0].send(IBV_TAG, &v, sizeof(T));
        sockets_[0].recv(IBV_TAG, result.data(), size_ * sizeof(T));
      }
    }

    return result;
  }

 private:
  int rank_;
  int size_;
  std::vector<detail::TCPSocket> sockets_;
};

/**
 * Manages a set of connections. Among other things it uses a side channel to
 * exchange the necessary information and then configure the connections to be
 * ready for RDMA operations.
 */
class ConnectionManager {
 public:
  ConnectionManager(
      int rank,
      const std::vector<std::string>& device_names,
      const char* coordinator_addr)
      : rank_(rank),
        size_(device_names.size()),
        side_channel_(rank_, size_, coordinator_addr) {
    create_contexts(device_names);
    if (connections_[rank_].ctx != nullptr) {
      throw std::runtime_error("[jaccl] Malformed device file");
    }
  }

  int rank() const {
    return rank_;
  }

  int size() const {
    return size_;
  }

  /**
   * Performs the connection initialization. Namely, after this call all
   * Connection objects should have a queue pair in RTS state.
   */
  void initialize(int num_buffers, size_t num_bytes) {
    // Create the queue pairs
    for (auto& conn : connections_) {
      if (conn.ctx == nullptr) {
        continue;
      }
      conn.allocate_protection_domain();
      conn.create_completion_queue(MAX_SEND_WR + MAX_RECV_WR);
      conn.create_queue_pair();
    }

    allocate_buffers(num_buffers, num_bytes);

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

  void allocate_buffers(int num_buffers, size_t num_bytes) {
    // Deregister any buffers and free the memory
    buffers_.clear();

    // Allocate the memory
    for (int i = 0; i < num_buffers; i++) {
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(num_bytes);
      }
    }

    for (int i = 0; i < num_buffers; i++) {
      for (int j = 0; j < size_; j++) {
        // This is our send buffer so register it with all pds so we can send
        // it to all connected devices.
        if (j == rank_) {
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[i * size_ + j].register_to_protection_domain(
                  conn.protection_domain);
            }
          }
        }

        // This is the recv buffer from rank j so register it to rank j's
        // protection domain.
        else {
          buffers_[i * size_ + j].register_to_protection_domain(
              connections_[j].protection_domain);
        }
      }
    }
  }

  void send_to(int rank, int buff) {
    connections_[rank].post_send(
        buffers_[buff * size_ + rank_], SEND_WR << 16 | buff << 8 | rank);
  }

  void recv_from(int rank, int buff) {
    connections_[rank].post_recv(
        buffers_[buff * size_ + rank], RECV_WR << 16 | buff << 8 | rank);
  }

  /**
   * Poll all connections and save the work completions and return the
   * corresponding length.
   */
  int poll(int num_completions, ibv_wc* work_completions) {
    int completions = 0;
    for (int r = 0; r < size_; r++) {
      if (r == rank_) {
        continue;
      }
      if (completions >= num_completions) {
        return completions;
      }

      int c = ibv_poll_cq(
          connections_[r].completion_queue,
          num_completions - completions,
          work_completions + completions);

      completions += c;
    }
    return completions;
  }

  /**
   *
   */
  int poll(int rank, int num_completions, ibv_wc* work_completions) {
    return ibv_poll_cq(
        connections_[rank].completion_queue, num_completions, work_completions);
  }

  SharedBuffer& send_buffer(int buff) {
    return buffers_[buff * size_ + rank_];
  }

  SharedBuffer& buffer(int rank, int buff) {
    return buffers_[buff * size_ + rank];
  }

  void barrier() {
    side_channel_.all_gather<int>(0);
  }

 private:
  void create_contexts(const std::vector<std::string>& device_names) {
    int num_devices = 0;
    ibv_device** devices = ibv().get_device_list(&num_devices);
    for (auto& name : device_names) {
      // Empty so add a nullptr context
      if (name.empty()) {
        connections_.emplace_back(nullptr);
        continue;
      }

      // Search for the name and try to open the device
      for (int i = 0; i < num_devices; i++) {
        if (name == ibv().get_device_name(devices[i])) {
          auto ctx = ibv().open_device(devices[i]);
          if (ctx == nullptr) {
            std::ostringstream msg;
            msg << "[jaccl] Could not open device " << name;
            throw std::runtime_error(msg.str());
          }
          connections_.emplace_back(ctx);
          break;
        }
      }
    }
    ibv().free_device_list(devices);
  }

  int rank_;
  int size_;
  SideChannel side_channel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
};

std::vector<std::string> load_device_names(int rank, const char* dev_file) {
  std::vector<std::string> device_names;
  std::ifstream f(dev_file);

  json devices = json::parse(f);
  devices = devices[rank];
  for (auto it = devices.begin(); it != devices.end(); it++) {
    std::string n;
    if (!it->is_null()) {
      n = *it;
    }
    device_names.emplace_back(std::move(n));
  }

  return device_names;
}

namespace mlx::core::distributed::jaccl {

class IBVGroup : public GroupImpl {
 public:
  IBVGroup(ConnectionManager cm)
      : cm_(std::move(cm)), rank_(cm.rank()), size_(cm.size()) {}

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }

  int rank() override {
    return cm_.rank();
  }

  int size() override {
    return cm_.size();
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      all_reduce<T>(input, output, stream, detail::SumOp<T>{});
    });
  }

  void all_max(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      all_reduce<T>(input, output, stream, detail::MaxOp<T>{});
    });
  }

  void all_min(const array& input, array& output, Stream stream) override {
    dispatch_all_types(output.dtype(), [&](auto type_tag) {
      using T = MLX_GET_TYPE(type_tag);
      all_reduce<T>(input, output, stream, detail::MinOp<T>{});
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
      // Copy our data to the appropriate place
      std::memcpy(out_ptr + rank_ * n_bytes, in_ptr, n_bytes);

      // Fully connected all gather
      char* data = out_ptr;
      char* our_data = out_ptr + rank_ * n_bytes;
      constexpr int64_t N = BUFFER_SIZE;
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
        post_recv_all(buff);
        std::copy(
            our_data + read_offset,
            our_data + std::min(read_offset + N, total),
            cm_.send_buffer(buff).begin<char>());
        post_send_all(buff);

        buff++;
        in_flight += 2 * num_peers;
        read_offset += N;
      }

      // Main loop
      //
      // Keep going until we have no longer data in flight.
      while (in_flight > 0) {
        ibv_wc wc[WC_NUM];
        int n = cm_.poll(WC_NUM, wc);
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
                  cm_.send_buffer(buff).begin<char>());
              post_send_all(buff);

              completed_send_count[buff] = 0;
              in_flight += num_peers;
              read_offset += N;
            }
          }

          // Recv completed. If we have more chunks then post another recv.
          else if (work_type == RECV_WR) {
            std::copy(
                cm_.buffer(rank, buff).begin<char>(),
                cm_.buffer(rank, buff).begin<char>() +
                    std::min(N, total - write_offset[rank]),
                data + rank * n_bytes + write_offset[rank]);
            write_offset[rank] += N;
            if (write_offset[rank] + N * (PIPELINE - 1) < total) {
              cm_.recv_from(rank, buff);
              in_flight++;
            }
          }
        }
      }
    });
  }

  void send(const array& input, int dst, Stream stream) override {
    auto data = input.data<char>();
    int64_t n_bytes = input.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.dispatch([data, n_bytes, dst, this]() {
      constexpr int PIPELINE = 2;
      constexpr int WC_NUM = PIPELINE;
      constexpr int N = BUFFER_SIZE;

      int in_flight = 0;
      int64_t read_offset = 0;

      // Prefill the pipeline
      int buff = 0;
      while (read_offset < n_bytes && buff < PIPELINE) {
        std::copy(
            data + read_offset,
            data + std::min(read_offset + N, n_bytes),
            cm_.send_buffer(buff).begin<char>());
        cm_.send_to(dst, buff);

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
        int n = cm_.poll(WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int rank = wc[i].wr_id & 0xff;

          in_flight--;

          if (read_offset < n_bytes) {
            std::copy(
                data + read_offset,
                data + std::min(read_offset + N, n_bytes),
                cm_.send_buffer(buff).begin<char>());
            cm_.send_to(dst, buff);

            read_offset += N;
            in_flight++;
          }
        }
      }
    });
  }

  void recv(array& out, int src, Stream stream) override {
    auto data = out.data<char>();
    int64_t n_bytes = out.nbytes();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_output_array(out);
    encoder.dispatch([data, n_bytes, src, this]() {
      constexpr int PIPELINE = 2;
      constexpr int WC_NUM = PIPELINE;
      constexpr int N = BUFFER_SIZE;

      int in_flight = 0;
      int64_t write_offset = 0;

      // Prefill the pipeline
      int buff = 0;
      while (write_offset < n_bytes && buff < PIPELINE) {
        cm_.recv_from(src, buff);

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
        int n = cm_.poll(WC_NUM, wc);
        for (int i = 0; i < n; i++) {
          int buff = (wc[i].wr_id >> 8) & 0xff;
          int rank = wc[i].wr_id & 0xff;

          in_flight--;

          std::copy(
              cm_.buffer(src, buff).begin<char>(),
              cm_.buffer(src, buff).begin<char>() +
                  std::min(n_bytes - write_offset, static_cast<int64_t>(N)),
              data + write_offset);
          write_offset += N;

          if (write_offset + (PIPELINE - 1) * N < n_bytes) {
            cm_.recv_from(src, buff);

            in_flight++;
          }
        }
      }
    });
  }

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[jaccl] sum_scatter not supported.");
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[jaccl] Group split not supported.");
  }

 private:
  void post_recv_all(int buffer) {
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      cm_.recv_from(i, buffer);
    }
  }

  void post_send_all(int buffer) {
    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      cm_.send_to(i, buffer);
    }
  }

  template <typename T, typename ReduceOp>
  void all_reduce(
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
      constexpr int64_t N = BUFFER_SIZE / sizeof(T);
      constexpr int PIPELINE = 2;
      constexpr int WC_NUM = PIPELINE * MAX_PEERS * 2;
      int64_t total = static_cast<int64_t>(size);
      int num_peers = size_ - 1;

      // Counters to maintain the state of transfers
      int in_flight = 0;
      int read_offset = 0;
      int completed_send_count[PIPELINE] = {0};
      int completed_recv_begin[MAX_PEERS] = {0};
      int completed_recv_end[MAX_PEERS] = {0};

      // Prefill the pipeline
      int buff = 0;
      while (read_offset < total && buff < PIPELINE) {
        post_recv_all(buff);
        std::copy(
            data + read_offset,
            data + std::min(read_offset + N, total),
            cm_.send_buffer(buff).begin<T>());
        post_send_all(buff);

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
        int n = cm_.poll(WC_NUM, wc);
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
                  cm_.send_buffer(buff).begin<T>());
              post_send_all(buff);

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
                cm_.buffer(r, buff).begin<T>(),
                data + w,
                std::min(N, total - w));
            w += N;
            s++;
            if (w + (PIPELINE - 1) * N < total) {
              cm_.recv_from(r, buff);
              in_flight++;
            }
          }
          completed_recv_begin[r] = s;
        }
      }
    });
  }

  ConnectionManager cm_;
  int rank_;
  int size_;
};

bool is_available() {
  return ibv().is_available();
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  const char* dev_file = std::getenv("MLX_IBV_DEVICES");
  const char* coordinator = std::getenv("MLX_JACCL_COORDINATOR");
  const char* rank_str = std::getenv("MLX_RANK");

  if (!is_available() || !dev_file || !coordinator || !rank_str) {
    if (strict) {
      std::ostringstream msg;
      msg << "[jaccl] You need to provide via environment variables a rank (MLX_RANK), "
          << "a device file (MLX_IBV_DEVICES) and a coordinator ip/port (MLX_JACCL_COORDINATOR) "
          << "but provided MLX_RANK=\"" << ((rank_str) ? rank_str : "")
          << "\", MLX_IBV_DEVICES=\"" << ((dev_file) ? dev_file : "")
          << "\" and MLX_JACCL_COORDINATOR=\""
          << ((coordinator) ? coordinator : "");
      throw std::runtime_error(msg.str());
    }
    return nullptr;
  }

  auto rank = std::atoi(rank_str);
  auto device_names = load_device_names(rank, dev_file);

  auto cm = ConnectionManager(rank, device_names, coordinator);
  if (cm.size() > MAX_PEERS) {
    std::ostringstream msg;
    msg << "[jaccl] The maximum number of supported peers is " << MAX_PEERS
        << " but " << cm.size() << " was provided";
    throw std::runtime_error(msg.str());
  }

  cm.initialize(NUM_BUFFERS, BUFFER_SIZE);
  cm.barrier();

  return std::make_shared<IBVGroup>(std::move(cm));
}

} // namespace mlx::core::distributed::jaccl
