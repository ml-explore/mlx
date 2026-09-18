// Copyright © 2025 Apple Inc.

#pragma once

#include <infiniband/verbs.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "jaccl/tcp.h"

constexpr const char* IBV_TAG = "[jaccl]";
constexpr int SEND_WR = 1;
constexpr int RECV_WR = 2;
constexpr int MAX_SEND_WR = 32;
constexpr int MAX_RECV_WR = 32;
constexpr int BUFFER_SIZES = 8;
constexpr int NUM_BUFFERS = 2;
constexpr int FRAME_SIZE = 4096;

namespace {

template <typename T, typename = void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<
    T,
    std::void_t<typename T::value_type, typename T::iterator>>
    : std::true_type {};

inline std::pair<int, int64_t> buffer_size_from_message(int64_t msg) {
  if (__builtin_available(macOS 26.3, iOS 26.3, tvOS 26.3, visionOS 26.3, *)) {
    for (int k = BUFFER_SIZES - 1; k > 0; k--) {
      if (msg >= FRAME_SIZE * (1 << k)) {
        return {k, FRAME_SIZE * (1 << k)};
      }
    }
  }
  return {0, FRAME_SIZE};
}

} // namespace

namespace jaccl {

/**
 * Wrapper for the ibverbs API.
 */
struct IBVWrapper {
  IBVWrapper();
  bool is_available() {
    return librdma_handle_ != nullptr;
  }

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

 private:
  void* librdma_handle_;
};

IBVWrapper& ibv();

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

/**
 * A buffer that can be registered to a number of protection domains.
 */
class SharedBuffer {
 public:
  SharedBuffer(size_t num_bytes);
  SharedBuffer(SharedBuffer&& b);
  ~SharedBuffer();

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;

  void register_to_protection_domain(ibv_pd* protection_domain);

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

  Connection(ibv_context* ctx_);
  Connection(Connection&& c);

  Connection(const Connection&) = delete;
  Connection& operator=(Connection&) = delete;

  ~Connection();
  void allocate_protection_domain();
  void create_completion_queue(int num_entries);
  void create_queue_pair();

  const Destination& info();
  void queue_pair_init();
  void queue_pair_rtr(const Destination& dst);
  void queue_pair_rts();

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

  int poll(int num_completions, ibv_wc* work_completions) {
    return ibv_poll_cq(completion_queue, num_completions, work_completions);
  }
};

std::vector<Connection> create_connections(
    const std::vector<std::string>& device_names);

inline int poll(
    std::span<const Connection> connections,
    int num_completions,
    ibv_wc* work_completions) {
  int completions = 0;
  for (auto& c : connections) {
    if (c.ctx == nullptr) {
      continue;
    }
    if (completions >= num_completions) {
      return completions;
    }

    int n = ibv_poll_cq(
        c.completion_queue,
        num_completions - completions,
        work_completions + completions);

    completions += n;
  }
  return completions;
}

inline int poll(
    std::span<const Connection> connections_1,
    std::span<const Connection> connections_2,
    int num_completions,
    ibv_wc* work_completions) {
  int completions = 0;
  completions += poll(connections_1, num_completions, work_completions);
  completions += poll(
      connections_2,
      num_completions - completions,
      work_completions + completions);
  return completions;
}

/**
 * A function that performs an all-gather across ranks.
 *
 * Args:
 *   src: Pointer to this rank's data of size n_bytes.
 *   dst: Pointer to an output buffer of size size_ * n_bytes. After the call,
 *        dst[r * n_bytes, (r+1) * n_bytes] contains the data from rank r.
 *   n_bytes: The number of bytes contributed by each rank.
 */
using AllGatherFn =
    std::function<void(const char* src, char* dst, size_t n_bytes)>;

// ── liveness + progress guard
// ───────────────────────────────────────────────── UC queue pairs give no
// error completion when a peer dies (measured on Apple's Thunderbolt RDMA
// stack, macOS 26.6.1, scripts/jaccl/rdma_pair.c): a survivor polls forever.
// The side channel TCP sockets live as long as the group, so a closed socket IS
// the death signal. ProgressGuard is ticked from every completion wait loop;
// while no completion arrives it checks the side channel at most every 250 ms
// (poll()/MSG_PEEK only: no reads, no locks, safe from several wire threads at
// once) and enforces a hard progress timeout (JACCL_PROGRESS_TIMEOUT_S /
// MLX_JACCL_PROGRESS_TIMEOUT_S, default 600 s, 0 = disabled) as a backstop for
// a lost UC frame or a wedged peer.
void check_peers_alive(std::span<const int> fds, int rank, const char* what);
double progress_timeout_s();

class ProgressGuard {
 public:
  ProgressGuard(std::span<const int> fds, int rank, const char* what);

  inline void tick(bool progressed) {
    if (progressed) {
      idle_polls_ = 0;
      progressed_ = true;
      return;
    }
    if ((++idle_polls_ & 255) != 0) {
      return;
    }
    slow_tick();
  }

 private:
  void slow_tick();

  std::span<const int> fds_;
  int rank_;
  const char* what_;
  std::chrono::steady_clock::time_point last_progress_;
  std::chrono::steady_clock::time_point last_check_;
  uint32_t idle_polls_ = 0;
  bool progressed_ = false;
};

class TCPAllGather {
 public:
  TCPAllGather(int rank, int size, const char* addr);

  TCPAllGather(const TCPAllGather&) = delete;
  TCPAllGather(TCPAllGather&&) = delete;
  TCPAllGather& operator=(const TCPAllGather&) = delete;
  TCPAllGather& operator=(TCPAllGather&&) = delete;

  void operator()(const char* src, char* dst, size_t n_bytes);

  // the socket fds (rank 0: one per peer in rank order; other
  // ranks: the coordinator). Values only, the sockets stay owned here.
  std::vector<int> fds() const {
    std::vector<int> r;
    r.reserve(sockets_.size());
    for (auto& s : sockets_) {
      r.push_back(static_cast<int>(s));
    }
    return r;
  }

 private:
  int rank_;
  int size_;
  std::vector<TCPSocket> sockets_;
  std::mutex mutex_;
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
  SideChannel(int rank, int size, AllGatherFn agf);
  // same, with the side channel socket fds to watch (kept as a
  // separate overload so the stock 3-argument symbol stays exported: ABI).
  SideChannel(
      int rank,
      int size,
      AllGatherFn agf,
      std::vector<int> liveness_fds);
  SideChannel(SideChannel&& sc);

  // side channel socket fds to watch for peer death (empty when
  // the side channel is not TCP based, e.g. a user supplied all-gather).
  const std::vector<int>& liveness_fds() const {
    return liveness_fds_;
  }

  SideChannel(const SideChannel&) = delete;
  SideChannel& operator=(const SideChannel&) = delete;

  template <typename T>
  std::vector<T> all_gather(const T& v) {
    std::vector<T> result(size_);

    // T is a container of stuff like std::vector or std::string
    if constexpr (is_container<T>::value) {
      using U = typename T::value_type;

      // Share the lengths first to set the communication size to be the
      // maximum length of the containers.
      auto lengths = all_gather<int>(v.size());
      auto max_len = *std::max_element(lengths.begin(), lengths.end());

      // Allocate flat memory for the all gather
      std::vector<U> buffer;
      buffer.resize(max_len * (size_ + 1));

      // Copy our value and share it
      std::copy(v.begin(), v.end(), buffer.begin());
      all_gather_fn_(
          reinterpret_cast<const char*>(&buffer[0]),
          reinterpret_cast<char*>(&buffer[max_len]),
          max_len * sizeof(U));

      // Put the values into the individual containers
      for (int i = 0; i < size_; i++) {
        std::copy(
            buffer.begin() + (i + 1) * max_len,
            buffer.begin() + (i + 1) * max_len + lengths[i],
            std::inserter(result[i], result[i].end()));
      }
    }

    // T is a scalar
    else {
      all_gather_fn_(
          reinterpret_cast<const char*>(&v),
          reinterpret_cast<char*>(result.data()),
          sizeof(T));
    }

    return result;
  }

  void barrier() {
    // Twice has proven to be more robust to initialization issues.
    all_gather<int>(0);
    all_gather<int>(0);
  }

 private:
  int rank_;
  int size_;
  AllGatherFn all_gather_fn_;
  std::vector<int> liveness_fds_;
};

} // namespace jaccl
