// Copyright © 2025 Apple Inc.

#pragma once

#include <infiniband/verbs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
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
constexpr uint64_t STALL_CHECK_POLLS = 1024;
constexpr std::chrono::milliseconds STALL_QUERY_INTERVAL{100};
constexpr std::chrono::milliseconds STALL_WAKE_QUERY{2};

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
  int (*query_qp)(ibv_qp*, ibv_qp_attr*, int, ibv_qp_init_attr*);
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

// Shared by all connections of a group. The first failure wins.
struct FailureState {
  std::atomic<bool> failed{false};
  std::once_flag once;
  std::string reason;
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

  void share_failure(std::shared_ptr<FailureState> failure) {
    failure_ = std::move(failure);
  }

  void post_send(const SharedBuffer& buff, uint64_t work_request_id) {
    throw_if_failed();

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
      fail("Send failed with error code " + std::to_string(status));
    }
  }

  void post_recv(const SharedBuffer& buff, uint64_t work_request_id) {
    throw_if_failed();

    ibv_recv_wr work_request, *bad_work_request;

    auto entry = buff.to_scatter_gather_entry(protection_domain);
    work_request.wr_id = work_request_id;
    work_request.sg_list = &entry;
    work_request.num_sge = 1;
    work_request.next = nullptr;

    if (int status =
            ibv_post_recv(queue_pair, &work_request, &bad_work_request);
        status != 0) {
      fail("Recv failed with error code " + std::to_string(status));
    }
  }

  int poll(int num_completions, ibv_wc* work_completions) const {
    int n = ibv_poll_cq(completion_queue, num_completions, work_completions);
    if (n > 0) {
      check_completions({work_completions, static_cast<size_t>(n)});
      if (empty_polls_ >= STALL_CHECK_POLLS) {
        check_stall_end();
      }
      empty_polls_ = 0;
    } else if (n < 0) {
      fail(
          "Polling the completion queue failed with error code " +
          std::to_string(n));
    } else if ((++empty_polls_ % STALL_CHECK_POLLS) == 0) {
      check_stalled();
    }
    return n;
  }

 private:
  [[noreturn]] void fail(const std::string& reason) const {
    std::call_once(failure_->once, [this, &reason] {
      failure_->reason = "[jaccl] " + reason;
      failure_->failed.store(true, std::memory_order_release);
    });
    throw std::runtime_error(failure_->reason);
  }

  void throw_if_failed() const {
    if (failure_->failed.load(std::memory_order_acquire)) {
      throw std::runtime_error(failure_->reason);
    }
  }

  static constexpr int severity(const ibv_wc& wc) {
    switch (wc.status) {
      case IBV_WC_SUCCESS:
        return 2;
      case IBV_WC_WR_FLUSH_ERR:
        return 1;
      default:
        return 0;
    }
  }

  void check_completions(std::span<const ibv_wc> wcs) const {
    const auto& worst = *std::ranges::min_element(wcs, {}, severity);
    if (worst.status == IBV_WC_SUCCESS) {
      return;
    }
    std::ostringstream msg;
    msg << "Work completion failed with status "
        << static_cast<int>(worst.status) << " (qp " << worst.qp_num
        << ", wr_id " << worst.wr_id << ", opcode "
        << static_cast<int>(worst.opcode) << ")";
    fail(msg.str());
  }

  void check_stalled() const;
  void check_stall_end() const;
  void check_queue_pair() const;

  std::shared_ptr<FailureState> failure_ = std::make_shared<FailureState>();
  mutable uint64_t empty_polls_{0};
  mutable std::chrono::steady_clock::time_point stall_start_{};
  mutable std::chrono::steady_clock::time_point last_query_{};
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

    int n =
        c.poll(num_completions - completions, work_completions + completions);

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

class TCPAllGather {
 public:
  TCPAllGather(int rank, int size, const char* addr);

  TCPAllGather(const TCPAllGather&) = delete;
  TCPAllGather(TCPAllGather&&) = delete;
  TCPAllGather& operator=(const TCPAllGather&) = delete;
  TCPAllGather& operator=(TCPAllGather&&) = delete;

  void operator()(const char* src, char* dst, size_t n_bytes);

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
  SideChannel(SideChannel&& sc);

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
};

} // namespace jaccl
