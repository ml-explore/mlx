// Copyright © 2024 Apple Inc.

#include <fcntl.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <list>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <json.hpp>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/reduction_ops.h"
#include "mlx/distributed/utils.h"
#include "mlx/threadpool.h"

#ifndef SOL_TCP
#define SOL_TCP IPPROTO_TCP
#endif

#define SWITCH_TYPE(x, ...)  \
  switch ((x).dtype()) {     \
    case bool_: {            \
      using T = bool;        \
      __VA_ARGS__;           \
    } break;                 \
    case int8: {             \
      using T = int8_t;      \
      __VA_ARGS__;           \
    } break;                 \
    case int16: {            \
      using T = int16_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case int32: {            \
      using T = int32_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case int64: {            \
      using T = int64_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case uint8: {            \
      using T = uint8_t;     \
      __VA_ARGS__;           \
    } break;                 \
    case uint16: {           \
      using T = uint16_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case uint32: {           \
      using T = uint32_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case uint64: {           \
      using T = uint64_t;    \
      __VA_ARGS__;           \
    } break;                 \
    case bfloat16: {         \
      using T = bfloat16_t;  \
      __VA_ARGS__;           \
    } break;                 \
    case float16: {          \
      using T = float16_t;   \
      __VA_ARGS__;           \
    } break;                 \
    case float32: {          \
      using T = float;       \
      __VA_ARGS__;           \
    } break;                 \
    case float64: {          \
      using T = double;      \
      __VA_ARGS__;           \
    } break;                 \
    case complex64: {        \
      using T = complex64_t; \
      __VA_ARGS__;           \
    } break;                 \
  }

namespace mlx::core::distributed::ring {

constexpr const size_t ALL_SUM_SIZE = 8 * 1024 * 1024;
constexpr const size_t ALL_SUM_BUFFERS = 2;
constexpr const int CONN_ATTEMPTS = 5;
constexpr const int CONN_WAIT = 1000;
constexpr const char* RING_TAG = "[ring]";

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

template <typename T>
void log(std::ostream& os, T first) {
  os << first << std::endl;
}

template <typename T, typename... Args>
void log(std::ostream& os, T first, Args... args) {
  log(os << first << " ", args...);
}

template <typename... Args>
void log_info(bool verbose, Args... args) {
  if (!verbose) {
    return;
  }

  log(std::cerr, "[ring]", args...);
}

template <typename T, typename U>
decltype(T() * U()) ceildiv(T a, U b) {
  return (a + b - 1) / b;
}

class SocketThread {
 public:
  SocketThread(int fd) : fd_(fd), stop_(false) {
    worker_ = std::thread(&SocketThread::worker, this);
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
  ~SocketThread() {
    stop_ = true;
    condition_.notify_all();
    worker_.join();
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);
  }

  template <typename T>
  std::future<void> send(const T* buffer, size_t size) {
    return send_impl(reinterpret_cast<const char*>(buffer), size * sizeof(T));
  }

  template <typename T>
  std::future<void> recv(T* buffer, size_t size) {
    return recv_impl(reinterpret_cast<char*>(buffer), size * sizeof(T));
  }

 private:
  struct SocketTask {
    SocketTask(void* b, size_t s, std::promise<void>&& p)
        : buffer(b), size(s), promise(std::move(p)) {}
    SocketTask(SocketTask&& t)
        : buffer(t.buffer), size(t.size), promise(std::move(t.promise)) {}
    void* buffer;
    size_t size;
    std::promise<void> promise;
  };

  std::future<void> send_impl(const char* buffer, size_t size) {
    std::promise<void> send_completed_promise;
    auto send_completed_future = send_completed_promise.get_future();
    if (size == 0) {
      send_completed_promise.set_value();
      return send_completed_future;
    }

    {
      std::unique_lock lock(queue_mutex_);
      sends_.emplace_back(SocketTask(
          const_cast<char*>(buffer), size, std::move(send_completed_promise)));
    }
    condition_.notify_one();
    return send_completed_future;
  }

  std::future<void> recv_impl(char* buffer, size_t size) {
    std::promise<void> recv_completed_promise;
    auto recv_completed_future = recv_completed_promise.get_future();
    if (size == 0) {
      recv_completed_promise.set_value();
      return recv_completed_future;
    }

    {
      std::unique_lock lock(queue_mutex_);
      recvs_.emplace_back(
          SocketTask(buffer, size, std::move(recv_completed_promise)));
    }
    condition_.notify_one();
    return recv_completed_future;
  }

  bool have_tasks() {
    return !(sends_.empty() && recvs_.empty());
  }

  void worker() {
    int error_count = 0;
    bool delete_recv = false;
    bool delete_send = false;
    while (true) {
      {
        std::unique_lock lock(queue_mutex_);

        if (delete_recv) {
          recvs_.front().promise.set_value();
          recvs_.pop_front();
          delete_recv = false;
        }
        if (delete_send) {
          sends_.front().promise.set_value();
          sends_.pop_front();
          delete_send = false;
        }

        if (stop_) {
          return;
        }

        if (!have_tasks()) {
          condition_.wait(lock, [this] { return stop_ || have_tasks(); });
          if (stop_) {
            return;
          }
        }
      }

      if (!recvs_.empty()) {
        auto& task = recvs_.front();
        ssize_t r = ::recv(fd_, task.buffer, task.size, 0);
        if (r > 0) {
          task.buffer = static_cast<char*>(task.buffer) + r;
          task.size -= r;
          delete_recv = task.size == 0;
          error_count = 0;
        } else if (errno != EAGAIN) {
          error_count++;
          log_info(
              true, "Receiving from socket", fd_, "failed with errno", errno);
        }
      }
      if (!sends_.empty()) {
        auto& task = sends_.front();
        ssize_t r = ::send(fd_, task.buffer, task.size, 0);
        if (r > 0) {
          task.buffer = static_cast<char*>(task.buffer) + r;
          task.size -= r;
          delete_send = task.size == 0;
          error_count = 0;
        } else if (errno != EAGAIN) {
          error_count++;
          log_info(true, "Sending to socket", fd_, "failed with errno", errno);
        }
      }

      if (error_count >= 10) {
        log_info(true, "Too many send/recv errors. Aborting...");
        return;
      }
    }
  }

  int fd_;
  bool stop_;
  std::thread worker_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::list<SocketTask> sends_;
  std::list<SocketTask> recvs_;
};

class CommunicationThreads {
 public:
  void add(const std::vector<int>& sockets) {
    for (int sock : sockets) {
      threads_.emplace(sock, sock);
    }
  }

  template <typename T>
  std::future<void> send(int socket, T* buffer, size_t size) {
    return threads_.at(socket).send<T>(buffer, size);
  }

  template <typename T>
  std::future<void> recv(int socket, T* buffer, size_t size) {
    return threads_.at(socket).recv<T>(buffer, size);
  }

 private:
  std::unordered_map<int, SocketThread> threads_;
};

/**
 * Load all addresses from the json hostfile. The hostfile is a list of
 * addresses in order of rank. For each rank there can be many addresses so
 * that we can have multiple connections between peers.
 *
 * For example:
 *  [
 *    ["ip1:5000", "ip1:5001"],
 *    ["ip2:5000", "ip2:5001"],
 *    ["ip3:5000", "ip3:5001"],
 *  ]
 */
std::vector<std::vector<detail::address_t>> load_nodes(const char* hostfile) {
  std::vector<std::vector<detail::address_t>> nodes;
  std::ifstream f(hostfile);

  json hosts = json::parse(f);
  for (auto& h : hosts) {
    std::vector<detail::address_t> host;
    for (auto& ips : h) {
      host.push_back(std::move(detail::parse_address(ips.get<std::string>())));
    }
    nodes.push_back(std::move(host));
  }

  return nodes;
}

/**
 * Create a socket and accept one connection for each of the provided
 * addresses.
 */
std::vector<int> accept_connections(
    const std::vector<detail::address_t>& addresses) {
  std::vector<int> sockets;
  int success;

  for (auto& address : addresses) {
    detail::TCPSocket socket(RING_TAG);
    socket.listen(RING_TAG, address);
    sockets.push_back(socket.accept(RING_TAG).detach());
  }

  return sockets;
}

/**
 * The counterpoint of `accept_connections`. Basically connect to each of the
 * provided addresses.
 */
std::vector<int> make_connections(
    const std::vector<detail::address_t>& addresses,
    bool verbose) {
  std::vector<int> sockets;
  int success;

  for (auto& address : addresses) {
    sockets.push_back(detail::TCPSocket::connect(
                          RING_TAG,
                          address,
                          CONN_ATTEMPTS,
                          CONN_WAIT,
                          [verbose](int attempt, int wait) {
                            log_info(
                                verbose,
                                "Attempt",
                                attempt,
                                "waiting",
                                wait,
                                "ms (error:",
                                errno,
                                ")");
                          })
                          .detach());
  }

  return sockets;
}

} // namespace

class RingGroup : public GroupImpl {
 public:
  RingGroup(
      int rank,
      std::vector<std::vector<detail::address_t>> nodes,
      bool verbose)
      : rank_(rank), verbose_(verbose), pool_(0) {
    if (rank_ > 0 && rank_ >= nodes.size()) {
      throw std::runtime_error(
          "[ring] Rank cannot be larger than the size of the group");
    }

    size_ = nodes.size();
    int connect_to = (rank_ + 1) % size_;

    // We define the connection order by having the rank_ == size_ - 1 connect
    // first and accept after.
    if (rank_ < connect_to) {
      log_info(verbose_, "Rank", rank_, "accepting");
      sockets_left_ = accept_connections(nodes[rank_]);
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      sockets_right_ = make_connections(nodes[connect_to], verbose);
    } else {
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      sockets_right_ = make_connections(nodes[connect_to], verbose);
      log_info(verbose_, "Rank", rank_, "accepting");
      sockets_left_ = accept_connections(nodes[rank_]);
    }

    // Failure if we couldn't make right or left sockets
    if (sockets_right_.empty()) {
      std::ostringstream msg;
      msg << "[ring] Rank " << rank_ << " has no sockets to the right.";
      throw std::invalid_argument(msg.str());
    }
    if (sockets_left_.empty()) {
      std::ostringstream msg;
      msg << "[ring] Rank " << rank_ << " has no sockets to the left.";
      throw std::invalid_argument(msg.str());
    }

    // The following could be relaxed since we can define non-homogeneous rings
    // but it makes things a bit simpler for now.
    if (sockets_right_.size() != sockets_left_.size()) {
      std::ostringstream msg;
      msg << "[ring] It is required to have as many connections to the left as "
          << "to the right but rank " << rank_ << " has "
          << sockets_right_.size() << " connections to the right and "
          << sockets_left_.size() << " to the left.";
      throw std::invalid_argument(msg.str());
    }

    // Configure all sockets to use TCP no delay.
    int one = 1;
    for (int i = 0; i < sockets_right_.size(); i++) {
      setsockopt(sockets_right_[i], SOL_TCP, TCP_NODELAY, &one, sizeof(one));
      setsockopt(sockets_left_[i], SOL_TCP, TCP_NODELAY, &one, sizeof(one));
    }

    // Start the all reduce threads. One all reduce per direction per ring.
    pool_.resize(sockets_right_.size() + sockets_left_.size());

    // Create a communication thread per socket. This also converts them to
    // non-blocking.
    comm_.add(sockets_right_);
    comm_.add(sockets_left_);

    // Allocate buffers for the all sum
    buffers_.resize(
        (sockets_right_.size() + sockets_left_.size()) * ALL_SUM_BUFFERS *
        ALL_SUM_SIZE);
  }

  ~RingGroup() {
    for (auto s : sockets_right_) {
      shutdown(s, 2);
      close(s);
    }
    for (auto s : sockets_left_) {
      shutdown(s, 2);
      close(s);
    }
  }

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::cpu);
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    SWITCH_TYPE(
        output, all_reduce<T>(input, output, stream, detail::SumOp<T>()));
  }

  void all_max(const array& input, array& output, Stream stream) override {
    SWITCH_TYPE(
        output, all_reduce<T>(input, output, stream, detail::MaxOp<T>()));
  }

  void all_min(const array& input, array& output, Stream stream) override {
    SWITCH_TYPE(
        output, all_reduce<T>(input, output, stream, detail::MinOp<T>()));
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[ring] Group split not supported.");
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.set_output_array(output);
    encoder.dispatch([input_ptr = input.data<char>(),
                      nbytes = input.nbytes(),
                      output_ptr = output.data<char>(),
                      this]() {
      constexpr size_t min_send_size = 262144;
      size_t n_gathers = std::max(
          std::min(
              sockets_right_.size() + sockets_left_.size(),
              nbytes / min_send_size),
          size_t(1));
      size_t bytes_per_gather = ceildiv(nbytes, n_gathers);
      std::vector<std::future<void>> all_gathers;
      for (int i = 0; i < n_gathers; i++) {
        auto offset = i * bytes_per_gather;
        all_gathers.emplace_back(pool_.enqueue(std::bind(
            &RingGroup::all_gather_impl,
            this,
            input_ptr + offset,
            output_ptr + offset,
            nbytes,
            offset + bytes_per_gather > nbytes ? nbytes - offset
                                               : bytes_per_gather,
            sockets_right_[i / 2],
            sockets_left_[i / 2],
            (i % 2) ? -1 : 1)));
      }
      for (auto& f : all_gathers) {
        f.wait();
      }
    });
  }

  void send(const array& input, int dst, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_input_array(input);
    encoder.dispatch(
        [input_ptr = input.data<char>(), nbytes = input.nbytes(), dst, this]() {
          int right = (rank_ + 1) % size_;
          int left = (rank_ + size_ - 1) % size_;
          if (dst == right) {
            send(sockets_right_, input_ptr, nbytes);
          } else if (dst == left) {
            send(sockets_left_, input_ptr, nbytes);
          } else {
            std::ostringstream msg;
            msg << "[ring] Send only supported to direct neighbors "
                << "but tried to send to " << dst << " from " << rank_
                << std::endl;
            throw std::runtime_error(msg.str());
          }
        });
  }

  void recv(array& out, int src, Stream stream) override {
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_output_array(out);
    encoder.dispatch(
        [out_ptr = out.data<char>(), nbytes = out.nbytes(), src, this]() {
          // NOTE: We 'll check the sockets with the opposite order of send so
          // that they work even with 2 nodes where left and right is the same
          // neighbor.
          int right = (rank_ + 1) % size_;
          int left = (rank_ + size_ - 1) % size_;
          if (src == left) {
            recv(sockets_left_, out_ptr, nbytes);
          } else if (src == right) {
            recv(sockets_right_, out_ptr, nbytes);
          } else {
            std::ostringstream msg;
            msg << "[ring] Recv only supported from direct neighbors "
                << "but tried to recv from " << src << " to " << rank_
                << std::endl;
            throw std::runtime_error(msg.str());
          }
        });
  }

  void sum_scatter(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[ring] sum_scatter not supported.");
  }

 private:
  template <typename T, typename ReduceOp>
  void all_reduce(
      const array& input,
      array& output,
      Stream stream,
      ReduceOp reduce_op) {
    auto in_ptr = input.data<char>();
    auto out_ptr = output.data<char>();
    auto& encoder = cpu::get_command_encoder(stream);
    encoder.set_output_array(output);
    encoder.dispatch([in_ptr, out_ptr, size = input.size(), this, reduce_op]() {
      // If the input data cannot be split into size_ segments then copy it and
      // all reduce a local buffer prefilled with 0s.
      size_t nbytes = size * sizeof(T);
      if (size < size_) {
        // TODO: Maybe allocate dynamically so we don't have the constraint
        // below?
        if (sizeof(T) * size_ > 1024) {
          std::ostringstream msg;
          msg << "Can't perform the ring all reduce of " << size
              << " elements with a ring of size " << size_;
          throw std::runtime_error(msg.str());
        }

        char buffer[1024];
        std::memset(buffer, 0, size_ * sizeof(T));
        std::memcpy(buffer, in_ptr, nbytes);
        all_reduce_impl<T, ReduceOp>(
            reinterpret_cast<T*>(buffers_.data()),
            reinterpret_cast<T*>(buffer),
            size_,
            sockets_right_[0],
            sockets_left_[0],
            -1,
            reduce_op);
        std::memcpy(out_ptr, buffer, nbytes);
        return;
      }

      // If not inplace all reduce then copy the input to the output first
      if (in_ptr != out_ptr) {
        std::memcpy(out_ptr, in_ptr, nbytes);
      }

      // Split the all reduces so that each member has at least 1 buffer to
      // send/recv per segment.
      constexpr size_t min_send_size = 262144;
      size_t n_reduces = std::max(
          std::min(
              sockets_right_.size() + sockets_left_.size(),
              nbytes / (size_ * min_send_size)),
          size_t(1));
      size_t step = ceildiv(size, n_reduces);
      std::vector<std::future<void>> all_sums;

      for (int i = 0; i < n_reduces; i++) {
        all_sums.emplace_back(pool_.enqueue(std::bind(
            &RingGroup::all_reduce_impl<T, ReduceOp>,
            this,
            reinterpret_cast<T*>(
                buffers_.data() + i * ALL_SUM_SIZE * ALL_SUM_BUFFERS),
            reinterpret_cast<T*>(out_ptr) + i * step,
            std::min(size, (i + 1) * step) - i * step,
            sockets_right_[i / 2],
            sockets_left_[i / 2],
            (i % 2) ? -1 : 1,
            reduce_op)));
      }
      for (auto& f : all_sums) {
        f.wait();
      }
    });
  }

  template <typename T, typename ReduceOp>
  void all_reduce_impl(
      T* buffer,
      T* data,
      size_t data_size,
      int socket_right,
      int socket_left,
      int direction,
      ReduceOp reduce_op) {
    // Choose which socket we send to and recv from
    int socket_send = (direction < 0) ? socket_right : socket_left;
    int socket_recv = (direction < 0) ? socket_left : socket_right;

    // We split the data into `size_` segments of size `segment_size` and each
    // of these in smaller segments of ALL_SUM_SIZE which we 'll call packets.
    size_t segment_size = ceildiv(data_size, size_);
    size_t BUFFER_SIZE = std::max(
        size_t(32768), std::min(ALL_SUM_SIZE / sizeof(T), segment_size / 2));
    size_t n_packets = ceildiv(segment_size, BUFFER_SIZE);

    // Initial segments
    int send_segment = rank_;
    int recv_segment = (rank_ + direction + size_) % size_;

    // Plan the whole reduce in terms of sends and recvs as indices in data.
    // It makes the actual async send and recv a bit simpler to follow when
    // there are less offset calculations around.
    std::vector<std::pair<size_t, size_t>> send_plan;
    std::vector<std::pair<size_t, size_t>> recv_plan;

    // Two times the same send/recv operations, first scatter reduce and then
    // gather.
    for (int k = 0; k < 2; k++) {
      for (int i = 0; i < size_ - 1; i++) {
        size_t send_start = send_segment * segment_size;
        size_t send_stop =
            std::min((send_segment + 1) * segment_size, data_size);
        size_t recv_start = recv_segment * segment_size;
        size_t recv_stop =
            std::min((recv_segment + 1) * segment_size, data_size);

        for (size_t j = 0; j < n_packets; j++) {
          send_plan.emplace_back(
              std::min(send_start + j * BUFFER_SIZE, send_stop),
              std::min(send_start + (j + 1) * BUFFER_SIZE, send_stop));
          recv_plan.emplace_back(
              std::min(recv_start + j * BUFFER_SIZE, recv_stop),
              std::min(recv_start + (j + 1) * BUFFER_SIZE, recv_stop));
        }

        send_segment = (send_segment + size_ + direction) % size_;
        recv_segment = (recv_segment + size_ + direction) % size_;
      }
    }

    // Running the plan is fairly simple, we keep a send and a recv in flight
    // while doing the summation.
    T* recv_buffers[ALL_SUM_BUFFERS];
    for (int i = 0; i < ALL_SUM_BUFFERS; i++) {
      recv_buffers[i] = buffer + i * BUFFER_SIZE;
    }
    std::future<void> sends[2], recvs[2];
    int a = 0;
    int b = (n_packets > 1) ? 1 : 0;
    for (int i = 0, j = -b; i < send_plan.size(); j++, i++) {
      sends[a] = comm_.send(
          socket_send,
          data + send_plan[i].first,
          send_plan[i].second - send_plan[i].first);
      if (2 * i < send_plan.size()) {
        recvs[a] = comm_.recv(
            socket_recv,
            recv_buffers[i % ALL_SUM_BUFFERS],
            recv_plan[i].second - recv_plan[i].first);
      } else {
        recvs[a] = comm_.recv(
            socket_recv,
            data + recv_plan[i].first,
            recv_plan[i].second - recv_plan[i].first);
      }

      if (j >= 0) {
        sends[b].wait();
        recvs[b].wait();
        if (2 * j < send_plan.size()) {
          reduce_op(
              recv_buffers[j % ALL_SUM_BUFFERS],
              data + recv_plan[j].first,
              recv_plan[j].second - recv_plan[j].first);
        }
      }

      std::swap(a, b);
    }
    sends[b].wait();
    recvs[b].wait();
  }

  void all_gather_impl(
      const char* input,
      char* output,
      size_t input_size,
      size_t data_size,
      int socket_right,
      int socket_left,
      int direction) {
    // Choose which socket we send to and recv from
    int socket_send = (direction < 0) ? socket_right : socket_left;
    int socket_recv = (direction < 0) ? socket_left : socket_right;

    // Initial segments
    int send_segment = rank_;
    int recv_segment = (rank_ + direction + size_) % size_;

    // Copy our own segment in the output
    std::memcpy(output + rank_ * input_size, input, data_size);

    // Simple send/recv all gather. Possible performance improvement by
    // splitting to multiple chunks and allowing send/recv to run a bit ahead.
    // See all_sum_impl for an example.
    for (int i = 0; i < size_ - 1; i++) {
      auto sent = comm_.send(
          socket_send, output + send_segment * input_size, data_size);
      auto recvd = comm_.recv(
          socket_recv, output + recv_segment * input_size, data_size);

      send_segment = (send_segment + size_ + direction) % size_;
      recv_segment = (recv_segment + size_ + direction) % size_;

      sent.wait();
      recvd.wait();
    }
  }

  void
  send(const std::vector<int>& sockets, const char* data, size_t data_size) {
    size_t segment_size =
        std::max(size_t(1024), ceildiv(data_size, sockets.size()));
    std::vector<std::future<void>> sends;
    for (int i = 0; i < sockets.size(); i++) {
      if (i * segment_size >= data_size) {
        break;
      }
      sends.emplace_back(comm_.send(
          sockets[i],
          data + i * segment_size,
          std::min(data_size, (i + 1) * segment_size) - i * segment_size));
    }
    for (auto& f : sends) {
      f.wait();
    }
  }

  void recv(const std::vector<int>& sockets, char* data, size_t data_size) {
    size_t segment_size =
        std::max(size_t(1024), ceildiv(data_size, sockets.size()));
    std::vector<std::future<void>> recvs;
    for (int i = 0; i < sockets.size(); i++) {
      if (i * segment_size >= data_size) {
        break;
      }
      recvs.emplace_back(comm_.recv(
          sockets[i],
          data + i * segment_size,
          std::min(data_size, (i + 1) * segment_size) - i * segment_size));
    }
    for (auto& f : recvs) {
      f.wait();
    }
  }

  int rank_;
  int size_;

  bool verbose_;

  ThreadPool pool_;
  CommunicationThreads comm_;

  std::vector<int> sockets_right_;
  std::vector<int> sockets_left_;

  std::vector<char> buffers_;
};

bool is_available() {
  return true;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  const char* hostfile = std::getenv("MLX_HOSTFILE");
  const char* rank_str = std::getenv("MLX_RANK");
  const char* ring_verbose = std::getenv("MLX_RING_VERBOSE");

  if (!hostfile || !rank_str) {
    if (strict) {
      std::ostringstream msg;
      msg << "[ring] You need to provide via environment variables both a rank (MLX_RANK) "
          << "and a hostfile (MLX_HOSTFILE) but provided MLX_RANK=\""
          << ((rank_str) ? rank_str : "") << "\" and MLX_HOSTFILE=\""
          << ((hostfile) ? hostfile : "") << "\"";
      throw std::runtime_error(msg.str());
    }
    return nullptr;
  }

  auto nodes = load_nodes(hostfile);
  int rank = std::atoi(rank_str);

  return std::make_shared<RingGroup>(rank, nodes, ring_verbose != nullptr);
}

} // namespace mlx::core::distributed::ring
