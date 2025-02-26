// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
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

#include "mlx/backend/cpu/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
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
  std::future<void> send(T* buffer, size_t size) {
    return send_impl(reinterpret_cast<char*>(buffer), size * sizeof(T));
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

  std::future<void> send_impl(char* buffer, size_t size) {
    std::promise<void> send_completed_promise;
    auto send_completed_future = send_completed_promise.get_future();
    if (size == 0) {
      send_completed_promise.set_value();
      return send_completed_future;
    }

    {
      std::unique_lock lock(queue_mutex_);
      sends_.emplace_back(
          SocketTask(buffer, size, std::move(send_completed_promise)));
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
        } else if (errno != EAGAIN) {
          log_info(
              true, "Receiving from socket", fd_, "failed with errno", errno);
          return;
        }
      }
      if (!sends_.empty()) {
        auto& task = sends_.front();
        ssize_t r = ::send(fd_, task.buffer, task.size, 0);
        if (r > 0) {
          task.buffer = static_cast<char*>(task.buffer) + r;
          task.size -= r;
          delete_send = task.size == 0;
        } else if (errno != EAGAIN) {
          log_info(true, "Sending to socket", fd_, "failed with errno", errno);
          return;
        }
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

struct address_t {
  sockaddr_storage addr;
  socklen_t len;

  const sockaddr* get() const {
    return (struct sockaddr*)&addr;
  }
};

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port) {
  struct addrinfo hints, *res;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  if (status != 0) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

  address_t result;
  memcpy(&result.addr, res->ai_addr, res->ai_addrlen);
  result.len = res->ai_addrlen;
  freeaddrinfo(res);

  return result;
}

/**
 * Parse a sockaddr provided as an <ip>:<port> string.
 */
address_t parse_address(const std::string& ip_port) {
  auto colon = ip_port.find(":");
  if (colon == std::string::npos) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip_port;
    throw std::runtime_error(msg.str());
  }
  std::string ip(ip_port.begin(), ip_port.begin() + colon);
  std::string port(ip_port.begin() + colon + 1, ip_port.end());

  return parse_address(ip, port);
}

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
std::vector<std::vector<address_t>> load_nodes(const char* hostfile) {
  std::vector<std::vector<address_t>> nodes;
  std::ifstream f(hostfile);

  json hosts = json::parse(f);
  for (auto& h : hosts) {
    std::vector<address_t> host;
    for (auto& ips : h) {
      host.push_back(std::move(parse_address(ips.get<std::string>())));
    }
    nodes.push_back(std::move(host));
  }

  return nodes;
}

/**
 * Create a socket and accept one connection for each of the provided
 * addresses.
 */
std::vector<int> accept_connections(const std::vector<address_t>& addresses) {
  std::vector<int> sockets;
  int success;

  for (auto& address : addresses) {
    // Create the socket to wait for connections from the peers
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::ostringstream msg;
      msg << "[ring] Couldn't create socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Make sure we can launch immediately after shutdown by setting the
    // reuseaddr option so that we don't get address already in use errors
    int enable = 1;
    success = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
    if (success < 0) {
      shutdown(sock, 2);
      close(sock);
      std::ostringstream msg;
      msg << "[ring] Couldn't enable reuseaddr (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
    success = setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));
    if (success < 0) {
      shutdown(sock, 2);
      close(sock);
      std::ostringstream msg;
      msg << "[ring] Couldn't enable reuseport (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Bind the socket to the address and port
    success = bind(sock, address.get(), address.len);
    if (success < 0) {
      shutdown(sock, 2);
      close(sock);
      std::ostringstream msg;
      msg << "[ring] Couldn't bind socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Wait for connections
    success = listen(sock, 0);
    if (success < 0) {
      shutdown(sock, 2);
      close(sock);
      std::ostringstream msg;
      msg << "[ring] Couldn't listen (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    int peer_socket = accept(sock, nullptr, nullptr);
    if (peer_socket < 0) {
      shutdown(sock, 2);
      close(sock);
      std::ostringstream msg;
      msg << "[ring] Accept failed (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Close the listening socket
    shutdown(sock, 2);
    close(sock);

    sockets.push_back(peer_socket);
  }

  return sockets;
}

/**
 * The counterpoint of `accept_connections`. Basically connect to each of the
 * provided addresses.
 */
std::vector<int> make_connections(
    const std::vector<address_t>& addresses,
    bool verbose) {
  std::vector<int> sockets;
  int success;

  for (auto& address : addresses) {
    int sock;

    // Attempt to connect to the peer CONN_ATTEMPTS times with exponential
    // backoff. TODO: Do we need that?
    for (int attempt = 0; attempt < CONN_ATTEMPTS; attempt++) {
      // Create the socket
      sock = socket(AF_INET, SOCK_STREAM, 0);
      if (sock < 0) {
        std::ostringstream msg;
        msg << "[ring] Couldn't create socket (error: " << errno << ")";
        throw std::runtime_error(msg.str());
      }

      if (attempt > 0) {
        int wait = (1 << (attempt - 1)) * CONN_WAIT;
        log_info(
            verbose,
            "Attempt",
            attempt,
            "wait",
            wait,
            "ms (error:",
            errno,
            ")");
        std::this_thread::sleep_for(std::chrono::milliseconds(wait));
      }

      success = connect(sock, address.get(), address.len);
      if (success == 0) {
        break;
      }
    }
    if (success < 0) {
      std::ostringstream msg;
      msg << "[ring] Couldn't connect (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    sockets.push_back(sock);
  }

  return sockets;
}

array ensure_row_contiguous(const array& arr) {
  if (arr.flags().row_contiguous) {
    return arr;
  } else {
    array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
    copy(arr, arr_copy, CopyType::General);
    return arr_copy;
  }
}

template <typename T>
void sum_inplace(const T* input, T* output, size_t N) {
  while (N-- > 0) {
    *output += *input;
    input++;
    output++;
  }
}

} // namespace

class RingGroup : public GroupImpl {
 public:
  RingGroup(int rank, std::vector<std::vector<address_t>> nodes, bool verbose)
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
      sockets_left_ = std::move(accept_connections(nodes[rank_]));
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      sockets_right_ = std::move(make_connections(nodes[connect_to], verbose));
    } else {
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      sockets_right_ = std::move(make_connections(nodes[connect_to], verbose));
      log_info(verbose_, "Rank", rank_, "accepting");
      sockets_left_ = std::move(accept_connections(nodes[rank_]));
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

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input_, array& output) override {
    SWITCH_TYPE(output, all_sum<T>(input_, output));
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[ring] Group split not supported.");
  }

  void all_gather(const array& input, array& output) override {
    throw std::runtime_error("[ring] All gather not supported.");
  }

  void send(const array& input_, int dst) override {
    // Make sure that the input is row contiguous
    array input = ensure_row_contiguous(input_);

    int right = (rank_ + 1) % size_;
    int left = (rank_ + size_ - 1) % size_;
    if (dst == right) {
      send(sockets_right_, input.data<char>(), input.nbytes());
    } else if (dst == left) {
      send(sockets_left_, input.data<char>(), input.nbytes());
    } else {
      std::ostringstream msg;
      msg << "[ring] Send only supported to direct neighbors "
          << "but tried to send to " << dst << " from " << rank_ << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  void recv(array& out, int src) override {
    // NOTE: We 'll check the sockets with the opposite order of send so that
    //       they work even with 2 nodes where left and right is the same
    //       neighbor.
    int right = (rank_ + 1) % size_;
    int left = (rank_ + size_ - 1) % size_;
    if (src == left) {
      recv(sockets_left_, out.data<char>(), out.nbytes());
    } else if (src == right) {
      recv(sockets_right_, out.data<char>(), out.nbytes());
    } else {
      std::ostringstream msg;
      msg << "[ring] Recv only supported from direct neighbors "
          << "but tried to recv from " << src << " to " << rank_ << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

 private:
  template <typename T>
  void all_sum(const array& input_, array& output) {
    // Make sure that the input is row contiguous
    array input = ensure_row_contiguous(input_);

    // If the input data cannot be split into size_ segments then copy it and
    // all reduce a local buffer prefilled with 0s.
    if (input.size() < size_) {
      // TODO: Maybe allocate dynamically so we don't have the constraint
      // below?
      if (input.itemsize() * size_ > 1024) {
        std::ostringstream msg;
        msg << "Can't perform the ring all reduce of " << output.size()
            << " elements with a ring of size " << size_;
        throw std::runtime_error(msg.str());
      }

      char buffer[1024];
      std::memset(buffer, 0, size_ * input.itemsize());
      std::memcpy(buffer, input.data<char>(), input.nbytes());
      all_sum_impl<T>(
          reinterpret_cast<T*>(buffers_.data()),
          reinterpret_cast<T*>(buffer),
          size_,
          sockets_right_[0],
          sockets_left_[0],
          -1);
      std::memcpy(output.data<char>(), buffer, output.nbytes());
      return;
    }

    // If not inplace all reduce then copy the input to the output first
    if (input.data<void>() != output.data<void>()) {
      std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
    }

    // Split the all reduces so that each member has at least 1 buffer to
    // send/recv per segment.
    constexpr size_t min_send_size = 262144;
    size_t n_reduces = std::max(
        std::min(
            sockets_right_.size() + sockets_left_.size(),
            output.nbytes() / (size_ * min_send_size)),
        1UL);
    size_t step = ceildiv(output.size(), n_reduces);
    std::vector<std::future<void>> all_sums;

    for (int i = 0; i < n_reduces; i++) {
      all_sums.emplace_back(pool_.enqueue(std::bind(
          &RingGroup::all_sum_impl<T>,
          this,
          reinterpret_cast<T*>(
              buffers_.data() + i * ALL_SUM_SIZE * ALL_SUM_BUFFERS),
          output.data<T>() + i * step,
          std::min(output.size(), (i + 1) * step) - i * step,
          sockets_right_[i / 2],
          sockets_left_[i / 2],
          (i % 2) ? -1 : 1)));
    }
    for (auto& f : all_sums) {
      f.wait();
    }
  }

  template <typename T>
  void all_sum_impl(
      T* buffer,
      T* data,
      size_t data_size,
      int socket_right,
      int socket_left,
      int direction) {
    // Choose which socket we send to and recv from
    int socket_send = (direction < 0) ? socket_right : socket_left;
    int socket_recv = (direction < 0) ? socket_left : socket_right;

    // We split the data into `size_` segments of size `segment_size` and each
    // of these in smaller segments of ALL_SUM_SIZE which we 'll call packets.
    size_t segment_size = ceildiv(data_size, size_);
    size_t BUFFER_SIZE =
        std::max(32768UL, std::min(ALL_SUM_SIZE / sizeof(T), segment_size / 2));
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
          sum_inplace<T>(
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

  void send(const std::vector<int>& sockets, char* data, size_t data_size) {
    size_t segment_size = std::max(1024UL, ceildiv(data_size, sockets.size()));
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
    size_t segment_size = std::max(1024UL, ceildiv(data_size, sockets.size()));
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
