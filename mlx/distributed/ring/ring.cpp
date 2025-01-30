// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include "mlx/backend/common/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/io/json.h"
#include "mlx/threadpool.h"

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
    case complex64: {        \
      using T = complex64_t; \
      __VA_ARGS__;           \
    } break;                 \
  }

namespace mlx::core::distributed::ring {

constexpr const size_t PACKET_SIZE = 262144;
constexpr const int CONN_ATTEMPTS = 5;
constexpr const int CONN_WAIT = 1000;

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

namespace {

class Barrier {
 public:
  explicit Barrier(int n_threads)
      : n_threads_(n_threads), count_(0), flag_(false) {}

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);

    // Keep the flag that marks the current use of the barrier. The next use is
    // going to have this flag flipped.
    bool initial_flag = flag_;

    // Increment the count
    count_++;

    // We are the last thread to arrive so reset the count, change the flag and
    // notify everybody.
    if (count_ == n_threads_) {
      count_ = 0;
      flag_ = !flag_;
      cv_.notify_all();
    }

    // Wait for the rest to arrive
    else {
      cv_.wait(lock, [this, initial_flag]() { return initial_flag != flag_; });
    }
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  int n_threads_;

  int count_;
  bool flag_; // we need this for sequential use of the barrier
};

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

  io::json hosts = io::parse_json(f);
  for (auto& h : hosts) {
    std::vector<address_t> host;
    for (std::string ips : h) {
      host.push_back(std::move(parse_address(ips)));
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

template <typename T>
void _send(int sock, T* data, size_t start, size_t stop) {
  if (stop <= start) {
    return;
  }
  data += start;
  size_t len = (stop - start) * sizeof(T);
  const char* buffer = (const char*)data;
  while (len > 0) {
    ssize_t r = send(sock, buffer, len, 0);
    if (r <= 0) {
      std::ostringstream msg;
      msg << "Send of " << len << " bytes failed (errno: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
    buffer += r;
    len -= r;
  }
}

template <typename T>
void _recv(int sock, T* data, size_t start, size_t stop) {
  if (stop <= start) {
    return;
  }
  data += start;
  size_t len = (stop - start) * sizeof(T);
  char* buffer = (char*)data;
  while (len > 0) {
    ssize_t r = recv(sock, buffer, len, 0);
    if (r <= 0) {
      std::ostringstream msg;
      msg << "Recv of " << len << " bytes failed (errno: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
    buffer += r;
    len -= r;
  }
}

template <typename T>
void _recv_sum(int sock, T* data, size_t start, size_t stop) {
  if (stop <= start) {
    return;
  }
  data += start;
  char buffer[PACKET_SIZE];
  size_t len = (stop - start) * sizeof(T);
  while (len > 0) {
    ssize_t r = 0;
    do {
      ssize_t partial_r =
          recv(sock, buffer + r, std::min(len, PACKET_SIZE) - r, 0);
      if (partial_r <= 0) {
        std::ostringstream msg;
        msg << "Recv of " << len << " bytes failed (errno: " << errno << ")";
        throw std::runtime_error(msg.str());
      }
      r += partial_r;
    } while (r % sizeof(T));
    sum_inplace((const T*)buffer, data, r / sizeof(T));
    data += r / sizeof(T);
    len -= r;
  }
}

template <typename T>
void ring_send(
    Barrier& barrier,
    int socket,
    int rank,
    int size,
    T* data,
    size_t data_size,
    int direction = -1) {
  // We split the data into `size_` segments of size `segment_size`
  size_t segment_size = ceildiv(data_size, size);

  // Initial segment
  int segment = rank;

  // 1st send
  for (int i = 0; i < size - 1; i++) {
    size_t start = segment * segment_size;
    size_t stop = std::min((segment + 1) * segment_size, data_size);
    _send<T>(socket, data, start, stop);
    barrier.arrive_and_wait();
    segment = (segment + size + direction) % size;
  }

  // 2nd send
  for (int i = 0; i < size - 1; i++) {
    size_t start = segment * segment_size;
    size_t stop = std::min((segment + 1) * segment_size, data_size);
    _send<T>(socket, data, start, stop);
    barrier.arrive_and_wait();
    segment = (segment + size + direction) % size;
  }
}

template <typename T>
void ring_recv_sum(
    Barrier& barrier,
    int socket,
    int rank,
    int size,
    T* data,
    size_t data_size,
    int direction = -1) {
  // We split the data into `size_` segments of size `segment_size`
  size_t segment_size = ceildiv(data_size, size);

  // Initial segment
  int segment = (rank + size + direction) % size;

  // Recv sum
  for (int i = 0; i < size - 1; i++) {
    size_t start = segment * segment_size;
    size_t stop = std::min((segment + 1) * segment_size, data_size);
    _recv_sum<T>(socket, data, start, stop);
    barrier.arrive_and_wait();
    segment = (segment + size + direction) % size;
  }

  // Recv
  for (int i = 0; i < size - 1; i++) {
    size_t start = segment * segment_size;
    size_t stop = std::min((segment + 1) * segment_size, data_size);
    _recv<T>(socket, data, start, stop);
    barrier.arrive_and_wait();
    segment = (segment + size + direction) % size;
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
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      send_sockets_ = std::move(make_connections(nodes[connect_to], verbose));
    } else {
      log_info(verbose_, "Rank", rank_, "connecting to", connect_to);
      send_sockets_ = std::move(make_connections(nodes[connect_to], verbose));
      log_info(verbose_, "Rank", rank_, "accepting");
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
    }

    // Failure if we couldn't make send or recv sockets
    if (send_sockets_.empty()) {
      std::ostringstream msg;
      msg << "[ring] Rank " << rank_ << " has no send sockets.";
      throw std::invalid_argument(msg.str());
    }
    if (recv_sockets_.empty()) {
      std::ostringstream msg;
      msg << "[ring] Rank " << rank_ << " has no recv sockets.";
      throw std::invalid_argument(msg.str());
    }

    // The following could be relaxed since we can define non-homogeneous rings
    // but it makes things a bit simpler for now.
    if (send_sockets_.size() != recv_sockets_.size()) {
      std::ostringstream msg;
      msg << "[ring] It is required to have as many connections to the left as "
          << "to the right but rank " << rank_ << " has "
          << send_sockets_.size() << " connections to the right and "
          << recv_sockets_.size() << " to the left.";
      throw std::invalid_argument(msg.str());
    }

    // Start the necessary threads for completely parallel operation on all
    // channels. One thread to send, one to receive per socket.
    pool_.resize(send_sockets_.size() * 2 * 2);
  }

  ~RingGroup() {
    for (auto s : send_sockets_) {
      shutdown(s, 2);
      close(s);
    }
    for (auto s : recv_sockets_) {
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
  void send(const array& input, int dst) override {
    throw std::runtime_error("[ring] Send not supported.");
  }
  void recv(array& out, int src) override {
    throw std::runtime_error("[ring] Recv not supported.");
  }

 private:
  template <typename T>
  void all_sum(const array& input_, array& output) {
    // Make sure that the input is row contiguous
    array input = ensure_row_contiguous(input_);

    // If the input data cannot be split into size_ segments then copy it and
    // all reduce a local buffer prefilled with 0s.
    if (input.size() < size_) {
      // TODO: Maybe allocate dynamically so we don't have the constraint below?
      if (input.itemsize() * size_ > 1024) {
        std::ostringstream msg;
        msg << "Can't perform the ring all reduce of " << output.size()
            << " elements with a ring of size " << size_;
        throw std::runtime_error(msg.str());
      }

      std::future<void> sent, recvd;
      auto barrier = std::make_unique<Barrier>(2);
      char buffer[1024];
      std::memset(buffer, 0, size_ * input.itemsize());
      std::memcpy(buffer, input.data<char>(), input.nbytes());
      sent = pool_.enqueue(
          ring_send<T>,
          std::reference_wrapper(*barrier),
          send_sockets_[0],
          rank_,
          size_,
          (T*)buffer,
          size_,
          -1);
      recvd = pool_.enqueue(
          ring_recv_sum<T>,
          std::reference_wrapper(*barrier),
          recv_sockets_[0],
          rank_,
          size_,
          (T*)buffer,
          size_,
          -1);
      sent.wait();
      recvd.wait();
      std::memcpy(output.data<char>(), buffer, output.nbytes());
      return;
    }

    // If not inplace all reduce then copy the input to the output first
    if (input.data<void>() != output.data<void>()) {
      std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
    }

    // All reduce in place. We have `send_channels_.size()` bidirectional
    // channels so let's split the message up and perform as many parallel
    // ring-reductions as possible.
    std::vector<std::future<void>> reductions;
    std::vector<std::unique_ptr<Barrier>> barriers;
    size_t packets = ceildiv(output.size(), size_ * PACKET_SIZE);

    // Large all reduce territory so let's use all we got
    if (packets >= 2 * send_sockets_.size()) {
      size_t segment = ceildiv(output.size(), 2 * send_sockets_.size());
      for (int i = 0; i < send_sockets_.size(); i++) {
        // 1st ring reduce
        barriers.emplace_back(std::make_unique<Barrier>(2));
        reductions.push_back(pool_.enqueue(
            ring_send<T>,
            std::reference_wrapper(*barriers.back()),
            send_sockets_[i],
            rank_,
            size_,
            output.data<T>() + 2 * i * segment,
            std::min(output.size() - 2 * i * segment, segment),
            -1));
        reductions.push_back(pool_.enqueue(
            ring_recv_sum<T>,
            std::reference_wrapper(*barriers.back()),
            recv_sockets_[i],
            rank_,
            size_,
            output.data<T>() + 2 * i * segment,
            std::min(output.size() - 2 * i * segment, segment),
            -1));

        // 2nd ring reduce
        barriers.emplace_back(std::make_unique<Barrier>(2));
        reductions.push_back(pool_.enqueue(
            ring_send<T>,
            std::reference_wrapper(*barriers.back()),
            recv_sockets_[i],
            rank_,
            size_,
            output.data<T>() + (2 * i + 1) * segment,
            std::min(output.size() - (2 * i + 1) * segment, segment),
            1));
        reductions.push_back(pool_.enqueue(
            ring_recv_sum<T>,
            std::reference_wrapper(*barriers.back()),
            send_sockets_[i],
            rank_,
            size_,
            output.data<T>() + (2 * i + 1) * segment,
            std::min(output.size() - (2 * i + 1) * segment, segment),
            1));
      }
    }

    // At least 2 reductions so we can be from small to medium
    else if (packets > 1) {
      size_t segment = ceildiv(output.size(), packets);
      for (int i = 0; i < send_sockets_.size(); i++) {
        barriers.emplace_back(std::make_unique<Barrier>(2));
        reductions.push_back(pool_.enqueue(
            ring_send<T>,
            std::reference_wrapper(*barriers.back()),
            send_sockets_[i],
            rank_,
            size_,
            output.data<T>() + i * segment,
            std::min(output.size() - i * segment, segment),
            -1));
        reductions.push_back(pool_.enqueue(
            ring_recv_sum<T>,
            std::reference_wrapper(*barriers.back()),
            recv_sockets_[i],
            rank_,
            size_,
            output.data<T>() + i * segment,
            std::min(output.size() - i * segment, segment),
            -1));
      }
      for (int i = 0; i < packets - send_sockets_.size(); i++) {
        barriers.emplace_back(std::make_unique<Barrier>(2));
        reductions.push_back(pool_.enqueue(
            ring_send<T>,
            std::reference_wrapper(*barriers.back()),
            recv_sockets_[i],
            rank_,
            size_,
            output.data<T>() + (send_sockets_.size() + i) * segment,
            std::min(
                output.size() - (send_sockets_.size() + i) * segment, segment),
            1));
        reductions.push_back(pool_.enqueue(
            ring_recv_sum<T>,
            std::reference_wrapper(*barriers.back()),
            send_sockets_[i],
            rank_,
            size_,
            output.data<T>() + (send_sockets_.size() + i) * segment,
            std::min(
                output.size() - (send_sockets_.size() + i) * segment, segment),
            1));
      }
    }

    // Small reduction which won't really benefit much from parallelization.
    // TODO: Verify that this is true cause PACKET_SIZE * size_ can still be a
    //       fairly large array.
    else {
      barriers.emplace_back(std::make_unique<Barrier>(2));
      reductions.push_back(pool_.enqueue(
          ring_send<T>,
          std::reference_wrapper(*barriers.back()),
          send_sockets_[0],
          rank_,
          size_,
          output.data<T>(),
          output.size(),
          -1));
      reductions.push_back(pool_.enqueue(
          ring_recv_sum<T>,
          std::reference_wrapper(*barriers.back()),
          recv_sockets_[0],
          rank_,
          size_,
          output.data<T>(),
          output.size(),
          -1));
    }

    // Wait for the reductions to finish.
    for (auto& f : reductions) {
      f.wait();
    }
  }

  int rank_;
  int size_;

  bool verbose_;

  ThreadPool pool_;

  std::vector<int> send_sockets_;
  std::vector<int> recv_sockets_;
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
