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

#include <json.hpp>

#include "mlx/backend/common/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/io/threadpool.h"

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
constexpr const int MAX_THREADS = 6;

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
using json = nlohmann::json;

namespace {

struct address_t {
  sockaddr_storage addr;
  socklen_t len;

  const sockaddr* sockaddr() const {
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
    success = bind(sock, address.sockaddr(), address.len);
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
std::vector<int> make_connections(const std::vector<address_t>& addresses) {
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
        std::cout << "Attempt " << attempt << " wait " << wait << " ms "
                  << "error: " << errno << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(wait));
      }

      success = connect(sock, address.sockaddr(), address.len);
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
  data += start;
  char buffer[PACKET_SIZE];
  size_t len = (stop - start) * sizeof(T);
  while (len > 0) {
    ssize_t r = recv(sock, buffer, std::min(len, PACKET_SIZE), 0);
    if (r <= 0) {
      std::ostringstream msg;
      msg << "Recv of " << len << " bytes failed (errno: " << errno << ")";
      throw std::runtime_error(msg.str());
    }
    sum_inplace((const T*)buffer, data, r / sizeof(T));
    data += r / sizeof(T);
    len -= r;
  }
}

} // namespace

class RingGroup : public GroupImpl {
 public:
  RingGroup(int rank, std::vector<std::vector<address_t>> nodes)
      : rank_(rank), pool_(MAX_THREADS) {
    if (rank_ > 0 && rank_ >= nodes.size()) {
      throw std::runtime_error(
          "[ring] Rank cannot be larger than the size of the group");
    }

    size_ = nodes.size();
    int sendto = (rank_ + 1) % size_;

    int success;

    if (rank_ < sendto) {
      std::cout << "Rank " << rank_ << " accepting" << std::endl;
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
      std::cout << "Rank " << rank_ << " connecting to " << sendto << std::endl;
      send_sockets_ = std::move(make_connections(nodes[sendto]));
    } else {
      std::cout << "Rank " << rank_ << " connecting to " << sendto << std::endl;
      send_sockets_ = std::move(make_connections(nodes[sendto]));
      std::cout << "Rank " << rank_ << " accepting" << std::endl;
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
    }

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
    // Make sure that the input is row contiguous
    array input = ensure_row_contiguous(input_);

    // If not inplace all reduce then copy the input to the output first
    if (input.data<void>() != output.data<void>()) {
      std::memcpy(output.data<char>(), input.data<char>(), input.nbytes());
    }

    // All reduce in place
    if (output.size() < size_) {
      if (output.itemsize() * size_ > 1024) {
        std::ostringstream msg;
        msg << "Can't perform the ring all reduce of " << output.size()
            << " elements with a ring of size " << size_;
        throw std::runtime_error(msg.str());
      }
      char buffer[1024];
      memset(buffer, 0, size_ * output.itemsize());
      memcpy(buffer, output.data<char>(), output.nbytes());
      SWITCH_TYPE(output, all_sum<T>((T*)buffer, size_));
      memcpy(output.data<char>(), buffer, output.nbytes());
    } else {
      SWITCH_TYPE(output, all_sum<T>(output.data<T>(), output.size()));
    }
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
  void all_sum(T* data, size_t data_size) {
    size_t send_channels = send_sockets_.size();
    size_t recv_channels = recv_sockets_.size();
    std::vector<std::future<void>> futures;
    futures.reserve(send_channels + recv_channels);

    size_t step = data_size / size_ + (data_size % size_ > 0);

    // Scatter reduce steps
    int send_segment = rank_;
    int recv_segment = (send_segment + size_ - 1) % size_;
    for (int i = 0; i < size_ - 1; i++) {
      // Compute the send and recv locations
      size_t send_start = send_segment * step;
      size_t send_stop = std::min((send_segment + 1) * step, data_size);
      size_t recv_start = recv_segment * step;
      size_t recv_stop = std::min((recv_segment + 1) * step, data_size);

      // Send and recv sum
      size_t send_size = send_stop - send_start;
      size_t send_step =
          send_size / send_channels + (send_size % send_channels > 0);
      size_t recv_size = recv_stop - recv_start;
      size_t recv_step =
          recv_size / recv_channels + (recv_size % recv_channels > 0);
      for (int i = 0; i < std::max(send_channels, recv_channels); i++) {
        if (i < send_sockets_.size()) {
          futures.push_back(pool_.enqueue(
              _send<T>,
              send_sockets_[i],
              data,
              send_start + i * send_step,
              std::min(send_stop, send_start + (i + 1) * send_step)));
        }
        if (i < recv_sockets_.size()) {
          futures.push_back(pool_.enqueue(
              _recv_sum<T>,
              recv_sockets_[i],
              data,
              recv_start + i * recv_step,
              std::min(recv_stop, recv_start + (i + 1) * recv_step)));
        }
      }

      // Wait for all the communication to finish
      for (auto& f : futures) {
        f.wait();
      }
      futures.clear();

      send_segment = (send_segment + size_ - 1) % size_;
      recv_segment = (recv_segment + size_ - 1) % size_;
    }

    // Gather results
    for (int i = 0; i < size_ - 1; i++) {
      // Compute the send and recv locations
      size_t send_start = send_segment * step;
      size_t send_stop = std::min((send_segment + 1) * step, data_size);
      size_t recv_start = recv_segment * step;
      size_t recv_stop = std::min((recv_segment + 1) * step, data_size);

      // Send and recv
      size_t send_size = send_stop - send_start;
      size_t send_step =
          send_size / send_channels + (send_size % send_channels > 0);
      size_t recv_size = recv_stop - recv_start;
      size_t recv_step =
          recv_size / recv_channels + (recv_size % recv_channels > 0);
      for (int i = 0; i < std::max(send_channels, recv_channels); i++) {
        if (i < send_sockets_.size()) {
          futures.push_back(pool_.enqueue(
              _send<T>,
              send_sockets_[i],
              data,
              send_start + i * send_step,
              std::min(send_stop, send_start + (i + 1) * send_step)));
        }
        if (i < recv_sockets_.size()) {
          futures.push_back(pool_.enqueue(
              _recv<T>,
              recv_sockets_[i],
              data,
              recv_start + i * recv_step,
              std::min(recv_stop, recv_start + (i + 1) * recv_step)));
        }
      }

      // Wait for all the communication to finish
      for (auto& f : futures) {
        f.wait();
      }
      futures.clear();

      send_segment = (send_segment + size_ - 1) % size_;
      recv_segment = (recv_segment + size_ - 1) % size_;
    }
  }

  int rank_;
  int size_;

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

  return std::make_shared<RingGroup>(rank, nodes);
}

} // namespace mlx::core::distributed::ring
