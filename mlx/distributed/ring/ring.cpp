// Copyright © 2024 Apple Inc.

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

#include <json.hpp>

#include "mlx/backend/common/copy.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/io/threadpool.h"

namespace mlx::core::distributed::ring {

constexpr const size_t PACKET_SIZE = 262144;
constexpr const int CONN_ATTEMPTS = 5;
constexpr const int CONN_WAIT = 1000;

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
 *    ["hostname1:5000", "hostname1:5001"],
 *    ["hostname2:5000", "hostname2:5001"],
 *    ["hostname3:5000", "hostname3:5001"],
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
    // Create the socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::ostringstream msg;
      msg << "[ring] Couldn't create socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    // Attempt to connect to the peer CONN_ATTEMPTS times with exponential
    // backoff. TODO: Do we need that?
    for (int attempt = 0; attempt < CONN_ATTEMPTS; attempt++) {
      if (attempt > 0) {
        int wait = (1 << (attempt - 1)) * CONN_WAIT;
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
  }

  return sockets;
}

} // namespace

class RingGroup : public GroupImpl {
 public:
  RingGroup(int rank, std::vector<std::vector<address_t>> nodes) : rank_(rank) {
    if (rank_ > 0 && rank_ >= nodes.size()) {
      throw std::runtime_error(
          "[ring] Rank cannot be larger than the size of the group");
    }

    size_ = nodes.size();
    int sendto = (rank_ + 1) % size_;

    int success;

    if (rank_ < sendto) {
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
      send_sockets_ = std::move(make_connections(nodes[sendto]));
    } else {
      send_sockets_ = std::move(make_connections(nodes[sendto]));
      recv_sockets_ = std::move(accept_connections(nodes[rank_]));
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

  void all_sum(const array& input, array& output) override {}

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
  int rank_;
  int size_;

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
