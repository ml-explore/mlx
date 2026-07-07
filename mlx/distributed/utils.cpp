// Copyright © 2025 Apple Inc.

#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <thread>

#include "mlx/distributed/utils.h"

namespace mlx::core::distributed::detail {

/**
 * Parse a sockaddr from an ip and port provided as strings.
 */
address_t parse_address(const std::string& ip, const std::string& port) {
  struct addrinfo hints, *res;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  if (status != 0) {
    std::ostringstream msg;
    msg << "Can't parse address " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

  // Keep all resolved addresses (v4 and v6) so socket setup can try each
  // candidate until one binds/connects.
  address_t result;
  for (struct addrinfo* p = res; p != nullptr; p = p->ai_next) {
    address_t::candidate_t c;
    c.family = p->ai_family;
    c.socktype = p->ai_socktype;
    c.protocol = p->ai_protocol;
    memcpy(&c.addr, p->ai_addr, p->ai_addrlen);
    c.len = p->ai_addrlen;
    result.candidates.push_back(c);
  }
  freeaddrinfo(res);

  if (result.candidates.empty()) {
    std::ostringstream msg;
    msg << "No usable addresses for " << ip << ":" << port;
    throw std::runtime_error(msg.str());
  }

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

TCPSocket::TCPSocket(const char* tag) {
  // Socket creation is deferred to listen()/connect() where the address
  // family of the chosen candidate is known.
  (void)tag;
  sock_ = -1;
}

TCPSocket::TCPSocket(TCPSocket&& s) {
  sock_ = s.sock_;
  s.sock_ = -1;
}

TCPSocket& TCPSocket::operator=(TCPSocket&& s) {
  if (this != &s) {
    sock_ = s.sock_;
    s.sock_ = -1;
  }
  return *this;
}

TCPSocket::TCPSocket(int s) : sock_(s) {}

TCPSocket::~TCPSocket() {
  if (sock_ > 0) {
    shutdown(sock_, 2);
    close(sock_);
  }
}

int TCPSocket::detach() {
  int s = sock_;
  sock_ = -1;
  return s;
}

void TCPSocket::listen(const char* tag, const address_t& addr) {
  int last_errno = 0;

  // Try each resolved candidate and keep the first that binds and listens.
  for (const auto& c : addr.candidates) {
    int sock = socket(c.family, c.socktype, c.protocol);
    if (sock < 0) {
      last_errno = errno;
      continue;
    }

    // Make sure we can launch immediately after shutdown by setting the
    // reuseaddr option so that we don't get address already in use errors
    int enable = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0 ||
        setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int)) < 0) {
      last_errno = errno;
      close(sock);
      continue;
    }

    // Bind the socket to the address and port
    if (bind(sock, c.get(), c.len) < 0) {
      last_errno = errno;
      close(sock);
      continue;
    }

    // Prepare waiting for connections
    if (::listen(sock, 0) < 0) {
      last_errno = errno;
      close(sock);
      continue;
    }

    sock_ = sock;
    return;
  }

  std::ostringstream msg;
  msg << tag << " Couldn't listen on any resolved address (last error: "
      << last_errno << ")";
  throw std::runtime_error(msg.str());
}

TCPSocket TCPSocket::accept(const char* tag) {
  int peer = ::accept(sock_, nullptr, nullptr);
  if (peer < 0) {
    std::ostringstream msg;
    msg << tag << " Accept failed (error: " << errno << ")";
    throw std::runtime_error(msg.str());
  }

  return TCPSocket(peer);
}

void TCPSocket::send(const char* tag, const void* data, size_t len) {
  while (len > 0) {
    auto n = ::send(sock_, data, len, 0);
    if (n <= 0) {
      std::ostringstream msg;
      msg << tag << " Send failed with errno=" << errno;
      throw std::runtime_error(msg.str());
    }
    len -= n;
    data = static_cast<const char*>(data) + n;
  }
}

void TCPSocket::recv(const char* tag, void* data, size_t len) {
  while (len > 0) {
    auto n = ::recv(sock_, data, len, 0);
    if (n <= 0) {
      std::ostringstream msg;
      msg << tag << " Recv failed with errno=" << errno;
      throw std::runtime_error(msg.str());
    }
    len -= n;
    data = static_cast<char*>(data) + n;
  }
}

TCPSocket TCPSocket::connect(
    const char* tag,
    const address_t& addr,
    int num_retries,
    int wait,
    std::function<void(int, int)> cb) {
  int last_errno = 0;

  // Attempt to connect `num_retries` times with exponential backoff. Each
  // attempt tries every resolved candidate; failed sockets are closed before
  // moving on.
  for (int attempt = 0; attempt < num_retries; attempt++) {
    for (const auto& c : addr.candidates) {
      int sock = socket(c.family, c.socktype, c.protocol);
      if (sock < 0) {
        last_errno = errno;
        continue;
      }

      if (::connect(sock, c.get(), c.len) == 0) {
        return TCPSocket(sock);
      }

      last_errno = errno;
      close(sock);
    }

    if (cb != nullptr) {
      cb(attempt, wait);
    }
    if (wait > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(wait));
    }

    wait <<= 1;
  }

  std::ostringstream msg;
  msg << tag << " Couldn't connect to any resolved address (last error: "
      << last_errno << ")";
  throw std::runtime_error(msg.str());
}

} // namespace mlx::core::distributed::detail
