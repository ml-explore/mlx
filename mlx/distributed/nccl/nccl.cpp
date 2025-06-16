#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>

#include "mlx/backend/cuda/device.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"

namespace mlx::core::distributed::nccl {

#define CHECK_CUDA(cmd)              \
  do {                               \
    cudaError_t e = cmd;             \
    if (e != cudaSuccess) {          \
      fprintf(                       \
          stderr,                    \
          "CUDA error %s:%d '%s'\n", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(e));    \
      exit(1);                       \
    }                                \
  } while (0)

#define CHECK_NCCL(cmd)              \
  do {                               \
    ncclResult_t r = cmd;            \
    if (r != ncclSuccess) {          \
      fprintf(                       \
          stderr,                    \
          "NCCL error %s:%d '%s'\n", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(r));    \
      exit(1);                       \
    }                                \
  } while (0)

namespace detail {

inline void sendAll(int sock, const void* buf, size_t len) {
  const char* ptr = reinterpret_cast<const char*>(buf);
  while (len > 0) {
    ssize_t sent = send(sock, ptr, len, 0);
    if (sent <= 0) {
      perror("send");
      exit(1);
    }
    ptr += sent;
    len -= sent;
  }
}

inline void recvAll(int sock, void* buf, size_t len) {
  char* ptr = reinterpret_cast<char*>(buf);
  while (len > 0) {
    ssize_t rec = recv(sock, ptr, len, 0);
    if (rec <= 0) {
      perror("recv");
      exit(1);
    }
    ptr += rec;
    len -= rec;
  }
}

inline void bootstrapUniqueId(
    ncclUniqueId& id,
    int rank,
    int size,
    const std::string& initMethod) {
  // Parse the init method to extract the host and port
  if (initMethod.rfind("tcp://", 0) != 0)
    throw;
  auto hostport = initMethod.substr(6);
  auto colon = hostport.find(':');
  std::string host = hostport.substr(0, colon);
  int port = std::stoi(hostport.substr(colon + 1));

  if (rank == 0) {
    // create a unique id on the rank 0
    CHECK_NCCL(ncclGetUniqueId(&id));

    // create a socket to send the unique id to all other ranks
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0) {
      std::ostringstream msg;
      msg << "[nccl] Couldn't create socket (error: " << errno << ")";
      throw std::runtime_error(msg.str());
    }

    sockaddr_in serv = {};
    serv.sin_family = AF_INET;
    serv.sin_addr.s_addr = htonl(INADDR_ANY);
    serv.sin_port = htons(port);

    int reuse = 1;
    // Without this, if I crash or restart your rank-0 process quickly,
    // the OS might refuse to let you bind to the same port, so reuse
    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
      std::ostringstream msg;
      msg << "[nccl] setsockopt() failed: " << strerror(errno);
      throw std::runtime_error(msg.str());
    }

    if (bind(sock, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) < 0) {
      std::ostringstream msg;
      msg << "[nccl] bind() failed: " << strerror(errno);
      throw std::runtime_error(msg.str());
    }
    if (listen(sock, size - 1) < 0) {
      std::ostringstream msg;
      msg << "[nccl] listen() failed: " << strerror(errno);
      throw std::runtime_error(msg.str());
    }

    for (int peer = 1; peer < size; ++peer) {
      int conn = accept(sock, nullptr, nullptr);
      if (conn < 0) {
        std::ostringstream msg;
        msg << "[nccl] accept() failed: " << strerror(errno);
        throw std::runtime_error(msg.str());
      }
      sendAll(conn, &id, sizeof(id));
      close(conn);
    }
    close(sock);

  } else {
    // Here just wanted to make show that rank 0 has enough time to bind
    // so we will retry to connect until max attempts

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      std::ostringstream msg;
      msg << "[nccl] socket() failed: " << strerror(errno);
      throw std::runtime_error(msg.str());
    }

    hostent* he = gethostbyname(host.c_str());
    if (!he) {
      throw std::runtime_error("[nccl] lookup failed for host: " + host);
    }
    sockaddr_in serv = {};
    serv.sin_family = AF_INET;
    memcpy(&serv.sin_addr, he->h_addr_list[0], he->h_length);
    serv.sin_port = htons(port);

    const int max_retries = 30;
    int attempt = 0;
    bool connected = false;

    for (attempt = 0; attempt < max_retries; ++attempt) {
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) ==
          0) {
        connected = true;
        std::cout << "[Rank " << rank << "] Connected successfully on attempt "
                  << attempt + 1 << std::endl;
        break;
      }
      if (errno != ECONNREFUSED) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (!connected) {
      std::ostringstream msg;
      msg << "[Rank " << rank << "] connect() failed after " << attempt
          << " retries: " << strerror(errno);
      close(sock);
      throw std::runtime_error(msg.str());
    }
    recvAll(sock, &id, sizeof(id));
    close(sock);
  }
}

inline ncclDataType_t datatype(const array& arr) {
  switch (arr.dtype()) {
    case bool_:
      throw std::invalid_argument("[nccl] Boolean arrays not supported");
    case int8:
      return ncclChar;
    case uint8:
      return ncclUint8;
    case int32:
      return ncclInt;
    case uint32:
      return ncclUint32;
    case int64:
      return ncclInt64;
    case uint64:
      return ncclUint64;
    case float16:
      return ncclHalf;
    case float32:
      return ncclFloat;
    case float64:
      return ncclDouble;
    case bfloat16:
      return ncclBfloat16;
    default:
      throw std::invalid_argument("[nccl] Unknown or unsupported dtype");
  }
}

} // namespace detail

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
// init communication in the constructor (?)
class NCCLGroup : public GroupImpl {
 public:
  NCCLGroup(int worldRank, int worldSize, const std::string initMethod)
      : rank_(worldRank),
        size_(worldSize),
        comm_(nullptr),
        initMethod_(initMethod) {
    if (initialized_)
      return;
    int ndev;
    CHECK_CUDA(cudaGetDeviceCount(&ndev));
    CHECK_CUDA(cudaSetDevice(rank_ % ndev));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    detail::bootstrapUniqueId(uniqueId_, rank_, size_, initMethod_);
    CHECK_NCCL(ncclCommInitRank(&comm_, size_, uniqueId_, rank_));
    initialized_ = true;
  }

  ~NCCLGroup() {
    ncclCommDestroy(comm_);
    ncclGroupEnd();
    cudaStreamDestroy(stream_);
    initialized_ = false;
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    if (input.size() != output.size()) {
      throw std::runtime_error(
          "[nccl] Input and output arrays must have the same size.");
    }
    all_reduce_impl<float>(input, output, stream, ncclSum);
  }

  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[nccl] Group split not supported.");
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    if (input.size() != output.size() / size_) {
      throw std::runtime_error(
          "[nccl] Input size must match output size divided by group size.");
    }
  }

  void send(const array& input, int dst, Stream stream) override {
    if (input.size() == 0) {
      return; // Nothing to send
    }
  }

  void recv(array& output, int src, Stream stream) override {
    if (output.size() == 0) {
      return; // Nothing to receive
    }
  }

  void all_max(const array& input, array& output, Stream stream) override {
    if (input.size() != output.size()) {
      throw std::runtime_error(
          "[nccl] Input and output arrays must have the same size.");
    }
    all_reduce_impl<float>(input, output, stream, ncclMax);
  }

  void all_min(const array& input, array& output, Stream stream) override {
    if (input.size() != output.size()) {
      throw std::runtime_error(
          "[nccl] Input and output arrays must have the same size.");
    }
    all_reduce_impl<float>(input, output, stream, ncclMin);
  }

  template <typename T>
  void all_reduce_impl(
      const array& input,
      array& output,
      Stream stream,
      ncclRedOp_t op) {
    ncclDataType_t dt = detail::datatype(input);

    CHECK_NCCL(ncclAllReduce(
        input.data<T>(),
        output.data<T>(),
        input.size(),
        dt,
        op,
        comm_,
        stream_));
  }

  int rank_, size_;
  std::string initMethod_;
  ncclUniqueId uniqueId_;
  ncclComm_t comm_;
  cudaStream_t stream_;
  bool initialized_ = false;
};

bool is_available() {
  return true;
}

namespace detail {
static std::string get_env_var_or_throw(const char* env_var_name) {
  const char* value = std::getenv(env_var_name);
  if (value == nullptr) {
    std::ostringstream msg;
    msg << "[nccl] Required environment variable '" << env_var_name
        << "' is not set. "
        << "Please set it before initializing the distributed backend.";
    throw std::runtime_error(msg.str());
  }
  return std::string(value);
}
} // namespace detail

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  std::string host = detail::get_env_var_or_throw("NCCL_HOST_IP");
  std::string port = detail::get_env_var_or_throw("NCCL_PORT");
  std::string rank_str = detail::get_env_var_or_throw("MLX_RANK");
  std::string n_nodes_str = detail::get_env_var_or_throw("MLX_WORLD_SIZE");

  int rank = std::stoi(rank_str);
  int n_nodes = std::stoi(n_nodes_str);
  std::string init_method = "tcp://" + host + ":" + port;

  return std::make_shared<NCCLGroup>(rank, n_nodes, init_method);
}
} // namespace mlx::core::distributed::nccl
