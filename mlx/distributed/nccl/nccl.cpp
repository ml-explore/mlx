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
#include <type_traits>

#include "mlx/backend/cuda/device.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

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

#define MLX_NCCL_TYPE_LIST(X) \
  X(int8_t, ncclChar)         \
  X(uint8_t, ncclUint8)       \
  X(int32_t, ncclInt)         \
  X(uint32_t, ncclUint32)     \
  X(int64_t, ncclInt64)       \
  X(uint64_t, ncclUint64)     \
  X(float16_t, ncclHalf)      \
  X(bfloat16_t, ncclBfloat16) \
  X(float, ncclFloat)         \
  X(double, ncclDouble)

template <class>
struct nccl_map {
  static constexpr bool ok = false; // default: unsupported
};

#define MLX_DEF_NCCL_MAP(T, E)                 \
  template <>                                  \
  struct nccl_map<T> {                         \
    static constexpr bool ok = true;           \
    static constexpr ncclDataType_t value = E; \
  };

MLX_NCCL_TYPE_LIST(MLX_DEF_NCCL_MAP)
#undef MLX_DEF_NCCL_MAP

namespace detail {

template <typename F>
void dispatch_dtype(const array& arr, F&& f) {
  dispatch_all_types(arr.dtype(), [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    if constexpr (nccl_map<T>::ok) {
      f(type_tag, nccl_map<T>::value);
    } else {
      throw std::invalid_argument("[nccl] Unknown or unsupported dtype");
    }
  });
}

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

inline void bootstrap_unique_id(
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
    // Without this, if rank-0 crashes or restarts process quickly,
    // the OS might refuse to let binding to the same port, so reuse

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

    bool do_log = std::getenv("NCCL_DEBUG") == "INFO";
    for (attempt = 0; attempt < max_retries; ++attempt) {
      if (connect(sock, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) ==
          0) {
        connected = true;
        if (do_log) {
          std::cout << "[Rank " << rank
                    << "] Connected successfully on attempt " << attempt + 1
                    << std::endl;
          break;
        }
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

} // namespace detail

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
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
    detail::bootstrap_unique_id(uniqueId_, rank_, size_, initMethod_);
    CHECK_NCCL(ncclCommInitRank(&comm_, size_, uniqueId_, rank_));
    initialized_ = true;
  }

  ~NCCLGroup() {
    ncclCommDestroy(comm_);
    ncclGroupEnd();
    initialized_ = false;
  }

  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s, Device::gpu);
  }

  int rank() override {
    return rank_;
  }

  int size() override {
    return size_;
  }

  void all_sum(const array& input, array& output, Stream stream) override {
    detail::dispatch_dtype(input, [&](auto type_tag, ncclDataType_t dt) {
      using T = typename decltype(type_tag)::type;
      all_reduce_impl<T>(input, output, stream, dt, ncclSum);
    });
  }

  virtual std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("[nccl] Group split not supported.");
  }

  void all_gather(const array& input, array& output, Stream stream) override {
    throw std::runtime_error(
        "[nccl] All gather not supported in NCCL backend.");
  }

  void send(const array& input, int dst, Stream stream) override {
    throw std::runtime_error("[nccl] Send not supported in NCCL backend.");
  }

  void recv(array& output, int src, Stream stream) override {
    throw std::runtime_error("[nccl] Recv not supported in NCCL backend.");
  }

  void all_max(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[nccl] All max not supported in NCCL backend.");
  }

  void all_min(const array& input, array& output, Stream stream) override {
    throw std::runtime_error("[nccl] All min not supported in NCCL backend.");
  }

  template <typename T>
  void all_reduce_impl(
      const array& input,
      array& output,
      Stream stream,
      ncclDataType_t dt,
      ncclRedOp_t op) {
    auto& encoder = cu::get_command_encoder(stream);

    CHECK_NCCL(ncclAllReduce(
        input.data<T>(),
        output.data<T>(),
        input.size(),
        dt,
        op,
        comm_,
        encoder.stream()));
  }

  int rank_, size_;
  std::string initMethod_;
  ncclUniqueId uniqueId_;
  ncclComm_t comm_;
  bool initialized_ = false;
};

bool is_available() {
  return true;
}

namespace detail {
std::string get_env_var_or_throw(const char* env_var_name, bool strict) {
  const char* value = std::getenv(env_var_name);
  if (value == nullptr && strict) {
    std::ostringstream msg;
    msg << "[nccl] Required environment variable '" << env_var_name
        << "' is not set. "
        << "Please set it before initializing the distributed backend.";
    throw std::runtime_error(msg.str());
  }
  if (value == nullptr) {
    return "";
  }
  return std::string(value);
}
} // namespace detail

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  std::string host = detail::get_env_var_or_throw("NCCL_HOST_IP", strict);
  std::string port = detail::get_env_var_or_throw("NCCL_PORT", strict);
  std::string rank_str = detail::get_env_var_or_throw("MLX_RANK", strict);
  std::string n_nodes_str =
      detail::get_env_var_or_throw("MLX_WORLD_SIZE", strict);
  if (!strict &&
      (host.empty() || port.empty() || rank_str.empty() ||
       n_nodes_str.empty())) {
    return nullptr;
  }

  int rank = std::stoi(rank_str);
  int n_nodes = std::stoi(n_nodes_str);
  std::string init_method = "tcp://" + host + ":" + port;

  return std::make_shared<NCCLGroup>(rank, n_nodes, init_method);
}
} // namespace mlx::core::distributed::nccl
