// Copyright © 2025 Apple Inc.

#include <fstream>

#include <json.hpp>

#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/distributed/jaccl/utils.h"

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
using json = nlohmann::json;

namespace {

std::vector<std::string> load_device_names(int rank, const char* dev_file) {
  std::vector<std::string> device_names;
  std::ifstream f(dev_file);

  json devices = json::parse(f);
  devices = devices[rank];
  for (auto it = devices.begin(); it != devices.end(); it++) {
    std::string n;
    if (!it->is_null()) {
      n = *it;
    }
    device_names.emplace_back(std::move(n));
  }

  return device_names;
}

} // namespace

namespace mlx::core::distributed::jaccl {

bool is_available() {
  return ibv().is_available();
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  const char* dev_file = std::getenv("MLX_IBV_DEVICES");
  const char* coordinator = std::getenv("MLX_JACCL_COORDINATOR");
  const char* rank_str = std::getenv("MLX_RANK");

  if (!is_available() || !dev_file || !coordinator || !rank_str) {
    if (strict) {
      std::ostringstream msg;
      msg << "[jaccl] You need to provide via environment variables a rank (MLX_RANK), "
          << "a device file (MLX_IBV_DEVICES) and a coordinator ip/port (MLX_JACCL_COORDINATOR) "
          << "but provided MLX_RANK=\"" << ((rank_str) ? rank_str : "")
          << "\", MLX_IBV_DEVICES=\"" << ((dev_file) ? dev_file : "")
          << "\" and MLX_JACCL_COORDINATOR=\""
          << ((coordinator) ? coordinator : "");
      throw std::runtime_error(msg.str());
    }
    return nullptr;
  }

  auto rank = std::atoi(rank_str);
  auto device_names = load_device_names(rank, dev_file);

  return std::make_shared<MeshGroup>(rank, device_names, coordinator);
}

} // namespace mlx::core::distributed::jaccl
