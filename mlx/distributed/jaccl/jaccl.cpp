// Copyright © 2025 Apple Inc.

#include <fstream>
#include <sstream>

#include <json.hpp>

#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/jaccl/mesh.h"
#include "mlx/distributed/jaccl/ring.h"
#include "mlx/distributed/jaccl/utils.h"

using GroupImpl = mlx::core::distributed::detail::GroupImpl;
using json = nlohmann::json;

namespace {

struct DeviceFile {
  DeviceFile(const char* dev_file) {
    std::ifstream f(dev_file);
    json devices = json::parse(f);
    if (!devices.is_array()) {
      throw std::runtime_error(
          "[jaccl] The device file should start with an array");
    }

    devices_.resize(devices.size());
    for (int rank = 0; rank < devices.size(); rank++) {
      auto conn = devices[rank];
      if (!conn.is_array()) {
        throw std::runtime_error(
            "[jaccl] The device file should have an array of arrays");
      }
      if (conn.size() != devices_.size()) {
        std::ostringstream msg;
        msg << "[jaccl] The device file should contain the connectivity of each rank to "
            << "all other ranks but rank " << rank << " contains only "
            << conn.size() << " entries.";
        throw std::runtime_error(msg.str());
      }

      devices_[rank].resize(conn.size());
      for (int dst = 0; dst < conn.size(); dst++) {
        auto names = conn[dst];
        if (names.is_string()) {
          devices_[rank][dst].push_back(names);
        } else if (names.is_array()) {
          for (auto name_it = names.begin(); name_it != names.end();
               name_it++) {
            devices_[rank][dst].push_back(*name_it);
          }
        } else if (!names.is_null()) {
          throw std::runtime_error(
              "[jaccl] Device names should be null, a string or array of strings.");
        }
      }
    }
  }

  int size() {
    return devices_.size();
  }

  bool is_valid_mesh() {
    for (int src = 0; src < size(); src++) {
      for (int dst = 0; dst < size(); dst++) {
        if (devices_[src][dst].size() != static_cast<size_t>(src != dst)) {
          return false;
        }
      }
    }

    return true;
  }

  bool is_valid_ring() {
    int num_connections = devices_[0][1].size();
    if (num_connections == 0) {
      return false;
    }

    for (int src = 0; src < size(); src++) {
      int left = (src + size() - 1) % size();
      int right = (src + 1) % size();
      for (int dst = 0; dst < size(); dst++) {
        if (dst != left && dst != right) {
          if (devices_[src][dst].size() != 0) {
            return false;
          }
        } else {
          if (devices_[src][dst].size() != num_connections) {
            return false;
          }
        }
      }
    }

    return true;
  }

  std::vector<std::string> extract_mesh_connectivity(int rank) {
    std::vector<std::string> devices(size());
    for (int dst = 0; dst < size(); dst++) {
      if (dst != rank) {
        devices[dst] = devices_[rank][dst][0];
      }
    }
    return devices;
  }

  std::pair<std::vector<std::string>, std::vector<std::string>>
  extract_ring_connectivity(int rank) {
    int left = (rank + size() - 1) % size();
    int right = (rank + 1) % size();

    return std::make_pair(devices_[rank][left], devices_[rank][right]);
  }

  std::vector<std::vector<std::vector<std::string>>> devices_;
};

} // namespace

namespace mlx::core::distributed::jaccl {

bool is_available() {
  return ibv().is_available();
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  const char* dev_file = std::getenv("MLX_IBV_DEVICES");
  const char* coordinator = std::getenv("MLX_JACCL_COORDINATOR");
  const char* rank_str = std::getenv("MLX_RANK");
  const char* ring = std::getenv("MLX_JACCL_RING");

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
  bool prefer_ring = ring != nullptr;
  DeviceFile devices(dev_file);

  if (rank >= devices.size() || rank < 0) {
    std::ostringstream msg;
    msg << "[jaccl] Invalid rank " << rank << ". It should be between 0 and "
        << devices.size();
    throw std::runtime_error(msg.str());
  }

  if (prefer_ring && devices.is_valid_ring()) {
    auto [left, right] = devices.extract_ring_connectivity(rank);
    return std::make_shared<RingGroup>(
        rank, devices.size(), left, right, coordinator);
  } else if (devices.is_valid_mesh()) {
    auto device_names = devices.extract_mesh_connectivity(rank);
    return std::make_shared<MeshGroup>(rank, device_names, coordinator);
  } else if (devices.is_valid_ring()) {
    auto [left, right] = devices.extract_ring_connectivity(rank);
    return std::make_shared<RingGroup>(
        rank, devices.size(), left, right, coordinator);
  } else {
    throw std::runtime_error(
        "[jaccl] The device file should define a valid mesh or a valid ring.");
  }
}

} // namespace mlx::core::distributed::jaccl
