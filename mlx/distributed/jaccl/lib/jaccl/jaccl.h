// Copyright © 2025 Apple Inc.

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "jaccl/group.h"
#include "jaccl/rdma.h"

namespace jaccl {

class Config {
 public:
  Config();

  Config& set_rank(const char* rank_str);
  Config& set_rank(int rank);
  Config& set_coordinator(const char* coordinator);
  Config& set_coordinator(std::string coordinator);
  Config& set_devices_from_file(const char* dev_file);
  Config& set_devices(
      std::vector<std::vector<std::vector<std::string>>> devices);
  Config& prefer_ring(bool prefer = true);
  Config& set_all_gather(AllGatherFn agf);
  Config& set_all_gather_factory(std::function<AllGatherFn(int, int)> factory);

  bool is_valid_mesh() const;
  bool is_valid_ring() const;
  bool is_valid() const;

  int get_rank() const {
    return rank_;
  }

  int get_size() const {
    return size_;
  }

  std::string get_coordinator() const {
    return coordinator_;
  }

  bool get_prefer_ring() const {
    return prefer_ring_;
  }

  static Config from_env();

  friend std::shared_ptr<Group> init(const Config& cfg, bool strict);

 private:
  std::vector<std::string> get_mesh_connectivity() const;
  std::pair<std::vector<std::string>, std::vector<std::string>>
  get_ring_connectivity() const;
  SideChannel get_side_channel() const;

  int rank_;
  int size_;
  std::string coordinator_;
  std::vector<std::vector<std::vector<std::string>>> devices_;
  bool prefer_ring_;
  AllGatherFn all_gather_fn_;
  std::function<AllGatherFn(int, int)> all_gather_factory_;
};

/**
 * Check if JACCL (RDMA over Thunderbolt) is available on this system.
 */
bool is_available();

/**
 * Initialize a JACCL communication group from environment variables.
 *
 * Reads configuration from environment variables:
 *   - JACCL_RANK / MLX_RANK: The rank of this process
 *   - JACCL_IBV_DEVICES / MLX_IBV_DEVICES: Path to the device connectivity
 *     JSON file
 *   - JACCL_COORDINATOR / MLX_JACCL_COORDINATOR: IP:port of the coordinator
 *   - JACCL_RING / MLX_JACCL_RING: If set, prefer ring topology
 *
 * Args:
 *   strict: If true, throw on failure. If false, return nullptr.
 *
 * Returns:
 *   A shared_ptr to the Group, or nullptr on failure.
 */
std::shared_ptr<Group> init(bool strict = false);

/**
 * Initialize a JACCL communication group from environment variables, using a
 * custom all-gather factory for the side channel.
 *
 * The factory is called once per rank with the rank and group size, and must
 * return an all-gather function that will be used to exchange RDMA connection
 * metadata during setup.
 */
std::shared_ptr<Group> init(
    bool strict,
    std::function<AllGatherFn(int, int)> factory);

/**
 * Initialize a JACCL communication group from an explicit Config object.
 */
std::shared_ptr<Group> init(const Config& cfg, bool strict = false);

} // namespace jaccl
