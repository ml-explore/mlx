// Copyright © 2024 Apple Inc.

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"

namespace mlx::core::distributed {

int Group::rank() {
  return 0;
}

int Group::size() {
  return 1;
}

Group Group::split(int color, int key) {
  throw std::runtime_error("Cannot split the distributed group further");
}

void Group::barrier() {}

bool is_available() {
  return false;
}

Group init(bool strict /* = false */) {
  return Group(nullptr);
}

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

void all_sum(Group group, const array& input, array& output) {}
void all_gather(Group group, const array& input, array& output) {}
void send(Group group, const array& input, int dst) {}
void recv(Group group, array& out, int src) {}

} // namespace detail

} // namespace mlx::core::distributed
