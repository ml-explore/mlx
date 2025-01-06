// Copyright © 2024 Apple Inc.

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/mpi/mpi.h"
#include "mlx/scheduler.h"

namespace mlx::core::distributed {

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

void all_sum(Group group, const array& input, array& output) {
  group.raw_group()->all_sum(input, output);
}

void all_gather(Group group, const array& input, array& output) {
  group.raw_group()->all_gather(input, output);
}

void send(Group group, const array& input, int dst) {
  group.raw_group()->send(input, dst);
}

void recv(Group group, array& out, int src) {
  group.raw_group()->send(out, src);
}

class EmptyGroup : public GroupImpl {
 public:
  int rank() override {
    return 0;
  }

  int size() override {
    return 1;
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("Cannot split the distributed group further.");
  }

  void all_sum(const array& input, array& output) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void all_gather(const array& input, array& output) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void send(const array& input, int dst) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void recv(array& out, int src) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
};

} // namespace detail

bool is_available() {
  return mpi::is_available();
}

int Group::rank() {
  return group_->rank();
}

int Group::size() {
  return group_->size();
}

Group Group::split(int color, int key /* = -1 */) {
  return Group(group_->split(color, key));
}

Group init(bool strict /* = false */) {
  static std::shared_ptr<detail::GroupImpl> default_group = nullptr;

  if (default_group == nullptr) {
    default_group = mpi::init(strict);
  }

  if (default_group == nullptr) {
    default_group = std::make_shared<detail::EmptyGroup>();
  }

  // Ensure the communication stream is alive before
  // the graph is evaluated
  detail::communication_stream();
  return Group(default_group);
}

} // namespace mlx::core::distributed
