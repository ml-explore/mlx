// Copyright © 2024 Apple Inc.

#include <unordered_map>

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"
#include "mlx/distributed/mpi/mpi.h"
#include "mlx/distributed/nccl/nccl.h"
#include "mlx/distributed/ring/ring.h"

namespace mlx::core::distributed {

namespace detail {

Stream communication_stream(Group group, StreamOrDevice s /* = {} */) {
  return group.raw_group()->communication_stream(s);
}

void all_sum(Group group, const array& input, array& output, Stream stream) {
  group.raw_group()->all_sum(input, output, stream);
}

void all_max(Group group, const array& input, array& output, Stream stream) {
  group.raw_group()->all_max(input, output, stream);
}

void all_min(Group group, const array& input, array& output, Stream stream) {
  group.raw_group()->all_min(input, output, stream);
}

void all_gather(Group group, const array& input, array& output, Stream stream) {
  group.raw_group()->all_gather(input, output, stream);
}

void send(Group group, const array& input, int dst, Stream stream) {
  group.raw_group()->send(input, dst, stream);
}

void recv(Group group, array& out, int src, Stream stream) {
  group.raw_group()->recv(out, src, stream);
}

class EmptyGroup : public GroupImpl {
 public:
  Stream communication_stream(StreamOrDevice s) override {
    return to_stream(s);
  }

  int rank() override {
    return 0;
  }

  int size() override {
    return 1;
  }

  std::shared_ptr<GroupImpl> split(int color, int key = -1) override {
    throw std::runtime_error("Cannot split the distributed group further.");
  }

  void all_sum(const array&, array&, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void all_gather(const array&, array&, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void send(const array&, int, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
  void recv(array&, int, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }

  void all_max(const array&, array&, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }

  void all_min(const array&, array&, Stream) override {
    throw std::runtime_error(
        "Communication not implemented in an empty distributed group.");
  }
};

} // namespace detail

bool is_available() {
  return mpi::is_available() || ring::is_available() || nccl::is_available();
}

int Group::rank() const {
  return group_->rank();
}

int Group::size() const {
  return group_->size();
}

Group Group::split(int color, int key /* = -1 */) const {
  return Group(group_->split(color, key));
}

Group init(bool strict /* = false */, const std::string& bk /* = "any" */) {
  static std::unordered_map<std::string, std::shared_ptr<detail::GroupImpl>>
      backends;

  // Already initialized so return the group.
  if (auto g = backends.find(bk); g != backends.end()) {
    return Group(g->second);
  }

  // Create the requested communication group
  std::shared_ptr<detail::GroupImpl> group;
  std::string bk_ = bk;
  if (bk == "mpi") {
    group = mpi::init(strict);
  } else if (bk == "ring") {
    group = ring::init(strict);
  } else if (bk == "nccl") {
    group = nccl::init(strict);
  } else if (bk == "any") {
    group = ring::init(false);
    bk_ = "ring";
    if (group == nullptr) {
      group = mpi::init(false);
      bk_ = "mpi";
    }
    if (group == nullptr && strict) {
      throw std::runtime_error("[distributed] Couldn't initialize any backend");
    }
  } else {
    std::ostringstream msg;
    msg << "[distributed] The only valid values for backend are 'any', 'mpi' "
        << "and 'ring' but '" << bk << "' was provided.";
    throw std::invalid_argument(msg.str());
  }

  if (group == nullptr) {
    group = std::make_shared<detail::EmptyGroup>();
  } else {
    backends.insert({"any", group});
  }
  backends.insert({std::move(bk_), group});
  return Group(group);
}

} // namespace mlx::core::distributed
