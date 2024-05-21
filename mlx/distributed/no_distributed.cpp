// Copyright © 2024 Apple Inc.

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed {

namespace {

struct DummyGroup : public Group {
  virtual int rank() override {
    return 0;
  }
  virtual int size() override {
    return 1;
  }
  virtual std::shared_ptr<Group> split(int color, int key = -1) override {
    throw std::runtime_error("Cannot split the distributed group further");
  }
};

} // namespace

namespace detail {

Stream communication_stream() {
  static Stream comm_stream = new_stream(Device::cpu);
  return comm_stream;
}

} // namespace detail

bool is_available() {
  return false;
}

std::shared_ptr<Group> init() {
  return std::make_shared<DummyGroup>();
}

void all_reduce_sum(
    std::shared_ptr<Group> group,
    const array& input,
    array& output) {}
void all_gather(
    std::shared_ptr<Group> group,
    const array& input,
    array& output) {}

} // namespace mlx::core::distributed
