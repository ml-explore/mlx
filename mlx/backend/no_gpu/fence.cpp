// Copyright © 2024 Apple Inc.

#include "mlx/fence.h"
#include "mlx/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Event event;

  FenceImpl(uint32_t count, Stream s) : count(count), event(s) {}
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
}

void Fence::wait(Stream s, const array&) {
  cast<FenceImpl>().event.wait(s);
}

void Fence::update(Stream s, const array&, bool) {
  auto& f = cast<FenceImpl>();
  f.count++;
  f.event.set_value(f.count);
  f.event.signal(s);
}

} // namespace mlx::core
