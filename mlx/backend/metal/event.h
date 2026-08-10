// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/event.h"

namespace mlx::core::metal {

class EventImpl {
 public:
  EventImpl(Device& d);
  ~EventImpl();

  // Blocks until signaled. Does not throw: a failure that reaches this event
  // was parked on its stream first, so the stream is where it gets reported.
  void wait(uint64_t value);
  void signal(uint64_t value);
  void set_error(std::shared_ptr<std::string> error);

  const auto& error() const {
    return error_;
  }

  auto* mtl_event() {
    return mtl_event_.get();
  }

 private:
  // TODO: Use std::atomic<std::shared_ptr> when it gets supported in Xcode.
  std::shared_ptr<std::string> error_;

  NS::SharedPtr<MTL::SharedEvent> mtl_event_;
};

} // namespace mlx::core::metal
