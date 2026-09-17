// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/event.h"

namespace mlx::core::metal {

class EventImpl {
 public:
  EventImpl(Device& d);
  ~EventImpl();

  void wait(uint64_t value);
  void signal(uint64_t value);

  auto& error() {
    return error_;
  }

  auto* mtl_event() const {
    return mtl_event_.get();
  }

 private:
  // All streams outlive events so pointers would be always valid.
  std::atomic<Error*> error_;

  NS::SharedPtr<MTL::SharedEvent> mtl_event_;
};

} // namespace mlx::core::metal
