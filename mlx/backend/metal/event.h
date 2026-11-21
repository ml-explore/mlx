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

  // A wait for the value can drop the refs. A value that does not go up clears
  // them, because its wait can return on an old signal.
  void set_refs(uint64_t value, std::weak_ptr<DataRefs> refs);

  // Call when the GPU passed the value.
  void drop_refs(uint64_t value);

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

  std::mutex refs_mtx_;
  uint64_t refs_value_{0};
  std::weak_ptr<DataRefs> refs_;
};

} // namespace mlx::core::metal
