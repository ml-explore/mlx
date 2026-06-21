// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/metal/device.h"
#include "mlx/event.h"

#include <mutex>

namespace mlx::core::metal {

class EventImpl {
 public:
  EventImpl(Device& d);
  ~EventImpl();

  void wait(uint64_t value);
  void signal(uint64_t value);
  void set_error(std::shared_ptr<std::string> error);
  void set_error(std::exception_ptr error, uint64_t value);
  void check_error();
  std::exception_ptr exception() const;

  const auto& error() const {
    return error_;
  }

  auto* mtl_event() {
    return mtl_event_.get();
  }

 private:
  // TODO: Use std::atomic<std::shared_ptr> when it gets supported in Xcode.
  std::shared_ptr<std::string> error_;
  std::exception_ptr exception_;
  mutable std::mutex exception_mtx_;

  NS::SharedPtr<MTL::SharedEvent> mtl_event_;
};

} // namespace mlx::core::metal
