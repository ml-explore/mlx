// Copyright © 2026 Apple Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>

namespace mlx::core {

class Error {
 public:
  // TODO: Use std::atomic<std::shared_ptr> when it gets supported in Xcode.
  using Message = std::shared_ptr<std::string>;

  void set_message(Message msg) {
    std::atomic_store(&message_, std::move(msg));
  }

  bool valid() const {
    auto msg = std::atomic_load(&message_);
    return msg.get();
  }

  // If |ptr| is a valid event, copy and return true.
  bool store_if_valid(const Error* ptr) {
    if (ptr && this != ptr) {
      Message msg = std::atomic_load(&ptr->message_);
      if (msg) {
        set_message(std::move(msg));
        return true;
      }
    }
    return false;
  }

  // If current error is valid, throw and clear.
  void check() {
    auto msg = std::atomic_exchange(&message_, {});
    if (msg) {
      throw std::runtime_error(*msg);
    }
  }

 private:
  Message message_;
};

} // namespace mlx::core
