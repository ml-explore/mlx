// Copyright © 2025 Apple Inc.

#pragma once

#include <unordered_map>

#include "mlx/array.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

struct CommandEncoder {
  CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  CommandEncoder(CommandEncoder&&) = delete;
  CommandEncoder& operator=(CommandEncoder&&) = delete;

  void set_input_array(const array& a) {}
  void set_output_array(array& a) {}

  // Hold onto a temporary until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporary(array a) {
    scheduler::enqueue(stream_, [a = std::move(a)]() {});
  }

  // Hold onto temporaries until any already scheduled tasks which use it as
  // an input are complete.
  void add_temporaries(std::vector<array> arrays) {
    if (arrays.empty()) {
      return;
    }
    scheduler::enqueue(stream_, [arrays = std::move(arrays)]() {});
  }

  template <class F, class... Args>
  void dispatch(F&& f, Args&&... args) {
    auto task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    scheduler::enqueue(stream_, std::move(task));
  }

 private:
  Stream stream_;
};

CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::cpu
