// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/encoder.h"

#include <fmt/format.h>
#include <thread>

namespace mlx::core::cpu {

CommandEncoder& get_command_encoder(Stream s) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          fmt::format(
              "There is no Stream(cpu, {}) in current thread.", s.index));
    }
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

size_t thread_pool_size() {
  static size_t size = [] {
    auto n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : n;
  }();
  return size;
}

ThreadPool& thread_pool() {
  // Leak - see Scheduler singleton comment in scheduler.cpp.
  static ThreadPool* pool = new ThreadPool{thread_pool_size()};
  return *pool;
}

} // namespace mlx::core::cpu
