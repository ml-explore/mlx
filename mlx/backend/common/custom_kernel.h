// Copyright © 2026 Apple Inc.

#pragma once

#include <mutex>
#include <unordered_map>
#include <variant>

#include "mlx/dtype.h"

namespace mlx::core::fast {

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

enum class MathMode {
  Safe = 0,
  Relaxed = 1,
  Fast = 2,
};

struct CompileOptions {
  MathMode math_mode = MathMode::Safe;

  CompileOptions() = default;
  bool operator==(const CompileOptions&) const = default;

  // A simple way to make export work, needs more work when adding new options.
  using Data = int;
  CompileOptions(Data data) : math_mode(static_cast<MathMode>(data)) {}
  Data serialize() const {
    return static_cast<int>(math_mode);
  }
};

std::string template_arguments_hash(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args);

class GeneratedSourceCache {
 public:
  template <typename F>
  const auto& get(const std::string& name, F&& make) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(name);
    if (it != cache_.end()) {
      return it->second;
    }
    auto source = make();
    auto hash = std::hash<std::string>{}(source);
    return cache_.emplace(name, std::make_pair(std::move(source), hash))
        .first->second;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::pair<std::string, size_t>> cache_;
};

} // namespace mlx::core::fast
