// Copyright © 2026 Apple Inc.

#include "mlx/config.h"

#include <mutex>
#include <shared_mutex>

namespace mlx::core::config {

namespace {

inline int get_env(const char* name, int default_val) {
  const char* str = std::getenv(name);
  if (!str) {
    return default_val;
  }
  return std::atoi(str);
}

auto& global_configs() {
  static std::tuple<ConfigDict, std::shared_mutex> configs_and_mtx;
  return configs_and_mtx;
}

auto& scoped_configs() {
  thread_local std::list<ConfigDict> configs;
  return configs;
}

auto& config_handler() {
  static ConfigHandler handler = nullptr;
  return handler;
}

} // namespace

void update(const char* name, int val) {
  auto& [configs, mtx] = global_configs();
  std::unique_lock lock(mtx);
  configs[name] = val;
}

int get(const char* name, int default_val) {
  // First pass to the hook.
  if (auto& handler = config_handler()) {
    return handler(name, default_val);
  }
  // Then try scoped configs.
  auto& list = scoped_configs();
  for (auto& configs : list) {
    if (auto it = configs.find(name); it != configs.end()) {
      return it->second;
    }
  }
  // Then try global configs.
  auto& [configs, mtx] = global_configs();
  {
    std::shared_lock lock(mtx);
    if (auto it = configs.find(name); it != configs.end()) {
      return it->second;
    }
  }
  int val = get_env(name, default_val);
  update(name, val);
  return val;
}

ScopedUpdate::ScopedUpdate(ConfigDict configs) {
  auto& list = scoped_configs();
  list.push_front(std::move(configs));
  it_ = list.begin();
}

ScopedUpdate::ScopedUpdate(const ScopedUpdate& other) {
  auto& list = scoped_configs();
  list.push_front(*other.it_);
  it_ = list.begin();
}

ScopedUpdate::~ScopedUpdate() {
  auto& list = scoped_configs();
  list.erase(it_);
}

void register_config_handler(ConfigHandler handler) {
  config_handler() = handler;
}

} // namespace mlx::core::config
