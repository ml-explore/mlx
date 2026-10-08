// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/api.h"

#include <list>
#include <string>
#include <unordered_map>

namespace mlx::core::config {

MLX_API void update(const char* name, int val);
MLX_API int get(const char* name, int default_val);

using ConfigDict = std::unordered_map<std::string, int>;

class MLX_API ScopedUpdate {
 public:
  ScopedUpdate(ConfigDict configs);
  ScopedUpdate(const ScopedUpdate& other);
  ~ScopedUpdate();

 private:
  std::list<ConfigDict>::iterator it_;
};

inline ScopedUpdate scoped_update(ConfigDict configs) {
  return ScopedUpdate(std::move(configs));
}

using ConfigHandler = int (*)(const char* name, int default_val);

MLX_API void register_config_handler(ConfigHandler handler);

} // namespace mlx::core::config
