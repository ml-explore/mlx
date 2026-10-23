// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <string>

#include "mlx/config.h"

using namespace mlx::core;

namespace {

std::string handler_name;
int handler_default;

int test_config_handler(const char* name, int default_val) {
  handler_name = name;
  handler_default = default_val;
  return 42;
}

class ConfigHandlerReset {
 public:
  ~ConfigHandlerReset() {
    config::register_config_handler(nullptr);
  }
};

} // namespace

TEST_CASE("test scoped config update") {
  constexpr auto name = "MLX_TEST_SCOPED_CONFIG_UPDATE";
  config::update(name, 1);
  CHECK_EQ(config::get(name, 0), 1);

  {
    auto outer = config::scoped_update({{name, 2}});
    CHECK_EQ(config::get(name, 0), 2);

    {
      auto inner = config::scoped_update({{name, 3}});
      CHECK_EQ(config::get(name, 0), 3);
    }

    CHECK_EQ(config::get(name, 0), 2);
  }

  CHECK_EQ(config::get(name, 0), 1);
}

TEST_CASE("test config handler") {
  ConfigHandlerReset reset;
  config::update("MLX_TEST_CONFIG_HANDLER", 1);
  config::register_config_handler(test_config_handler);

  CHECK_EQ(config::get("MLX_TEST_CONFIG_HANDLER", 7), 42);
  CHECK_EQ(handler_name, "MLX_TEST_CONFIG_HANDLER");
  CHECK_EQ(handler_default, 7);
}
