// Copyright © 2023 Apple Inc.

#include <filesystem>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/export.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name);
}
} // namespace

TEST_CASE("test export basic functions") {
  std::string file_path = "model.mlxfn"; // get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {negative(exp(x[0]))};
  };

  export_function(file_path, fun, {array({1.0, 2.0})});

  auto imported_fun = import_function(file_path);

  // Check num inputs mismatch throws
  CHECK_THROWS_AS(
      imported_fun({array({1.0}), array({2.0})}), std::invalid_argument);

  // Check shape mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0})}), std::invalid_argument);

  // Check type mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0}, float16)}), std::invalid_argument);

  auto expected = fun({array({1.0, -1.0})});
  auto out = imported_fun({array({1.0, -1.0})});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}
