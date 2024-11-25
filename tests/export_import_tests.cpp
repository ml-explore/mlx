// Copyright © 2023 Apple Inc.

#include <filesystem>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/compile.h"
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
    return {abs(negative(exp(x[0])))};
  };

  export_function(file_path, fun, {array({1.0, 2.0})});

  auto imported_fun = import_function(file_path);
}
