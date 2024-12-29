// Copyright © 2024 Apple Inc.

#include <filesystem>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/export.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}
} // namespace

TEST_CASE("test export basic functions") {
  std::string file_path = get_temp_file("model.mlxfn");

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

TEST_CASE("test export function with no inputs") {
  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {zeros({2, 2})};
  };

  std::string file_path = get_temp_file("model.mlxfn");

  export_function(file_path, fun, {});

  auto imported_fun = import_function(file_path);

  auto expected = fun({});
  auto out = imported_fun({});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export multi output primitives") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {divmod(x[0], x[1])};
  };

  auto inputs = std::vector<array>{array({5.0, -10.0}), array({3.0, -2.0})};
  export_function(file_path, fun, inputs);

  auto imported_fun = import_function(file_path);

  auto expected = fun(inputs);
  auto out = imported_fun(inputs);
  CHECK(allclose(expected[0], out[0]).item<bool>());
  CHECK(allclose(expected[1], out[1]).item<bool>());
}

TEST_CASE("test export primitives with state") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {argpartition(x[0], 2, 0)};
  };

  auto x = array({1, 3, 2, 4, 5, 7, 6, 8}, {4, 2});
  export_function(file_path, fun, {x});

  auto imported_fun = import_function(file_path);

  auto expected = fun({x});
  auto out = imported_fun({x});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export functions with kwargs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun =
      [](const std::map<std::string, array>& kwargs) -> std::vector<array> {
    return {kwargs.at("x") + kwargs.at("y")};
  };

  export_function(file_path, fun, {{"x", array(1)}, {"y", array(2)}});
  auto fn = import_function(file_path);

  // Must use kwargs
  CHECK_THROWS(fn({array(1), array(2)}));

  // Wrong number of keys
  CHECK_THROWS(fn({{"x", array(1)}, {"y", array(2)}, {"z", array(3)}}));

  // Wrong keys
  CHECK_THROWS(fn({{"a", array(1)}, {"b", array(2)}}));

  // Works
  auto out = fn({{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
  out = fn({}, {{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
}

TEST_CASE("test export function with variable inputs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    auto out = array({1, 1, 1, 1});
    for (auto x : args) {
      out = out + x;
    }
    return {out};
  };

  {
    auto fn_exporter = exporter(file_path, fun);
    fn_exporter({array(0), array(0)});
    fn_exporter({array(0), array(0), array(0)});
  }

  auto imported_fun = import_function(file_path);

  // Call with two inputs
  auto out = imported_fun({array(1), array(2)})[0];

  CHECK(array_equal(out, array({4, 4, 4, 4})).item<bool>());

  // Call with three inputs
  out = imported_fun({array(1), array(2), array(3)})[0];
  CHECK(array_equal(out, array({7, 7, 7, 7})).item<bool>());
}

TEST_CASE("test export function on different stream") {
  std::string file_path = get_temp_file("model.mlxfn");

  // Caller is responsible for setting up streams before
  // importing functoins
  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0], Stream(1000, Device::cpu))};
  };

  export_function(file_path, fun, {array({0, 1, 2})});

  CHECK_THROWS(import_function(file_path));
}
