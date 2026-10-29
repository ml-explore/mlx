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

  auto fun = [](const Kwargs& kwargs) -> std::vector<array> {
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

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0], Stream(1000, Device::cpu))};
  };

  export_function(file_path, fun, {array({0, 1, 2})});

  // Should make a new stream that we can run computation on
  eval(import_function(file_path)({array({0, 1, 2})}));
}

TEST_CASE("test export import with metadata") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0])};
  };

  std::string metadata = "{\"name\": \"model\", \"params\": 7000000000}";

  export_function(file_path, fun, {array({0, 1, 2})}, false, metadata);

  auto imported = import_function(file_path);
  CHECK(imported.metadata() == metadata);
  eval(imported({array({0, 1, 2})}));

  // No metadata gives an empty string
  export_function(file_path, fun, {array({0, 1, 2})});
  CHECK(import_function(file_path).metadata().empty());

  // Metadata survives the per-trace header rewrite of a multi-trace export
  {
    auto fn_exporter = exporter(file_path, fun, false, metadata);
    fn_exporter({array({0, 1})});
    fn_exporter({array({0, 1, 2})});
  }
  CHECK(import_function(file_path).metadata() == metadata);
}

TEST_CASE("test import rejects mismatched trace input arrays") {
  // .mlxfn stores trace_input_ids and trace_inputs as parallel arrays. A
  // crafted file with fewer ids than inputs makes the import loop index the
  // ids vector out of bounds (an empty ids vector even reads through a null
  // data pointer), so the counts must be validated before the first lookup.
  std::vector<char> blob;
  auto push_u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++) {
      blob.push_back((v >> (8 * i)) & 0xff);
    }
  };
  auto push_i32 = [&](int32_t v) {
    for (int i = 0; i < 4; i++) {
      blob.push_back((v >> (8 * i)) & 0xff);
    }
  };
  auto push_string = [&](const std::string& s) {
    push_u64(s.size());
    blob.insert(blob.end(), s.begin(), s.end());
  };

  // Header + one function whose trace inputs are preceded by n_ids input
  // ids. Each trace input is a scalar float32 array (shape [] + dtype).
  auto write_file = [&](const std::string& name, size_t n_ids, size_t n_inputs) {
    blob.clear();
    push_string("x"); // mlx version
    push_i32(1); // function count
    blob.push_back(0); // shapeless
    push_string(""); // metadata
    push_u64(0); // kwarg keys
    push_u64(n_ids);
    for (size_t i = 0; i < n_ids; i++) {
      push_u64(i + 1);
    }
    push_u64(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) {
      push_u64(0); // scalar shape
      push_i32(10); // dtype val float32
      blob.push_back(4); // dtype size
    }
    push_u64(0); // trace output ids
    push_u64(0); // tape size
    std::string path = get_temp_file(name);
    io::FileWriter out(path);
    out.open();
    out.write(blob.data(), blob.size());
    return path;
  };

  // Empty ids with one input crashes without the count validation
  auto path = write_file("evil_input_ids_0.mlxfn", 0, 1);
  CHECK_THROWS_AS(import_function(path), std::runtime_error);

  // More ids than inputs is likewise corrupt and must be rejected
  path = write_file("evil_input_ids_2.mlxfn", 2, 1);
  CHECK_THROWS_AS(import_function(path), std::runtime_error);
}
