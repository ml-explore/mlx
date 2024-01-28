// Copyright © 2023 Apple Inc.

#include <filesystem>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name);
}

TEST_CASE("test save_safetensors") {
  std::string file_path = get_temp_file("test_arr.safetensors");
  auto map = std::unordered_map<std::string, array>();
  map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
  map.insert({"test2", ones({2, 2})});
  save_safetensors(file_path, map);
  auto dict = load_safetensors(file_path);
  CHECK_EQ(dict.size(), 2);
  CHECK_EQ(dict.count("test"), 1);
  CHECK_EQ(dict.count("test2"), 1);
  array test = dict.at("test");
  CHECK_EQ(test.dtype(), float32);
  CHECK_EQ(test.shape(), std::vector<int>({4}));
  CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  array test2 = dict.at("test2");
  CHECK_EQ(test2.dtype(), float32);
  CHECK_EQ(test2.shape(), std::vector<int>({2, 2}));
  CHECK(array_equal(test2, ones({2, 2})).item<bool>());
}

TEST_CASE("test gguf") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  {
    // Check saving loading just arrays, no metadata
    save_gguf(file_path, original_weights);
    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 0);
    CHECK_EQ(loaded_weights.size(), 2);
    CHECK_EQ(loaded_weights.count("test"), 1);
    CHECK_EQ(loaded_weights.count("test2"), 1);
    for (auto [k, v] : loaded_weights) {
      CHECK(array_equal(v, original_weights.at(k)).item<bool>());
    }
  }

  // Test saving and loading string metadata
  std::unordered_map<std::string, MetaData> original_metadata;
  original_metadata.insert({"test_str", "my string"});

  save_gguf(file_path, original_weights, original_metadata);
  auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
  CHECK_EQ(loaded_metadata.size(), 1);
  CHECK_EQ(loaded_metadata.count("test_str"), 1);
  CHECK_EQ(std::get<std::string>(loaded_metadata.at("test_str")), "my string");

  CHECK_EQ(loaded_weights.size(), 2);
  CHECK_EQ(loaded_weights.count("test"), 1);
  CHECK_EQ(loaded_weights.count("test2"), 1);
  for (auto [k, v] : loaded_weights) {
    CHECK(array_equal(v, original_weights.at(k)).item<bool>());
  }

  std::vector<Dtype> unsupported_types = {
      bool_, uint8, uint32, uint64, int64, bfloat16, complex64};
  for (auto t : unsupported_types) {
    dict to_save = {{"test", astype(arange(5), t)}};
    CHECK_THROWS(save_gguf(file_path, to_save, original_metadata));
  }

  std::vector<Dtype> supported_types = {int8, int32, float16, float32};
  for (auto t : supported_types) {
    auto arr = astype(arange(5), t);
    dict to_save = {{"test", arr}};
    save_gguf(file_path, to_save, original_metadata);
    const auto& [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK(array_equal(loaded_weights.at("test"), arr).item<bool>());
  }
}

TEST_CASE("test gguf metadata") {
  std::string file_path = get_temp_file("test_arr.gguf");
  using dict = std::unordered_map<std::string, array>;
  dict original_weights = {
      {"test", array({1.0f, 2.0f, 3.0f, 4.0f})},
      {"test2", reshape(arange(6), {3, 2})}};

  // Scalar array
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    original_metadata.insert({"test_arr", array(1.0)});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK_EQ(arr.item<float>(), 1.0f);
  }

  // 1D Array
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    auto arr = array({1.0, 2.0});
    original_metadata.insert({"test_arr", arr});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    auto loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());

    // Preserves dims
    arr = array({1.0});
    original_metadata["test_arr"] = arr;
    save_gguf(file_path, original_weights, original_metadata);

    std::tie(loaded_weights, loaded_metadata) = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("test_arr"), 1);

    loaded_arr = std::get<array>(loaded_metadata.at("test_arr"));
    CHECK(array_equal(arr, loaded_arr).item<bool>());
  }

  // > 1D array throws
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    original_metadata.insert({"test_arr", array({1.0}, {1, 1})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // empty array throws
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    original_metadata.insert({"test_arr", array({})});
    CHECK_THROWS(save_gguf(file_path, original_weights, original_metadata));
  }

  // vector of string
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta", data});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 1);
    CHECK_EQ(loaded_metadata.count("meta"), 1);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
  }

  // vector of string, string, scalar, and array
  {
    std::unordered_map<std::string, MetaData> original_metadata;
    std::vector<std::string> data = {"data1", "data2", "data1234"};
    original_metadata.insert({"meta1", data});
    original_metadata.insert({"meta2", array(2.5)});
    original_metadata.insert({"meta3", array({1, 2, 3})});
    original_metadata.insert({"meta4", "last"});
    save_gguf(file_path, original_weights, original_metadata);

    auto [loaded_weights, loaded_metadata] = load_gguf(file_path);
    CHECK_EQ(loaded_metadata.size(), 4);
    auto& strs = std::get<std::vector<std::string>>(loaded_metadata["meta1"]);
    CHECK_EQ(strs.size(), 3);
    for (int i = 0; i < strs.size(); ++i) {
      CHECK_EQ(strs[i], data[i]);
    }
    auto& arr = std::get<array>(loaded_metadata["meta2"]);
    CHECK_EQ(arr.item<float>(), 2.5);

    arr = std::get<array>(loaded_metadata["meta3"]);
    CHECK(array_equal(arr, array({1, 2, 3})).item<bool>());

    auto& str = std::get<std::string>(loaded_metadata["meta4"]);
    CHECK_EQ(str, "last");
  }
}

TEST_CASE("test single array serialization") {
  // Basic test
  {
    auto a = random::uniform(-5.f, 5.f, {2, 5, 12}, float32);

    std::string file_path = get_temp_file("test_arr.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  // Other shapes
  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            1,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_0.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(
        -5.f,
        5.f,
        {
            46,
        },
        float32);

    std::string file_path = get_temp_file("test_arr_1.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }

  {
    auto a = random::uniform(-5.f, 5.f, {5, 2, 1, 3, 4}, float32);

    std::string file_path = get_temp_file("test_arr_2.npy");

    save(file_path, a);
    auto b = load(file_path);

    CHECK_EQ(a.dtype(), b.dtype());
    CHECK_EQ(a.shape(), b.shape());
    CHECK(array_equal(a, b).item<bool>());
  }
}
