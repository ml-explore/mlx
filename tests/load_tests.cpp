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

TEST_CASE("test save_safetensor") {
  std::string file_path = get_temp_file("test_arr.safetensors");
  auto map = std::unordered_map<std::string, array>();
  map.insert({"test", array({1.0, 2.0, 3.0, 4.0})});
  map.insert({"test2", ones({2, 2})});
  save_safetensor(file_path, map);
  auto safeDict = load_safetensor(file_path);
  CHECK_EQ(safeDict.size(), 2);
  CHECK_EQ(safeDict.count("test"), 1);
  CHECK_EQ(safeDict.count("test2"), 1);
  array test = safeDict.at("test");
  CHECK_EQ(test.dtype(), float32);
  CHECK_EQ(test.shape(), std::vector<int>({4}));
  CHECK(array_equal(test, array({1.0, 2.0, 3.0, 4.0})).item<bool>());
  array test2 = safeDict.at("test2");
  MESSAGE("test2: " << test2);
  CHECK_EQ(test2.dtype(), float32);
  CHECK_EQ(test2.shape(), std::vector<int>({2, 2}));
  CHECK(array_equal(test2, ones({2, 2})).item<bool>());
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
