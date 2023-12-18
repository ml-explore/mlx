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

TEST_CASE("test tokenizer") {
  auto raw = std::string(" { \"testing\": [1 , \"test\"]}   ");
  auto tokenizer = io::Tokenizer(raw.c_str(), raw.size());
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::CURLY_OPEN);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::STRING);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::COLON);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::ARRAY_OPEN);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::NUMBER);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::COMMA);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::STRING);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::ARRAY_CLOSE);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::CURLY_CLOSE);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::NULL_TYPE);

  raw = std::string(" { \"testing\": \"test\"}   ");
  tokenizer = io::Tokenizer(raw.c_str(), raw.size());
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::CURLY_OPEN);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::STRING);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::COLON);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::STRING);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::CURLY_CLOSE);
  CHECK_EQ(tokenizer.getToken().type, io::TOKEN::NULL_TYPE);
}

TEST_CASE("test parseJson") {
  auto raw = std::string("{}");
  auto res = io::parseJson(raw.c_str(), raw.size());
  CHECK(res.is_type(io::JSONNode::Type::OBJECT));

  raw = std::string("[]");
  res = io::parseJson(raw.c_str(), raw.size());
  CHECK(res.is_type(io::JSONNode::Type::LIST));

  raw = std::string("[");
  CHECK_THROWS_AS(io::parseJson(raw.c_str(), raw.size()), std::runtime_error);

  raw = std::string("[{}, \"test\"]");
  res = io::parseJson(raw.c_str(), raw.size());
  CHECK(res.is_type(io::JSONNode::Type::LIST));
  CHECK_EQ(res.getList()->size(), 2);
  CHECK(res.getList()->at(0)->is_type(io::JSONNode::Type::OBJECT));
  CHECK(res.getList()->at(1)->is_type(io::JSONNode::Type::STRING));
  CHECK_EQ(res.getList()->at(1)->getString(), "test");

  raw = std::string(
      "{\"test\":{\"dtype\":\"F32\",\"shape\":[4], \"data_offsets\":[0, 16]}}");
  res = io::parseJson(raw.c_str(), raw.size());
  CHECK(res.is_type(io::JSONNode::Type::OBJECT));
  CHECK_EQ(res.getObject()->size(), 1);
  CHECK(res.getObject()->at("test")->is_type(io::JSONNode::Type::OBJECT));
  CHECK_EQ(res.getObject()->at("test")->getObject()->size(), 3);
  CHECK(res.getObject()->at("test")->getObject()->at("dtype")->is_type(
      io::JSONNode::Type::STRING));
  CHECK_EQ(
      res.getObject()->at("test")->getObject()->at("dtype")->getString(),
      "F32");
  CHECK(res.getObject()->at("test")->getObject()->at("shape")->is_type(
      io::JSONNode::Type::LIST));
  CHECK_EQ(
      res.getObject()->at("test")->getObject()->at("shape")->getList()->size(),
      1);
  CHECK(res.getObject()
            ->at("test")
            ->getObject()
            ->at("shape")
            ->getList()
            ->at(0)
            ->is_type(io::JSONNode::Type::NUMBER));
  CHECK_EQ(
      res.getObject()
          ->at("test")
          ->getObject()
          ->at("shape")
          ->getList()
          ->at(0)
          ->getNumber(),
      4);
  CHECK(res.getObject()
            ->at("test")
            ->getObject()
            ->at("data_offsets")
            ->is_type(io::JSONNode::Type::LIST));
  CHECK_EQ(
      res.getObject()
          ->at("test")
          ->getObject()
          ->at("data_offsets")
          ->getList()
          ->size(),
      2);
  CHECK(res.getObject()
            ->at("test")
            ->getObject()
            ->at("data_offsets")
            ->getList()
            ->at(0)
            ->is_type(io::JSONNode::Type::NUMBER));
  CHECK_EQ(
      res.getObject()
          ->at("test")
          ->getObject()
          ->at("data_offsets")
          ->getList()
          ->at(0)
          ->getNumber(),
      0);
  CHECK(res.getObject()
            ->at("test")
            ->getObject()
            ->at("data_offsets")
            ->getList()
            ->at(1)
            ->is_type(io::JSONNode::Type::NUMBER));
  CHECK_EQ(
      res.getObject()
          ->at("test")
          ->getObject()
          ->at("data_offsets")
          ->getList()
          ->at(1)
          ->getNumber(),
      16);
}

TEST_CASE("test load_safetensor") {
  auto safeDict = load_safetensor("../../temp.safe");
  CHECK_EQ(safeDict.size(), 1);
  CHECK_EQ(safeDict.count("test"), 1);
  array test = safeDict.at("test");
  CHECK_EQ(test.dtype(), float32);
  CHECK_EQ(test.shape(), std::vector<int>({4}));
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
