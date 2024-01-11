// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

void test_arg_reduce_small(
    Device d,
    const array& x,
    ArgReduce::ReduceType r,
    std::vector<int> out_shape,
    int axis,
    std::vector<int> expected_output) {
  auto s = default_stream(d);
  auto y =
      array(out_shape, uint32, std::make_unique<ArgReduce>(s, r, axis), {x});
  y.eval();
  const uint32_t* ydata = y.data<uint32_t>();
  for (int i = 0; i < y.size(); i++) {
    CHECK_EQ(expected_output[i], ydata[i]);
  }
}

void test_arg_reduce_against_cpu(
    const array& x,
    ArgReduce::ReduceType r,
    std::vector<int> out_shape,
    int axis) {
  auto y1 = array(
      out_shape,
      uint32,
      std::make_unique<ArgReduce>(default_stream(Device::cpu), r, axis),
      {x});
  auto y2 = array(
      out_shape,
      uint32,
      std::make_unique<ArgReduce>(default_stream(Device::gpu), r, axis),
      {x});
  y1.eval();
  y2.eval();
  CHECK(array_equal(y1, y2).item<bool>());
}

TEST_CASE("test arg reduce small") {
  auto x = array(
      {0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5,
       0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5},
      {2, 3, 4});
  x.eval();
  test_arg_reduce_small(
      Device::cpu, x, ArgReduce::ArgMin, {2, 3}, 2, {0, 1, 3, 0, 1, 3});
  test_arg_reduce_small(
      Device::cpu, x, ArgReduce::ArgMin, {2, 4}, 1, {0, 1, 1, 2, 0, 1, 1, 2});
  test_arg_reduce_small(
      Device::cpu,
      x,
      ArgReduce::ArgMin,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test_arg_reduce_small(
      Device::cpu, x, ArgReduce::ArgMax, {2, 3}, 2, {3, 0, 1, 3, 0, 1});
  test_arg_reduce_small(
      Device::cpu, x, ArgReduce::ArgMax, {2, 4}, 1, {1, 2, 2, 0, 1, 2, 2, 0});
  test_arg_reduce_small(
      Device::cpu,
      x,
      ArgReduce::ArgMax,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  if (!metal::is_available()) {
    INFO("Skipping arg reduction gpu tests");
    return;
  }

  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMin, {2, 3}, 2, {0, 1, 3, 0, 1, 3});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMin, {2, 4}, 1, {0, 1, 1, 2, 0, 1, 1, 2});
  test_arg_reduce_small(
      Device::gpu,
      x,
      ArgReduce::ArgMin,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMax, {2, 3}, 2, {3, 0, 1, 3, 0, 1});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMax, {2, 4}, 1, {1, 2, 2, 0, 1, 2, 2, 0});
  test_arg_reduce_small(
      Device::gpu,
      x,
      ArgReduce::ArgMax,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

TEST_CASE("test arg reduce against cpu") {
  if (!metal::is_available()) {
    INFO("Skipping arg reduction gpu tests");
    return;
  }

  auto x = random::uniform(array(0.0), array(1.0), {127, 92, 55});
  x.eval();
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMin, {127, 92}, 2);
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMin, {127, 55}, 1);
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMin, {92, 55}, 0);
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMax, {127, 92}, 2);
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMax, {127, 55}, 1);
  test_arg_reduce_against_cpu(x, ArgReduce::ArgMax, {92, 55}, 0);

  auto y = random::uniform(array(0.0), array(1.0), {1234});
  y.eval();
  test_arg_reduce_against_cpu(y, ArgReduce::ArgMin, {}, 0);
  test_arg_reduce_against_cpu(y, ArgReduce::ArgMax, {}, 0);
}

void test_arg_reduce_small_bool(
    Device d,
    ArgReduce::ReduceType r,
    std::vector<int> out_shape,
    int axis,
    std::vector<int> expected_output) {
  auto s = default_stream(d);
  auto x = array(
      {0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5,
       0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5},
      {2, 3, 4});
  x.eval();
  auto y =
      array(out_shape, uint32, std::make_unique<ArgReduce>(s, r, axis), {x});
  y.eval();
  const uint32_t* ydata = y.data<uint32_t>();
  for (int i = 0; i < y.size(); i++) {
    CHECK_EQ(expected_output[i], ydata[i]);
  }
}

TEST_CASE("test arg reduce bool") {
  if (!metal::is_available()) {
    INFO("Skipping arg reduction gpu tests");
    return;
  }
  auto x = array(
      {false, true,  true,  false, false, false, false, true,
       true,  false, true,  true,  false, true,  true,  false,
       false, false, false, true,  true,  false, true,  true},
      {2, 3, 4});
  x.eval();
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMin, {2, 3}, 2, {0, 0, 1, 0, 0, 1});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMin, {2, 4}, 1, {0, 1, 1, 0, 0, 1, 1, 0});
  test_arg_reduce_small(
      Device::gpu,
      x,
      ArgReduce::ArgMin,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMax, {2, 3}, 2, {1, 3, 0, 1, 3, 0});
  test_arg_reduce_small(
      Device::gpu, x, ArgReduce::ArgMax, {2, 4}, 1, {2, 0, 0, 1, 2, 0, 0, 1});
  test_arg_reduce_small(
      Device::gpu,
      x,
      ArgReduce::ArgMax,
      {3, 4},
      0,
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

TEST_CASE("test arg reduce edge cases") {
  auto a = argmin(array(1.0));
  CHECK_EQ(a.item<uint32_t>(), 0);
  auto b = argmax(array(1.0));
  CHECK_EQ(b.item<uint32_t>(), 0);
  CHECK_THROWS(argmin(array({})));
  CHECK_THROWS(argmax(array({})));
}

TEST_CASE("test arg reduce irregular strides") {
  auto x = array(
      {0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5,
       0, 2, 1, 7, 5, -5, 0, 2, 1, 7, 5, -5},
      {2, 3, 4});
  x = transpose(x, {2, 0, 1});
  x.eval();
  test_arg_reduce_small(
      Device::cpu, x, ArgReduce::ArgMin, {4, 2}, 2, {0, 0, 1, 1, 1, 1, 2, 2});

  if (!metal::is_available()) {
    INFO("Skipping arg reduction gpu tests");
    return;
  }
}
