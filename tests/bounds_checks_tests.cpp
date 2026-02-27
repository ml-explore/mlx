// Copyright © 2024 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test bounds checks gather") {
  if (!metal::is_available()) {
    return;
  }

  auto stream = default_stream(Device::gpu);
  reset_global_failure();

  // 1. 1D Array failure (gather_front kernel)
  {
    auto src = array({10, 20, 30, 40, 50}, {5});
    auto bad_indices = array({0, 2, 5}, {3});
    auto bad_op = gather(src, bad_indices, 0, {1});
    CHECK_THROWS_AS(eval(bad_op), std::out_of_range);
    reset_global_failure();
  }

  // 2. 2D Array failure (general gather kernel)
  {
    auto src_2d = array({1, 2, 3, 4, 5, 6}, {2, 3});
    auto bad_indices = array({0, 30}, {2});
    auto bad_op = gather(src_2d, bad_indices, 0, {1, 2});
    CHECK_THROWS_AS(eval(bad_op), std::out_of_range);
    reset_global_failure();
  }

  // 3. Valid 2D gather
  {
    auto src_2d = array({1, 2, 3, 4, 5, 6}, {2, 3});
    auto valid_indices = array({0, 1, 0}, {3});
    auto op = gather(src_2d, valid_indices, 0, {1, 3});
    eval(op);
    CHECK(op.is_available());
  }
}

TEST_CASE("test bounds checks dependent op failure propagation") {
  if (!metal::is_available()) {
    return;
  }

  auto stream = default_stream(Device::gpu);
  reset_global_failure();

  auto src = array({1, 2, 3, 4, 5}, {5});

  auto bad_indices = array({100}, {1});
  auto fail_op = gather(src, bad_indices, 0, {3});
  auto next_op = gather(fail_op, array({0}, {1}), 0, {1, 3});

  CHECK_THROWS_AS(eval(next_op), std::out_of_range);

  reset_global_failure();
  auto valid_op = gather(src, array({0}, {1}), 0, {3});
  eval(valid_op);
  CHECK(valid_op.is_available());
}

TEST_CASE("test bounds checks scatter") {
  if (!metal::is_available()) {
    return;
  }

  auto stream = default_stream(Device::gpu);
  reset_global_failure();

  auto dst = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5});

  // 1. 1D Scatter failure
  {
    auto indices = array({100}, {1});
    auto updates = array({10.0f}, {1, 1});
    auto bad_op = scatter(dst, indices, updates, 0);
    CHECK_THROWS_AS(eval(bad_op), std::out_of_range);
    reset_global_failure();
  }

  // 2. 2D Scatter failure
  {
    auto dst_2d = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    auto indices = std::vector<array>{array({0, 10}, {2})};
    auto updates = reshape(
        array({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}), {2, 1, 3});
    auto bad_op = scatter(dst_2d, indices, updates, {0});
    CHECK_THROWS_AS(eval(bad_op), std::out_of_range);
    reset_global_failure();
  }

  // 3. Valid scatter
  {
    auto updates = reshape(array({10.0f, 20.0f}, {2}), {2, 1});
    auto indices = array({0, 4}, {2});
    auto op = scatter(dst, indices, updates, 0);
    eval(op);
    CHECK(op.is_available());
    auto expected = std::vector<float>{10.0f, 2.0f, 3.0f, 4.0f, 20.0f};
    CHECK(array_equal(op, array(expected.data(), {5}, float32)).item<bool>());
  }

  // 4. scatter_add failure
  {
    auto dst = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5});
    auto indices = array({100}, {1});
    auto updates = array({1.0f}, {1, 1});
    auto bad_op = scatter_add(dst, indices, updates, 0);
    CHECK_THROWS_AS(eval(bad_op), std::out_of_range);
    reset_global_failure();
  }
}
