// Copyright © 2023-2024 Apple Inc.

#include <array>
#include "doctest/doctest.h"

#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/mlx.h"

using namespace mlx::core;

static const std::array<Dtype, 5> types =
    {bool_, uint32, int32, int64, float32};

TEST_CASE("test metal device") {
  // Make sure the device and library can load
  CHECK(metal::is_available());
  auto& device = metal::device(Device::gpu);
}

TEST_CASE("test metal arange") {
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto out_cpu = arange(1, 100, 2, t, Device::cpu);
    auto out_gpu = arange(1, 100, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

    out_cpu = arange(1, 5, 0.25, t, Device::cpu);
    out_gpu = arange(1, 5, 0.25, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test metal full") {
  for (auto t : types) {
    auto out_cpu = full({4, 4}, 2, t, Device::cpu);
    auto out_gpu = full({4, 4}, 2, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check broadcasting works
  {
    auto x = full({2, 2}, array({3, 4}, {2, 1}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 3, 4, 4}, {2, 2}), Device::cpu).item<bool>());
    x = full({2, 2}, array({3, 4}, {1, 2}), Device::gpu);
    CHECK(
        array_equal(x, array({3, 4, 3, 4}, {2, 2}), Device::cpu).item<bool>());
  }

  // Check zeros and ones
  {
    auto x = zeros({2, 2}, float32, Device::gpu);
    auto y = array({0.0, 0.0, 0.0, 0.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());

    x = ones({2, 2}, float32, Device::gpu);
    y = array({1.0, 1.0, 1.0, 1.0}, {2, 2});
    CHECK(array_equal(x, y, Device::cpu).item<bool>());
  }
}

TEST_CASE("test metal astype") {
  array x = array({-4, -3, -2, -1, 0, 1, 2, 3});
  // Check all types work
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  for (auto t : types) {
    auto out_cpu = astype(x, t, Device::cpu);
    auto out_gpu = astype(x, t, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }
}

TEST_CASE("test metal reshape") {
  array x = array({0, 1, 2, 3, 4, 5, 6, 7});
  auto out_cpu = reshape(x, {2, 2, 2});
  auto out_gpu = reshape(x, {2, 2, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  x = transpose(reshape(x, {2, 2, 2}), {1, 2, 0});
  out_cpu = reshape(x, {4, 2});
  out_gpu = reshape(x, {4, 2}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());

  out_cpu = reshape(x, {8});
  out_gpu = reshape(x, {8}, Device::gpu);
  CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
}

TEST_CASE("test metal reduce") {
  {
    array a(true);
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);

    a = array(std::initializer_list<bool>{});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 1);
    array a(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), true);

    vals[32] = 0;
    a = array(vals.data(), {33});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
  }

  {
    std::vector<int> vals(33, 0);
    array a(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[32] = 1;
    a = array(vals.data(), {33});
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  {
    std::vector<int> vals(1 << 14, 0);
    array a(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), false);

    vals[4] = 1;
    vals[999] = 1;
    vals[2000] = 1;
    a = array(vals.data(), {1 << 14});
    CHECK_EQ(all(a, Device::gpu).item<bool>(), false);
    CHECK_EQ(any(a, Device::gpu).item<bool>(), true);
  }

  // sum and prod
  {
    array a = array({true, false, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), false);

    a = array({true, true, true});
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 3);
    CHECK_EQ(prod(a, Device::gpu).item<bool>(), true);

    a = full({2, 2, 2}, 2.0f);
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 16.0f);
    CHECK_EQ(prod(a, Device::gpu).item<float>(), 256.0f);

    a = full({500, 2, 2}, 1u);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1u);

    a = full({500, 2, 2}, 1);
    CHECK_EQ(sum(a, Device::gpu).item<int32_t>(), 2000);
    CHECK_EQ(prod(a, Device::gpu).item<int32_t>(), 1);
  }

  // reducing only some axes and irregular layouts
  {
    array a(1.0f);
    a = broadcast_to(a, {2, 2, 2});
    CHECK_EQ(sum(a, Device::gpu).item<float>(), 8.0f);

    a = ones({2, 4, 8, 16});
    for (auto ax : {0, 1, 2, 3}) {
      auto out_gpu = sum(a, ax, false, Device::gpu);
      auto out_cpu = sum(a, ax, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }

    for (auto ax : {1, 2, 3}) {
      auto out_gpu = sum(a, {0, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }
    for (auto ax : {2, 3}) {
      auto out_gpu = sum(a, {0, 1, ax}, false, Device::gpu);
      auto out_cpu = sum(a, {0, 1, ax}, false, Device::cpu);
      CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    }
  }
}

TEST_CASE("test metal binary ops") {
  // scalar-scalar
  {
    array a(2.0f);
    array b(4.0f);
    auto out = add(a, b, Device::gpu);
    CHECK_EQ(out.item<float>(), 6.0f);
  }

  // scalar-vector and vector-scalar
  {
    array a(2.0f);
    array b({2.0f, 4.0f, 6.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({4.0f, 6.0f, 8.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
    out = add(b, a, Device::gpu);
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // vector-vector
  {
    array a({0.0f, 1.0f, 2.0f});
    array b({3.0f, 4.0f, 5.0f});
    auto out = add(a, b, Device::gpu);
    auto expected = array({3.0f, 5.0f, 7.0f});
    CHECK(array_equal(out, expected, Device::cpu).item<bool>());
  }

  // general
  {
    array a({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    array b({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, {2, 2, 2});
    a = transpose(a, {0, 2, 1});
    b = transpose(b, {1, 0, 2});
    auto out_gpu = add(a, b, Device::gpu);
    auto out_cpu = add(a, b, Device::cpu);
    auto expected =
        array({0.0f, 3.0f, 5.0f, 8.0f, 6.0f, 9.0f, 11.0f, 14.0f}, {2, 2, 2});
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
    CHECK(array_equal(out_gpu, expected, Device::cpu).item<bool>());
  }

  // Check all types work
  for (auto t : types) {
    auto a = astype(array({0, 1, 2}), t);
    auto b = astype(array({3, 4, 5}), t);
    auto out_cpu = add(a, b, Device::cpu);
    auto out_gpu = add(a, b, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Check subtraction
  {
    auto a = array({3, 2, 1});
    auto b = array({1, 1, 1});
    auto out = subtract(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 1, 0}), Device::cpu).item<bool>());
  }

  // Check multiplication
  {
    auto a = array({1, 2, 3});
    auto b = array({2, 2, 2});
    auto out = multiply(a, b, Device::gpu);
    CHECK(array_equal(out, array({2, 4, 6}), Device::cpu).item<bool>());
  }

  // Check division
  {
    auto x = array(1.0f);
    auto y = array(1.0f);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 1.0f);

    x = array(1.0f);
    y = array(0.5);
    CHECK_EQ(divide(x, y, Device::gpu).item<float>(), 2.0f);

    x = array(1.0f);
    y = array(0.0f);
    CHECK(std::isinf(divide(x, y, Device::gpu).item<float>()));

    x = array(0.0f);
    y = array(0.0f);
    CHECK(std::isnan(divide(x, y, Device::gpu).item<float>()));
  }

  // Check maximum and minimum
  {
    auto x = array(1.0f);
    auto y = array(0.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 1.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 0.0f);
    y = array(2.0f);
    CHECK_EQ(maximum(x, y, Device::gpu).item<float>(), 2.0f);
    CHECK_EQ(minimum(x, y, Device::gpu).item<float>(), 1.0f);
  }

  // Check equal
  {
    array x(1.0f);
    array y(1.0f);
    CHECK(equal(x, y, Device::gpu).item<bool>());
    x = array(0.0f);
    CHECK(!equal(x, y, Device::gpu).item<bool>());
  }

  // Greater and less
  {
    array x(1.0f);
    array y(0.0f);
    CHECK(greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());
    CHECK(!greater(y, x, Device::gpu).item<bool>());
    CHECK(!greater_equal(y, x, Device::gpu).item<bool>());
    y = array(1.0f);
    CHECK(!greater(x, y, Device::gpu).item<bool>());
    CHECK(greater_equal(x, y, Device::gpu).item<bool>());

    x = array(0.0f);
    y = array(1.0f);
    CHECK(less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
    CHECK(!less(y, x, Device::gpu).item<bool>());
    CHECK(!less_equal(y, x, Device::gpu).item<bool>());
    y = array(0.0f);
    CHECK(!less(x, y, Device::gpu).item<bool>());
    CHECK(less_equal(x, y, Device::gpu).item<bool>());
  }

  // Check logaddexp
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(inf);
    array y(2.0f);
    auto out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), inf);

    x = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), 2.0f);

    y = array(-inf);
    out = logaddexp(x, y, Device::gpu);
    CHECK_EQ(out.item<float>(), -inf);
  }
}

TEST_CASE("test metal unary ops") {
  // contiguous
  {
    array x({-1.0f, 0.0f, 1.0f});
    auto expected = array({1.0f, 0.0f, 1.0f});
    CHECK(array_equal(abs(x, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // general
  {
    array x({-1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 3.0f, -3.0f});
    auto y = slice(x, {0}, {8}, {2});
    auto expected = array({1.0f, 1.0f, 1.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());

    y = slice(x, {4}, {8});
    expected = array({1.0f, 1.0f, 3.0f, 3.0f});
    CHECK(array_equal(abs(y, Device::gpu), expected, Device::cpu).item<bool>());
  }

  // Test negative
  {
    array x(1.0f);
    CHECK_EQ(negative(x, Device::gpu).item<float>(), -1.0f);
  }

  // Check all types work
  for (auto t : types) {
    if (t == bool_) {
      continue;
    }
    auto in = astype(array({1}), t);
    auto out_cpu = negative(in, Device::cpu);
    auto out_gpu = negative(in, Device::gpu);
    CHECK(array_equal(out_gpu, out_cpu, Device::cpu).item<bool>());
  }

  // Test log1p
  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    array x(-1.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), -inf);

    x = array(0.0f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 0.0f);

    x = array(1e-9f);
    CHECK_EQ(log1p(x, Device::gpu).item<float>(), 1e-9f);

    x = array(-2.0f);
    CHECK(std::isnan(log1p(x, Device::gpu).item<float>()));
  }
}

TEST_CASE("test metal random") {
  {
    auto key = random::key(0);
    auto x = random::bits({}, 4, key, Device::gpu);
    auto y = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 1797259609u);
    CHECK_EQ(x.item<uint32_t>(), y.item<uint32_t>());
  }

  {
    auto key = random::key(1);
    auto x = random::bits({}, 4, key, Device::gpu);
    CHECK_EQ(x.item<uint32_t>(), 507451445u);
  }

  {
    auto key = random::key(0);
    auto x = random::bits({3, 1}, 4, key, Device::gpu);
    auto expected = array({4146024105u, 1351547692u, 2718843009u}, {3, 1});
    CHECK(array_equal(x, expected, Device::cpu).item<bool>());
  }
}

TEST_CASE("test metal matmul") {
  {
    auto a = ones({2, 2});
    auto b = ones({2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Batched matmul
  {
    auto a = ones({3, 2, 2});
    auto b = ones({3, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }

  // Broadcast batched matmul
  {
    auto a = ones({1, 3, 2, 2});
    auto b = ones({3, 1, 2, 2});
    auto out = matmul(a, b, Device::gpu);
    CHECK(array_equal(out, full({3, 3, 2, 2}, 2.0f), Device::cpu).item<bool>());
  }
}

TEST_CASE("test metal validation") {
  // Run this test with Metal validation enabled
  // METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0 ./tests/tests \
  //     -tc="test metal validation" \

  auto x = array({});
  eval(exp(x));

  auto y = array({});
  eval(add(x, y));

  eval(sum(x));

  x = array({1, 2, 3});
  y = array(0);
  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(gather(x, y, 0, {0}));
  eval(gather(x, y, 0, {2}));

  eval(scatter(x, y, array({2}), 0));

  x = arange(0, -3, 1);
  eval(x);
  array_equal(x, array({})).item<bool>();

  x = array({1.0, 0.0});
  eval(argmax(x));

  eval(scatter_max(array(1), {}, array(2), std::vector<int>{}));
}

TEST_CASE("test metal memory info") {
  // Test cache limits
  {
    auto old_limit = metal::set_cache_limit(0);
    {
      auto a = zeros({4096});
      eval(a);
    }
    CHECK_EQ(metal::get_cache_memory(), 0);
    CHECK_EQ(metal::set_cache_limit(old_limit), 0);
    CHECK_EQ(metal::set_cache_limit(old_limit), old_limit);
  }

  // Test memory limits
  {
    auto old_limit = metal::set_memory_limit(10);
    CHECK_EQ(metal::set_memory_limit(old_limit), 10);
    CHECK_EQ(metal::set_memory_limit(old_limit), old_limit);
  }

  // Query active and peak memory
  {
    auto a = zeros({4096});
    eval(a);
    auto active_mem = metal::get_active_memory();
    CHECK(active_mem >= 4096 * 4);
    {
      auto b = zeros({4096});
      eval(b);
    }
    auto new_active_mem = metal::get_active_memory();
    CHECK_EQ(new_active_mem, active_mem);
    auto peak_mem = metal::get_peak_memory();
    CHECK(peak_mem >= 4096 * 8);

    auto cache_mem = metal::get_cache_memory();
    CHECK(cache_mem >= 4096 * 4);
  }
}
