// Copyright © 2023-2024 Apple Inc.

#include <array>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

static const std::array<Dtype, 5> types =
    {bool_, uint32, int32, int64, float32};

TEST_CASE("test gpu arange") {
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

TEST_CASE("test gpu full") {
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

TEST_CASE("test gpu astype") {
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

TEST_CASE("test gpu reshape") {
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

TEST_CASE("test gpu reduce") {
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

  // sum and prod overflow
  {
    auto a = full({256, 2, 2}, 1u, uint8);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 256 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);

    a = full({65535, 2, 2}, 1u, uint16);
    CHECK_EQ(sum(a, Device::gpu).item<uint32_t>(), 65535 * 4);
    CHECK_EQ(prod(a, Device::gpu).item<uint32_t>(), 1);
  }
}

TEST_CASE("test gpu reduce with axes") {
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

TEST_CASE("test gpu binary ops") {
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

TEST_CASE("test gpu unary ops") {
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

TEST_CASE("test gpu random") {
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

TEST_CASE("test gpu matmul") {
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

TEST_CASE("test gpu validation") {
  // Run this test with Metal validation enabled
  // METAL_DEVICE_WRAPPER_TYPE=1 METAL_DEBUG_ERROR_MODE=0 ./tests/tests \
  //     -tc="test metal validation"

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

TEST_CASE("test memory info") {
  // Test cache limits
  {
    auto old_limit = set_cache_limit(0);
    {
      auto a = zeros({4096});
      eval(a);
    }
    CHECK_EQ(get_cache_memory(), 0);
    CHECK_EQ(set_cache_limit(old_limit), 0);
    CHECK_EQ(set_cache_limit(old_limit), old_limit);
  }

  // Test memory limits
  {
    auto old_limit = set_memory_limit(10);
    CHECK_EQ(set_memory_limit(old_limit), 10);
    CHECK_EQ(set_memory_limit(old_limit), old_limit);
  }

  // Query active and peak memory
  {
    auto a = zeros({4096});
    eval(a);
    synchronize();
    auto active_mem = get_active_memory();
    CHECK(active_mem >= 4096 * 4);
    {
      auto b = zeros({4096});
      eval(b);
    }
    synchronize();
    auto new_active_mem = get_active_memory();
    CHECK_EQ(new_active_mem, active_mem);
    auto peak_mem = get_peak_memory();
    CHECK(peak_mem >= 4096 * 8);

    auto cache_mem = get_cache_memory();
    CHECK(cache_mem >= 4096 * 4);
  }

  clear_cache();
  CHECK_EQ(get_cache_memory(), 0);
}

TEST_CASE("test gpu slice update with non-contiguous source") {
  // This test reproduces a bug where slice_update fails on CUDA when
  // the source array is non-contiguous (e.g., after transpose).
  // The bug manifests as "invalid resource handle" error on CUDA.

  // First test: contiguous source should work
  {
    auto source = ones({8, 2}, bfloat16);
    eval(source);
    synchronize();

    auto cache = zeros({8, 10}, bfloat16);
    eval(cache);
    synchronize();

    auto updated = slice_update(
        cache,
        source,
        {0, 0}, // start
        {8, 2}, // stop
        {1, 1} // stride
    );
    eval(updated);
    synchronize();

    CHECK_EQ(updated.shape(), Shape({8, 10}));
  }

  // Second test: transposed source (different stride pattern)
  {
    // Create source tensor: shape (2, 8)
    auto source = ones({2, 8}, bfloat16);
    eval(source);
    synchronize();

    // Transpose swaps the dimensions and strides
    auto transposed = transpose(source, {1, 0});
    eval(transposed);
    synchronize();

    // Create buffer: shape (8, 10)
    auto cache = zeros({8, 10}, bfloat16);
    eval(cache);
    synchronize();

    // Slice update with non-contiguous source
    auto updated = slice_update(
        cache,
        transposed,
        {0, 0}, // start
        {8, 2}, // stop
        {1, 1} // stride
    );
    eval(updated);
    synchronize();

    // Verify the shape is correct
    CHECK_EQ(updated.shape(), Shape({8, 10}));
  }

  // Test with float16 and ndim=4 (collapses to ndim=3)
  {
    auto source = ones({1, 2, 8, 128}, float16);
    auto transposed = transpose(source, {0, 2, 1, 3});
    eval(transposed);

    auto cache = zeros({1, 8, 256, 128}, float16);
    eval(cache);

    auto updated = slice_update(
        cache, transposed, {0, 0, 0, 0}, {1, 8, 2, 128}, {1, 1, 1, 1});
    eval(updated);

    CHECK_EQ(updated.shape(), Shape({1, 8, 256, 128}));
  }

  // Test with bfloat16 and ndim=4
  {
    auto source = ones({1, 2, 8, 128}, bfloat16);
    auto transposed = transpose(source, {0, 2, 1, 3});
    eval(transposed);

    auto cache = zeros({1, 8, 256, 128}, bfloat16);
    eval(cache);

    auto updated = slice_update(
        cache, transposed, {0, 0, 0, 0}, {1, 8, 2, 128}, {1, 1, 1, 1});
    eval(updated);

    CHECK_EQ(updated.shape(), Shape({1, 8, 256, 128}));
  }

  // Test that float32 works
  {
    auto source = ones({1, 2, 8, 128}, float32);
    auto transposed = transpose(source, {0, 2, 1, 3});
    eval(transposed);

    auto cache = zeros({1, 8, 256, 128}, float32);
    eval(cache);

    auto updated = slice_update(
        cache, transposed, {0, 0, 0, 0}, {1, 8, 2, 128}, {1, 1, 1, 1});
    eval(updated);

    CHECK_EQ(updated.shape(), Shape({1, 8, 256, 128}));
  }
}

// Comprehensive type conversion tests for all dtype pairs
// This tests the CUDA copy kernels for astype operations
TEST_CASE("test gpu astype comprehensive") {
  // All dtypes to test (excluding complex64 which has different rules)
  const std::array<Dtype, 12> all_types = {
      bool_,
      uint8,
      uint16,
      uint32,
      uint64,
      int8,
      int16,
      int32,
      int64,
      float16,
      float32,
      bfloat16};

  // Test all type pairs
  for (auto from_type : all_types) {
    // Create source array with appropriate values
    array src = [&]() -> array {
      if (from_type == bool_) {
        return array({true, false, true, false});
      } else {
        return astype(array({1, 0, 2, 3}), from_type);
      }
    }();
    eval(src);

    for (auto to_type : all_types) {
      try {
        // Test GPU astype
        auto out_gpu = astype(src, to_type, Device::gpu);
        auto out_cpu = astype(src, to_type, Device::cpu);
        eval(out_gpu);
        eval(out_cpu);

        CHECK_MESSAGE(
            array_equal(out_gpu, out_cpu, Device::cpu).item<bool>(),
            "astype from " << from_type << " to " << to_type << " failed");
      } catch (const std::exception& e) {
        MESSAGE(
            "Exception converting from " << from_type << " to " << to_type
                                         << ": " << e.what());
        CHECK_MESSAGE(
            false,
            "astype from " << from_type << " to " << to_type << " threw");
      }
    }
  }
}

// Test bool to/from all integer types specifically
// These were problematic on Windows CUDA
TEST_CASE("test gpu astype bool conversions") {
  const std::array<Dtype, 8> int_types = {
      uint8, uint16, uint32, uint64, int8, int16, int32, int64};

  // Test bool -> integer types
  {
    array bool_arr = array({true, false, true, true, false});
    eval(bool_arr);

    for (auto int_type : int_types) {
      auto out_gpu = astype(bool_arr, int_type, Device::gpu);
      auto out_cpu = astype(bool_arr, int_type, Device::cpu);
      eval(out_gpu);
      eval(out_cpu);

      CHECK_MESSAGE(
          array_equal(out_gpu, out_cpu, Device::cpu).item<bool>(),
          "bool -> " << int_type << " failed");

      // Also verify values are correct (1 for true, 0 for false)
      auto expected = array({1, 0, 1, 1, 0}, int_type);
      CHECK_MESSAGE(
          array_equal(out_gpu, expected, Device::cpu).item<bool>(),
          "bool -> " << int_type << " values incorrect");
    }
  }

  // Test integer types -> bool
  {
    for (auto int_type : int_types) {
      array int_arr = astype(array({0, 1, 2, 0, 255}), int_type);
      eval(int_arr);

      auto out_gpu = astype(int_arr, bool_, Device::gpu);
      auto out_cpu = astype(int_arr, bool_, Device::cpu);
      eval(out_gpu);
      eval(out_cpu);

      CHECK_MESSAGE(
          array_equal(out_gpu, out_cpu, Device::cpu).item<bool>(),
          int_type << " -> bool failed");

      // Verify: 0 -> false, non-zero -> true
      auto expected = array({false, true, true, false, true});
      CHECK_MESSAGE(
          array_equal(out_gpu, expected, Device::cpu).item<bool>(),
          int_type << " -> bool values incorrect");
    }
  }

  // Test bool -> float16/bfloat16
  {
    array bool_arr = array({true, false, true});
    eval(bool_arr);

    auto out_f16_gpu = astype(bool_arr, float16, Device::gpu);
    auto out_f16_cpu = astype(bool_arr, float16, Device::cpu);
    auto out_bf16_gpu = astype(bool_arr, bfloat16, Device::gpu);
    auto out_bf16_cpu = astype(bool_arr, bfloat16, Device::cpu);
    eval(out_f16_gpu);
    eval(out_f16_cpu);
    eval(out_bf16_gpu);
    eval(out_bf16_cpu);

    CHECK(array_equal(out_f16_gpu, out_f16_cpu, Device::cpu).item<bool>());
    CHECK(array_equal(out_bf16_gpu, out_bf16_cpu, Device::cpu).item<bool>());
  }

  // Test float16/bfloat16 -> bool
  {
    array f16_arr = array({0.0f, 1.0f, 0.5f}, float16);
    array bf16_arr = array({0.0f, 1.0f, 0.5f}, bfloat16);
    eval(f16_arr);
    eval(bf16_arr);

    auto out_f16_gpu = astype(f16_arr, bool_, Device::gpu);
    auto out_f16_cpu = astype(f16_arr, bool_, Device::cpu);
    auto out_bf16_gpu = astype(bf16_arr, bool_, Device::gpu);
    auto out_bf16_cpu = astype(bf16_arr, bool_, Device::cpu);
    eval(out_f16_gpu);
    eval(out_f16_cpu);
    eval(out_bf16_gpu);
    eval(out_bf16_cpu);

    CHECK(array_equal(out_f16_gpu, out_f16_cpu, Device::cpu).item<bool>());
    CHECK(array_equal(out_bf16_gpu, out_bf16_cpu, Device::cpu).item<bool>());
  }
}

// Test stack with various dtypes
// This exercises the copy_general kernels
TEST_CASE("test gpu stack dtypes") {
  const std::array<Dtype, 13> all_types = {
      bool_,
      uint8,
      uint16,
      uint32,
      uint64,
      int8,
      int16,
      int32,
      int64,
      float16,
      float32,
      bfloat16,
      complex64};

  for (auto dtype : all_types) {
    auto [x, y] = [&]() -> std::pair<array, array> {
      if (dtype == bool_) {
        return {array({true, false}), array({false, true})};
      } else if (dtype == complex64) {
        return {
            array({complex64_t{1.0f, 0.0f}, complex64_t{2.0f, 0.0f}}),
            array({complex64_t{3.0f, 0.0f}, complex64_t{4.0f, 0.0f}})};
      } else {
        return {astype(array({1, 2}), dtype), astype(array({3, 4}), dtype)};
      }
    }();
    eval(x);
    eval(y);

    // Test stack on GPU
    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_MESSAGE(
        array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>(),
        "stack failed for dtype " << dtype);

    // Test stack along axis 1
    stacked_gpu = stack({x, y}, 1, Device::gpu);
    stacked_cpu = stack({x, y}, 1, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_MESSAGE(
        array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>(),
        "stack axis=1 failed for dtype " << dtype);
  }
}

// Test concatenate with various dtypes
TEST_CASE("test gpu concatenate dtypes") {
  const std::array<Dtype, 9> test_types = {
      bool_, uint8, uint16, uint32, uint64, int8, int16, int32, int64};

  for (auto dtype : test_types) {
    auto [x, y] = [&]() -> std::pair<array, array> {
      if (dtype == bool_) {
        return {array({true, false}), array({false, true, true})};
      } else {
        return {astype(array({1, 2}), dtype), astype(array({3, 4, 5}), dtype)};
      }
    }();
    eval(x);
    eval(y);

    // Test concatenate on GPU
    auto concat_gpu = concatenate({x, y}, 0, Device::gpu);
    auto concat_cpu = concatenate({x, y}, 0, Device::cpu);
    eval(concat_gpu);
    eval(concat_cpu);

    CHECK_MESSAGE(
        array_equal(concat_gpu, concat_cpu, Device::cpu).item<bool>(),
        "concatenate failed for dtype " << dtype);
  }
}

// Test stack with mixed dtypes (type promotion)
// This tests the implicit astype during stack
TEST_CASE("test gpu stack mixed dtypes") {
  // int32 + float32 -> float32
  {
    auto x = array({1, 2}, int32);
    auto y = array({3.5f, 4.5f}, float32);
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), float32);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // bool + int16 -> int16
  {
    auto x = array({true, false});
    auto y = array({int16_t(3), int16_t(4)});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), int16);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // bool + uint16 -> uint16
  {
    auto x = array({true, false});
    auto y = array({uint16_t(3), uint16_t(4)});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), uint16);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // int64 + float16 -> float16
  {
    auto x = array({int64_t(1), int64_t(2)});
    auto y = array({3.0f, 4.0f}, float16);
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), float16);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // float32 + complex64 -> complex64
  {
    auto x = array({1.0f, 2.0f}, float32);
    auto y = array({complex64_t{3.0f, 0.0f}, complex64_t{4.0f, 0.0f}});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), complex64);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // int32 + complex64 -> complex64
  {
    auto x = array({1, 2}, int32);
    auto y = array({complex64_t{3.0f, 0.0f}, complex64_t{4.0f, 0.0f}});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), complex64);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // bool + complex64 -> complex64
  {
    auto x = array({true, false});
    auto y = array({complex64_t{3.0f, 0.0f}, complex64_t{4.0f, 0.0f}});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK_EQ(stacked_gpu.dtype(), complex64);
    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }
}

// Test 64-bit integer operations specifically
// These were problematic on Windows CUDA
TEST_CASE("test gpu 64bit integers") {
  // Test int64 stack
  {
    auto x = array({int64_t(1), int64_t(2)});
    auto y = array({int64_t(3), int64_t(4)});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // Test uint64 stack
  {
    auto x = array({uint64_t(1), uint64_t(2)});
    auto y = array({uint64_t(3), uint64_t(4)});
    eval(x);
    eval(y);

    auto stacked_gpu = stack({x, y}, 0, Device::gpu);
    auto stacked_cpu = stack({x, y}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }

  // Test int64 astype round-trips
  {
    auto x = array({int64_t(-1000000), int64_t(0), int64_t(1000000)});
    eval(x);

    // int64 -> float32 -> int64
    auto y_gpu = astype(astype(x, float32, Device::gpu), int64, Device::gpu);
    auto y_cpu = astype(astype(x, float32, Device::cpu), int64, Device::cpu);
    eval(y_gpu);
    eval(y_cpu);

    CHECK(array_equal(y_gpu, y_cpu, Device::cpu).item<bool>());
  }

  // Test uint64 large values
  {
    auto x = array({uint64_t(0), uint64_t(1), uint64_t(1000000)});
    eval(x);

    auto stacked_gpu = stack({x, x}, 0, Device::gpu);
    auto stacked_cpu = stack({x, x}, 0, Device::cpu);
    eval(stacked_gpu);
    eval(stacked_cpu);

    CHECK(array_equal(stacked_gpu, stacked_cpu, Device::cpu).item<bool>());
  }
}

// Test complex64 type conversions specifically
// complex64 has special rules for type promotion
TEST_CASE("test gpu astype complex64") {
  // All dtypes that can be converted to/from complex64
  const std::array<Dtype, 12> all_types = {
      bool_,
      uint8,
      uint16,
      uint32,
      uint64,
      int8,
      int16,
      int32,
      int64,
      float16,
      float32,
      bfloat16};

  // Test X -> complex64 for all types
  for (auto from_type : all_types) {
    array src = [&]() -> array {
      if (from_type == bool_) {
        return array({true, false, true, false});
      } else {
        return astype(array({1, 0, 2, 3}), from_type);
      }
    }();
    eval(src);

    try {
      auto out_gpu = astype(src, complex64, Device::gpu);
      auto out_cpu = astype(src, complex64, Device::cpu);
      eval(out_gpu);
      eval(out_cpu);

      bool eq = array_equal(out_gpu, out_cpu, Device::cpu).item<bool>();
      if (!eq && from_type == bool_) {
        // Debug output for bool case
        MESSAGE(
            "GPU output (first 4): ["
            << out_gpu.data<complex64_t>()[0].real() << "+"
            << out_gpu.data<complex64_t>()[0].imag() << "i, "
            << out_gpu.data<complex64_t>()[1].real() << "+"
            << out_gpu.data<complex64_t>()[1].imag() << "i]");
        MESSAGE(
            "CPU output (first 4): ["
            << out_cpu.data<complex64_t>()[0].real() << "+"
            << out_cpu.data<complex64_t>()[0].imag() << "i, "
            << out_cpu.data<complex64_t>()[1].real() << "+"
            << out_cpu.data<complex64_t>()[1].imag() << "i]");
      }
      CHECK_MESSAGE(eq, "astype from " << from_type << " to complex64 failed");
    } catch (const std::exception& e) {
      MESSAGE(
          "Exception converting from " << from_type
                                       << " to complex64: " << e.what());
      CHECK_MESSAGE(
          false, "astype from " << from_type << " to complex64 threw");
    }
  }

  // Test complex64 -> X for all types
  array c64_src = array(
      {complex64_t{1.0f, 0.0f},
       complex64_t{0.0f, 0.0f},
       complex64_t{2.0f, 0.0f},
       complex64_t{3.0f, 0.0f}});
  eval(c64_src);

  for (auto to_type : all_types) {
    try {
      auto out_gpu = astype(c64_src, to_type, Device::gpu);
      auto out_cpu = astype(c64_src, to_type, Device::cpu);
      eval(out_gpu);
      eval(out_cpu);

      CHECK_MESSAGE(
          array_equal(out_gpu, out_cpu, Device::cpu).item<bool>(),
          "astype from complex64 to " << to_type << " failed");
    } catch (const std::exception& e) {
      MESSAGE(
          "Exception converting from complex64 to " << to_type << ": "
                                                    << e.what());
      CHECK_MESSAGE(false, "astype from complex64 to " << to_type << " threw");
    }
  }
}
