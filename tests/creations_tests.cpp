// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test arange") {
  // Check type is inferred correclty
  {
    auto x = arange(10);
    CHECK_EQ(x.dtype(), int32);

    x = arange(10.0);
    CHECK_EQ(x.dtype(), float32);

    x = arange(10, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(10, float16);
    CHECK_EQ(x.dtype(), float16);

    x = arange(10, bfloat16);
    CHECK_EQ(x.dtype(), bfloat16);

    x = arange(10.0, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0, 10);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0.0, 10.0, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(0.0, 10.0);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0, 10, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0, 10, 0.1, float32);
    CHECK_EQ(x.dtype(), float32);

    x = arange(0.0, 10.0, 0.5, int32);
    CHECK_EQ(x.dtype(), int32);

    x = arange(10.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    x = arange(0.0, 10.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    x = arange(0.0, 10.0, 0.5, uint32);
    CHECK_EQ(x.dtype(), uint32);

    // arange unsupported for bool_
    CHECK_THROWS_AS(arange(10, bool_), std::invalid_argument);
  }

  // Check correct sizes
  {
    auto x = arange(10);
    CHECK_EQ(x.size(), 10);

    x = arange(0.0, 10.0, 0.5);
    CHECK_EQ(x.size(), 20);

    x = arange(0.0, 10.0, 0.45);
    CHECK_EQ(x.size(), 23);

    x = arange(0, 10, 10);
    CHECK_EQ(x.size(), 1);

    x = arange(0, 10, 9);
    CHECK_EQ(x.size(), 2);

    x = arange(0, 10, 100);
    CHECK_EQ(x.size(), 1);

    x = arange(0, -10, 1);
    CHECK_EQ(x.size(), 0);

    x = arange(0, -10, -1);
    CHECK_EQ(x.size(), 10);

    x = arange(0, -10, -10);
    CHECK_EQ(x.size(), 1);
  }

  // Check values
  {
    auto x = arange(0, 3);
    CHECK(array_equal(x, array({0, 1, 2})).item<bool>());

    x = arange(0, 3, 2);
    CHECK(array_equal(x, array({0, 2})).item<bool>());

    x = arange(0, 3, 3);
    CHECK(array_equal(x, array({0})).item<bool>());

    x = arange(0, -3, 1);
    CHECK(array_equal(x, array({})).item<bool>());

    x = arange(0, 3, -1);
    CHECK(array_equal(x, array({})).item<bool>());

    x = arange(0, -3, -1);
    CHECK(array_equal(x, array({0, -1, -2})).item<bool>());

    x = arange(0.0, 5.0, 0.5, int32);
    CHECK(array_equal(x, zeros({10})).item<bool>());

    x = arange(0.0, 5.0, 1.5, int32);
    CHECK(array_equal(x, array({0, 1, 2, 3})).item<bool>());

    x = arange(0.0, 5.0, 1.0, float16);
    CHECK(array_equal(x, array({0, 1, 2, 3, 4}, float16)).item<bool>());

    x = arange(0.0, 5.0, 1.0, bfloat16);
    CHECK(array_equal(x, array({0, 1, 2, 3, 4}, bfloat16)).item<bool>());

    x = arange(0.0, 5.0, 1.5, bfloat16);
    CHECK(array_equal(x, array({0., 1.5, 3., 4.5}, bfloat16)).item<bool>());
  }
}

TEST_CASE("test astype") {
  // Check type conversions
  {
    auto x = array(1);
    auto y = astype(x, float32);
    CHECK_EQ(y.dtype(), float32);
    CHECK_EQ(y.item<float>(), 1.0f);

    y = astype(x, int32);
    CHECK_EQ(y.dtype(), int32);
    CHECK_EQ(y.item<int>(), 1);

    x = array(-3.0f);
    y = astype(x, int32);
    CHECK_EQ(y.dtype(), int32);
    CHECK_EQ(y.item<int>(), -3);

    y = astype(x, uint32);
    CHECK_EQ(y.dtype(), uint32);

    // Use std::copy since the result is platform dependent
    uint32_t v;
    std::copy(x.data<float>(), x.data<float>() + 1, &v);
    CHECK_EQ(y.item<uint32_t>(), v);
  }
}

TEST_CASE("test full") {
  // Check full works for different types
  {
    auto x = full({}, 0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(x.item<int>(), 0);

    x = full({}, 0.0);
    CHECK_EQ(x.dtype(), float32);
    CHECK_EQ(x.item<float>(), 0);

    x = full({}, false);
    CHECK_EQ(x.item<bool>(), false);

    x = full({}, 0, int32);
    CHECK_EQ(x.item<int>(), 0);

    x = full({}, 0, float32);
    CHECK_EQ(x.item<float>(), 0);

    x = full({1, 2}, 2, float32);
    CHECK(array_equal(x, array({2.0, 2.0}, {1, 2})).item<bool>());

    x = full({2, 1}, 2, float32);
    CHECK(array_equal(x, array({2.0, 2.0}, {2, 1})).item<bool>());

    x = full({2}, false);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({false, false})).item<bool>());

    x = full({2}, 1.0, bool_);
    CHECK_EQ(x.dtype(), bool_);
    CHECK(array_equal(x, array({true, true})).item<bool>());

    x = full({2}, 1.0, uint32);
    CHECK_EQ(x.dtype(), uint32);
    CHECK(array_equal(x, array({1, 1})).item<bool>());

    CHECK_THROWS_AS(full({2}, array({})), std::invalid_argument);
  }

  // Check broadcasting works
  {
    auto x = full({2, 2}, array({3, 4}, {2, 1}));
    CHECK(array_equal(x, array({3, 3, 4, 4}, {2, 2})).item<bool>());
    x = full({2, 2}, array({3, 4}, {1, 2}));
    CHECK(array_equal(x, array({3, 4, 3, 4}, {2, 2})).item<bool>());
  }

  // Check zeros and ones
  {
    auto x = zeros({2, 2}, float32);
    CHECK_EQ(x.shape(), std::vector<int>{2, 2});
    CHECK_EQ(x.ndim(), 2);
    CHECK_EQ(x.dtype(), float32);
    auto y = array({0.0, 0.0, 0.0, 0.0}, {2, 2});
    CHECK(array_equal(x, y).item<bool>());

    x = ones({2, 2}, float32);
    CHECK_EQ(x.shape(), std::vector<int>{2, 2});
    CHECK_EQ(x.ndim(), 2);
    CHECK_EQ(x.dtype(), float32);
    y = array({1.0, 1.0, 1.0, 1.0}, {2, 2});
    CHECK(array_equal(x, y).item<bool>());

    x = zeros({2, 2}, int32);
    y = zeros_like(x);
    CHECK_EQ(y.dtype(), int32);
    CHECK(array_equal(x, y).item<bool>());

    x = ones({2, 2}, int32);
    y = ones_like(x);
    CHECK_EQ(y.dtype(), int32);
    CHECK(array_equal(x, y).item<bool>());
  }

  // Works for empty shape and empty array
  {
    array x = ones({}, int32);
    CHECK_EQ(x.shape(), std::vector<int>{});
    CHECK_EQ(x.item<int>(), 1);

    x = full({0}, array({}));
    CHECK_EQ(x.shape(), std::vector<int>{0});
    CHECK_EQ(x.size(), 0);

    CHECK_THROWS_AS(full({}, array({})), std::invalid_argument);
  }
}
