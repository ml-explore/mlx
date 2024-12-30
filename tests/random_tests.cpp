// Copyright © 2023 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test random key") {
  auto key = random::key(0);
  CHECK(array_equal(key, array({0, 0})).item<bool>());

  key = random::key(1);
  CHECK(array_equal(key, array({0, 1})).item<bool>());

  int64_t seed = static_cast<int64_t>(1) << 32;
  key = random::key(seed);
  CHECK(array_equal(key, array({1, 0})).item<bool>());

  key = random::key(seed + 1);
  CHECK(array_equal(key, array({1, 1})).item<bool>());
}

TEST_CASE("test global rng") {
  random::seed(4);
  auto x = random::bits({});
  auto y = random::bits({});

  random::seed(4);
  auto a = random::bits({});
  auto b = random::bits({});

  CHECK_EQ(x.item<uint32_t>(), a.item<uint32_t>());
  CHECK_EQ(y.item<uint32_t>(), b.item<uint32_t>());
}

TEST_CASE("test random split") {
  auto [key, subkey] = random::split(random::key(0));
  CHECK(array_equal(key, array({4146024105u, 967050713u})).item<bool>());
  CHECK(array_equal(subkey, array({2718843009u, 1272950319u})).item<bool>());

  auto keys = random::split(random::key(0), 3);
  auto expected = array(
      {2467461003u,
       428148500u,
       3186719485u,
       3840466878u,
       2562233961u,
       1946702221u},
      {3, 2});
  CHECK(array_equal(keys, expected).item<bool>());
}

TEST_CASE("test random bits") {
  // Test shapes, types, and sizes
  {
    auto key = random::key(0);
    auto x = random::bits({}, key);
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.dtype(), uint32);

    x = random::bits({0}, key);
    CHECK(array_equal(x, array({})).item<bool>());

    // Check wrong key type or shape
    key = array({0, 0});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0, 0}, {1, 2});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0u, 0u, 0u}, {3, 1});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0u, 0u}, {2, 1});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
  }

  // Expected bits in the following tests were generated from
  // Jax's Threefry 2x32 implementation using the following in
  // python:
  //
  // ```
  //   import jax
  //   import jax.prng
  //   shape = (SET THIS)
  //   seed = (SET THIS)
  //   width = (SET THIS)
  //   key = jax.random.PRNGKey(seed)
  //   print(jax.prng.threefry_prng_impl.random_bits(key, width, shape))

  {
    auto key = random::key(0);
    auto x = random::bits({}, key);
    auto y = random::bits({}, key);
    CHECK_EQ(x.item<uint32_t>(), 1797259609u);
    CHECK_EQ(x.item<uint32_t>(), y.item<uint32_t>());

    x = random::bits({}, 2, key);
    CHECK_EQ(x.item<uint16_t>(), 345);

    x = random::bits({}, 1, key);
    CHECK_EQ(x.item<uint8_t>(), 89);
  }

  {
    auto key = random::key(1);
    auto x = random::bits({}, key);
    CHECK_EQ(x.item<uint32_t>(), 507451445u);

    x = random::bits({}, 2, key);
    CHECK_EQ(x.item<uint16_t>(), 6197);

    x = random::bits({}, 1, key);
    CHECK_EQ(x.item<uint8_t>(), 53);

    CHECK_THROWS(random::bits({}, 0, key));
    CHECK_THROWS(random::bits({}, 5, key));
    CHECK_THROWS(random::bits({}, -1, key));
  }

  {
    auto key = random::key(0);
    auto x = random::bits({3, 1}, key);
    auto expected = array({4146024105u, 1351547692u, 2718843009u}, {3, 1});
    CHECK(array_equal(x, expected).item<bool>());

    x = random::bits({5}, 2, key);
    expected = array({20137, 63263, 64300, 20622, 16513}, uint16);
    CHECK(array_equal(x, expected).item<bool>());
    expected = array({20137, 63263, 64300, 20622, 16513, 41486}, uint16);
    x = random::bits({6}, 2, key);
    CHECK(array_equal(x, expected).item<bool>());
    expected = array({20137, 63263, 1497, 14756, 16513, 41486, 44591}, uint16);
    x = random::bits({7}, 2, key);
    CHECK(array_equal(x, expected).item<bool>());
    x = random::bits({8}, 2, key);
    expected =
        array({20137, 63263, 1497, 14756, 16513, 41486, 44591, 19423}, uint16);
    CHECK(array_equal(x, expected).item<bool>());
  }

  {
    auto key = array({0u, 0u, 1u, 1u}, {2, 2});
    auto shape = Shape{3};
    auto fn = [&shape](array k) { return random::bits(shape, k); };

    auto expected = array(
        {4146024105u,
         1351547692u,
         2718843009u,
         3725146706u,
         1802982961u,
         1349634643u},
        {2, 3});
    CHECK(array_equal(vmap(fn)(key), expected).item<bool>());
    expected = array(
        {2441914641u,
         1110694964u,
         3819641963u,
         2441914641u,
         1110694964u,
         3819641963u},
        {2, 3});
    CHECK(array_equal(vmap(fn, 1)(key), expected).item<bool>());

    // Vmap twice
    key = array(
        {0u,
         0u,
         1u,
         1u,
         2u,
         2u,

         3u,
         3u,
         4u,
         4u,
         5u,
         5u},
        {3, 2, 2});
    shape = {2};
    auto out = vmap(vmap(fn))(key);
    expected = array(
        {928981903u,
         3453687069u,
         3606183818u,
         460005496u,

         2799733733u,
         856293553u,
         4081856343u,
         3445925136u,

         2775548010u,
         1430281703u,
         305173070u,
         2615843348u},
        {3, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());

    out = vmap(vmap(fn, 1), 0)(key);
    expected = array(
        {1948878966u,
         4237131848u,
         1948878966u,
         4237131848u,

         2531170506u,
         1858648356u,
         2531170506u,
         1858648356u,

         740561898u,
         4234094099u,
         740561898u,
         4234094099u},
        {3, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }

  // Vmap smaller type
  {
    auto key = array({0u, 0u, 1u, 1u}, {2, 2});
    auto fn = [](array k) { return random::bits({5}, 2, k); };

    auto expected = array(
        {4146024105u,
         1351547692u,
         2718843009u,
         3725146706u,
         1802982961u,
         1349634643u},
        {2, 3});
    auto out = vmap(fn)(key);
    auto x1 = random::bits({5}, 2, take(key, array(0), 0));
    auto x2 = random::bits({5}, 2, take(key, array(1), 0));

    CHECK(array_equal(take(out, array(0), 0), x1).item<bool>());
    CHECK(array_equal(take(out, array(1), 0), x2).item<bool>());
  }
}

TEST_CASE("test random uniform") {
  // Test shapes, types, and sizes
  {
    auto x = random::uniform({});
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.dtype(), float32);

    x = random::uniform({}, float16);
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.dtype(), float16);

    x = random::uniform({0});
    CHECK(array_equal(x, array({})).item<bool>());

    // Non float type throws
    CHECK_THROWS_AS(random::uniform({}, int32), std::invalid_argument);

    // dtype respected
    x = random::uniform(-.1, .1, {0}, bfloat16);
    CHECK_EQ(x.dtype(), bfloat16);

    // Check broadcasting
    x = random::uniform(zeros({3, 1}), ones({1, 3}), {3, 3});
    CHECK_EQ(x.shape(), Shape{3, 3});
    CHECK_THROWS_AS(
        random::uniform(zeros({3, 3}), 1.0, {1, 3}), std::invalid_argument);
    CHECK_THROWS_AS(
        random::uniform(zeros({3, 3}), 1.0, {2, 3}), std::invalid_argument);
    CHECK_THROWS_AS(
        random::uniform(zeros({3, 1}), ones({1, 3}), {1, 3}),
        std::invalid_argument);

    // Check wrong key type or shape
    auto key = array({0, 0});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0, 0}, {1, 2});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0u, 0u, 0u}, {3, 1});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
    key = array({0u, 0u}, {2, 1});
    CHECK_THROWS_AS(random::uniform({}, key), std::invalid_argument);
  }

  // Expected bits in the following tests were generated from
  // Jax's Threefry 2x32 implementation using the following in
  // python:
  //
  // ```
  //   import jax
  //   import jax.prng
  //   shape = (SET THIS)
  //   seed = (SET THIS)
  //   key = jax.random.PRNGKey(seed)
  //   print(jax.prng.threefry_prng_impl.random_bits(key, 32, shape))

  constexpr auto to_float = [](uint32_t n) {
    return static_cast<float>(n) / UINT32_MAX;
  };

  {
    auto key = random::key(0);
    auto x = random::uniform({}, key);
    auto y = random::uniform({}, key);
    auto expected = to_float(1797259609);
    CHECK_EQ(x.item<float>(), expected);
    CHECK_EQ(x.item<float>(), y.item<float>());
  }

  {
    auto key = random::key(1);
    auto x = random::uniform({}, key);
    auto expected = to_float(507451445);
    CHECK_EQ(x.item<float>(), expected);
  }

  {
    auto key = random::key(0);
    auto x = random::uniform({3, 1}, key);
    auto expected = array(
        {to_float(4146024105), to_float(1351547692), to_float(2718843009)},
        {3, 1});
    CHECK(array_equal(x, expected).item<bool>());
  }

  // Check vmap
  {
    auto key = random::key(0);
    auto fun = [](array k, array low) {
      return random::uniform(low, 1, {3}, float32, k);
    };
    auto out = vmap(fun, -1)(key, zeros({2, 3}));
    CHECK_EQ(out.shape(), Shape{2, 3});

    key = zeros({2, 2}, uint32);
    out = vmap(fun)(key, zeros({2, 3}));
    CHECK_EQ(out.shape(), Shape{2, 3});
  }

  // Check bounds are respected
  {
    auto key = random::key(128291);
    auto out = random::uniform(array(-1.0f), array(1.0f), {100}, float32, key);
    CHECK(all(less(out, array(1.0f))).item<bool>());
    CHECK(all(greater_equal(out, array(-1.0f))).item<bool>());
  }

  // Check float16
  {
    auto key = random::key(0);
    auto out = random::uniform({100}, float16, key);
    CHECK_EQ(out.dtype(), float16);
    CHECK(all(less(out, array(1.0f))).item<bool>());
    CHECK(all(greater_equal(out, array(0.0f))).item<bool>());
    CHECK(!all(equal(out, array(0.0f))).item<bool>());
    CHECK(abs(float(mean(out).item<float16_t>()) - 0.5f) < 0.02);
  }

  {
    auto key = random::key(0);
    auto out = random::uniform({100}, bfloat16, key);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK(all(less(out, array(1.0f))).item<bool>());
    CHECK(all(greater_equal(out, array(0.0f))).item<bool>());
    CHECK(!all(equal(out, array(0.0f))).item<bool>());
    CHECK(abs(float(mean(out).item<bfloat16_t>()) - 0.5f) < 0.02);
  }
}

TEST_CASE("test random normal") {
  // Test shapes, types, and sizes
  {
    auto x = random::normal({});
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.dtype(), float32);

    x = random::uniform({0});
    CHECK(array_equal(x, array({})).item<bool>());

    // Non float type throws
    CHECK_THROWS_AS(random::normal({}, int32), std::invalid_argument);

    // Check wrong key type or shape
    auto key = array({0, 0});
    CHECK_THROWS_AS(random::normal({}, key), std::invalid_argument);
    key = array({0, 0}, {1, 2});
    CHECK_THROWS_AS(random::normal({}, key), std::invalid_argument);
    key = array({0u, 0u, 0u}, {3, 1});
    CHECK_THROWS_AS(random::normal({}, key), std::invalid_argument);
    key = array({0u, 0u}, {2, 1});
    CHECK_THROWS_AS(random::normal({}, key), std::invalid_argument);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::normal({100}, key);
    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(mean(out).item<float>()) < 0.1);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::normal({200}, float16, key);
    CHECK_EQ(out.dtype(), float16);
    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(float(mean(out).item<float16_t>())) < 0.1);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::normal({200}, bfloat16, key);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(float(mean(out).item<bfloat16_t>())) < 0.1);
  }
}

TEST_CASE("test random multivariate_normal") {
  {
    auto mean = zeros({3});
    auto cov = eye(3);
    auto x = random::multivariate_normal(mean, cov, {1000}, float32);
    CHECK_EQ(x.shape(), Shape{1000, 3});
    CHECK_EQ(x.dtype(), float32);
  }

  // Limit case
  {
    auto mean = array({0, 0});
    auto cov = array({1., -1, -.1, 1.});
    cov = reshape(cov, {2, 2});
    auto x = random::multivariate_normal(mean, cov, {1}, float32);
    CHECK_EQ(x.shape(), Shape{1, 2});
    CHECK_EQ(x.dtype(), float32);
  }

  // Check wrong shapes
  {
    auto mean = zeros({3, 1});
    auto cov = eye(3);
    CHECK_THROWS_AS(
        random::multivariate_normal(
            mean,
            cov,
            {
                1000,
            },
            float32),
        std::invalid_argument);
  }
  {
    auto mean = zeros({3});
    auto cov = zeros({1, 2, 3, 3});
    auto x = random::multivariate_normal(mean, cov, {1000, 2}, float32);
    CHECK_EQ(x.shape(), Shape{1000, 2, 3});
  }
  {
    auto mean = zeros({3});
    auto cov = eye(4);
    CHECK_THROWS_AS(
        random::multivariate_normal(mean, cov, {1000, 3}, float32),
        std::invalid_argument);
  }

  // Check wrong type
  {
    auto mean = zeros({3});
    auto cov = eye(3);
    CHECK_THROWS_AS(
        random::multivariate_normal(mean, cov, {1000, 3}, float16),
        std::invalid_argument);
  }
}

TEST_CASE("test random randint") {
  CHECK_THROWS_AS(
      random::randint(array(3), array(5), {1}, float32), std::invalid_argument);

  auto x = random::randint(0, 10, {}, uint32);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), uint32);

  x = random::randint(0, 2, {}, bool_);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), bool_);

  x = random::randint(0, 2, {}, int32);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), int32);

  x = random::randint(0, 2, {}, int64);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), int64);

  // Check all in bounds
  auto low = -10.0;
  auto high = 20.0;
  x = random::randint(low, high, {1000, 1000});
  CHECK((all(low <= x).item<bool>() && all(x < high).item<bool>()));

  // Check high < low => all equals to low
  low = 20.0;
  high = -10.0;
  x = random::randint(low, high, {3, 3});
  CHECK(all(equal(x, array(low))).item<bool>());

  // Check wrong key type or shape
  auto key = array({0, 0}, {1, 2});
  CHECK_THROWS_AS(
      random::randint(low, high, {}, float32, key), std::invalid_argument);
}

TEST_CASE("test random bernoulli") {
  auto x = random::bernoulli();

  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), bool_);

  // Bernoulli parameter can have floating point type
  x = random::bernoulli(array(0.5, float16));
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), bool_);

  CHECK_THROWS(random::bernoulli(array(1, int32)));

  // Negative numbers allowed in Jax
  x = random::bernoulli(array(-1.0));
  CHECK_FALSE(x.item<bool>());

  x = random::bernoulli(array(5.0));
  CHECK(x.item<bool>());

  // Return array with correct shape
  x = random::bernoulli(0.5, {3, 3});
  CHECK_EQ(x.shape(), Shape{3, 3});

  // Try with p = {}
  x = random::bernoulli(array({}));
  CHECK_EQ(x.size(), 0);

  // Try broadcasting
  auto p = array({0.1, 0.2, 0.3});
  p = reshape(p, {1, 3});
  x = random::bernoulli(p, {4, 3});
  CHECK_EQ(x.shape(), Shape{4, 3});

  CHECK_THROWS_AS(random::bernoulli(array({}), {3, 3}), std::invalid_argument);

  p = array({0.1, 0.2, 0.3});
  // Ask for the wrong shape => throws
  CHECK_THROWS_AS(random::bernoulli(p, {2}), std::invalid_argument);

  // Check wrong key type or shape
  auto key = array({0, 0}, {1, 2});
  CHECK_THROWS_AS(random::bernoulli(array(0.5), key), std::invalid_argument);
}

TEST_CASE("Test truncated normal") {
  auto x = random::truncated_normal(array(-2.0), array(2.0));

  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), float32);

  x = random::truncated_normal(array(-2.0), array(2.0), {}, float16);
  CHECK_EQ(x.size(), 1);
  CHECK_EQ(x.dtype(), float16);

  // Requested shape
  x = random::truncated_normal(array(-2.0), array(2.0), {3, 4});
  CHECK_EQ(x.shape(), Shape{3, 4});

  // Empty array
  x = random::truncated_normal(array({}), array({}));
  CHECK_EQ(x.size(), 0);

  // Broadcast
  auto lower = reshape(array({-2.0, -3.0}), {1, 2});
  auto higher = reshape(array({0.0, 3.0, 1.5}), {3, 1});
  x = random::truncated_normal(lower, higher);

  // All in bounds
  CHECK_EQ(x.shape(), Shape{3, 2});
  CHECK((all(x <= higher).item<bool>() && all(lower <= x).item<bool>()));

  // high < low => all equal to low
  x = random::truncated_normal(array(2.0), array(-2.0));
  CHECK(all(x == array(2.0)).item<bool>());

  // Non broadcastable => throws
  CHECK_THROWS_AS(
      random::truncated_normal(lower, higher, {4, 2}), std::invalid_argument);

  auto key = array({0, 0}, {1, 2});
  CHECK_THROWS_AS(
      random::truncated_normal(array(-2.0), array(2.0), {1, 1}, float32, key),
      std::invalid_argument);
}

TEST_CASE("test categorical") {
  auto logits = zeros({10, 20});

  using random::categorical;

  // Invalid axes
  CHECK_THROWS(categorical(logits, 2));
  CHECK_THROWS(categorical(logits, -3));

  // Invalid requested shapes
  CHECK_THROWS(categorical(logits, 1, Shape{1}));
  CHECK_THROWS(categorical(logits, 1, Shape{11}));
  CHECK_THROWS(categorical(logits, 1, {10, 1}));

  CHECK_EQ(categorical(logits, -1).shape(), Shape{10});
  CHECK_EQ(categorical(logits, 0).shape(), Shape{20});
  CHECK_EQ(categorical(logits, 1).shape(), Shape{10});

  auto out = categorical(logits);
  CHECK_EQ(out.shape(), Shape{10});
  CHECK_EQ(out.dtype(), uint32);
  CHECK(max(out).item<uint32_t>() < 20);

  out = categorical(logits, 0, {5, 20});
  CHECK_EQ(out.shape(), Shape{5, 20});
  CHECK(max(out).item<uint32_t>() < 10);

  float inf = std::numeric_limits<float>::infinity();
  logits = array({1.0f, -2.0f, inf, 4.0f, 3.0f});
  CHECK_EQ(categorical(logits).item<uint32_t>(), 2);

  logits = array({-inf, -2.0f, -inf, -inf});
  CHECK_EQ(categorical(logits).item<uint32_t>(), 1);

  logits = zeros({5, 4, 3});
  CHECK_EQ(categorical(logits, -1, 7).shape(), Shape{5, 4, 7});
  CHECK_EQ(categorical(logits, -2, 7).shape(), Shape{5, 3, 7});
  CHECK_EQ(categorical(logits, -3, 7).shape(), Shape{4, 3, 7});
}

TEST_CASE("test laplace") {
  // Test shapes, types, and sizes
  {
    auto x = random::laplace({});
    CHECK_EQ(x.size(), 1);
    CHECK_EQ(x.dtype(), float32);

    // Non float type throws
    CHECK_THROWS_AS(random::laplace({}, int32), std::invalid_argument);

    // Check wrong key type or shape
    auto key = array({0, 0});
    CHECK_THROWS_AS(random::laplace({}, key), std::invalid_argument);
    key = array({0, 0}, {1, 2});
    CHECK_THROWS_AS(random::laplace({}, key), std::invalid_argument);
    key = array({0u, 0u, 0u}, {3, 1});
    CHECK_THROWS_AS(random::laplace({}, key), std::invalid_argument);
    key = array({0u, 0u}, {2, 1});
    CHECK_THROWS_AS(random::laplace({}, key), std::invalid_argument);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::laplace({1000000}, key);
    float sample_mean = mean(out).item<float>();
    float sample_variance = var(out).item<float>();

    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(sample_mean) < 0.1);

    // Chebyshev's inequality.
    for (int k = 1; k <= 5; ++k) {
      float prob_above =
          mean(greater_equal(out, array(k * std::sqrt(sample_variance))))
              .item<float>();
      float bound = 1 / std::pow(k, 2);
      CHECK(prob_above < bound);
    }

    // Expected variance for Laplace distribution is 2*scale^2.
    float expected_variance = 2.0;
    CHECK(std::abs(sample_variance - expected_variance) < 0.01);

    // Expected kurtosis of Laplace distribution is 3.
    array fourth_pows = power(out - sample_mean, array(4));
    float sample_kurtosis =
        mean(fourth_pows).item<float>() / std::pow(sample_variance, 2) - 3;
    float expected_kurtosis = 3.0;
    CHECK(std::abs(sample_kurtosis - expected_kurtosis) < 0.1);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::laplace({10000}, float16, key);
    CHECK_EQ(out.dtype(), float16);
    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(float(mean(out).item<float16_t>())) < 0.1);
  }

  {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto key = random::key(128291);
    auto out = random::laplace({10000}, bfloat16, key);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK(all(less(abs(out), array(inf))).item<bool>());
    CHECK(abs(float(mean(out).item<bfloat16_t>())) < 0.1);
  }
}
