// Copyright © 2023 Apple Inc.

#pragma once

#include <chrono>
#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::random {

class KeySequence {
 public:
  explicit KeySequence(uint64_t seed);

  void seed(uint64_t seed);
  array next();

  // static default
  static KeySequence& default_() {
    static KeySequence ks(get_current_time_seed());
    return ks;
  }

 private:
  array key_;
  static uint64_t get_current_time_seed() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
  }
};

/** Get a PRNG key from a seed. */
array key(uint64_t seed);

/** Seed the default PRNG key. */
void seed(uint64_t seed);

/** Generate an array with type uint32 filled with random bits. */
array bits(
    const std::vector<int>& shape,
    int width,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array bits(
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bits(shape, 4, key, s);
}

/** Split the rng key into a pair of keys. */
std::pair<array, array> split(const array& key, StreamOrDevice s = {});

/** Split the rng key into `num` keys. */
array split(const array& key, int num, StreamOrDevice s = {});

/** Generate uniform random numbers between low and high. */
array uniform(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T, typename U>
array uniform(
    T low,
    U high,
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(array(low), array(high), shape, dtype, key, to_stream(s));
}

/** Generate uniform random numbers between 0 and 1. */
array uniform(
    const std::vector<int>& shape,
    Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array uniform(
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(shape, float32, key);
}

/** Generate samples from the standard normal distribution. */
array normal(
    const std::vector<int>& shape,
    Dtype dtype,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array normal(
    const std::vector<int>& shape,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, float32, loc, scale, key, s);
}
inline array normal(
    const std::vector<int>& shape,
    const Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, dtype, 0.0, 1.0, key, s);
}
inline array normal(
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, float32, 0.0, 1.0, key, s);
}

/** Generate integer samples uniformly at random */
array randint(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype = int32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T, typename U>
array randint(
    T low,
    U high,
    const std::vector<int>& shape,
    Dtype dtype = int32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return randint(array(low), array(high), shape, dtype, key, to_stream(s));
};

/** Generate binary variables with probability to be true equal to p */
array bernoulli(
    const array& p,
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
array bernoulli(
    const array& p,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T>
array bernoulli(
    T p,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bernoulli(array(p), key, s);
};

template <typename T>
array bernoulli(
    T p,
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bernoulli(array(p), shape, key, s);
};

array bernoulli(
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array truncated_normal(
    const array& lower,
    const array& upper,
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array truncated_normal(
    const array& lower,
    const array& upper,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array gumbel(
    const std::vector<int>& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array categorical(
    const array& logits,
    int axis,
    const std::vector<int>& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array categorical(
    const array& logits_,
    int axis,
    int num_samples,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

array categorical(
    const array& logits,
    int axis = -1,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::random
