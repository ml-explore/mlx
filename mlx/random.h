// Copyright © 2023 Apple Inc.

#pragma once

#include <chrono>
#include <optional>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::random {

class MLX_API KeySequence {
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
MLX_API array key(uint64_t seed);

/** Seed the default PRNG key. */
MLX_API void seed(uint64_t seed);

/** Generate an array with type uint32 filled with random bits. */
MLX_API array bits(
    const Shape& shape,
    int width,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array bits(
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bits(shape, 4, key, s);
}

/** Split the rng key into a pair of keys. */
MLX_API std::pair<array, array> split(const array& key, StreamOrDevice s = {});

/** Split the rng key into `num` keys. */
MLX_API array split(const array& key, int num, StreamOrDevice s = {});

/** Generate uniform random numbers between low and high. */
MLX_API array uniform(
    const array& low,
    const array& high,
    const Shape& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T, typename U>
array uniform(
    T low,
    U high,
    const Shape& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(array(low), array(high), shape, dtype, key, to_stream(s));
}

/** Generate uniform random numbers between 0 and 1. */
MLX_API array uniform(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array uniform(
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return uniform(shape, float32, key, s);
}

/** Generate samples from the standard normal distribution. */
MLX_API array normal(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& loc,
    const std::optional<array>& scale,
    const std::optional<array>& key,
    StreamOrDevice s = {});
inline array normal(
    const Shape& shape,
    Dtype dtype,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  auto loc_ = loc == 0 ? std::nullopt : std::make_optional(array(loc, dtype));
  auto scale_ =
      scale == 1 ? std::nullopt : std::make_optional(array(scale, dtype));
  return normal(shape, dtype, loc_, scale_, key, s);
}
inline array normal(
    const Shape& shape,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, float32, loc, scale, key, s);
}
inline array normal(
    const Shape& shape,
    const Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, dtype, std::nullopt, std::nullopt, key, s);
}
inline array normal(
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return normal(shape, float32, std::nullopt, std::nullopt, key, s);
}

/** Generate samples from a multivariate normal distribution. **/
MLX_API array multivariate_normal(
    const array& mean,
    const array& cov,
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

/** Generate integer samples uniformly at random */
MLX_API array randint(
    const array& low,
    const array& high,
    const Shape& shape,
    Dtype dtype = int32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T, typename U>
array randint(
    T low,
    U high,
    const Shape& shape,
    Dtype dtype = int32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return randint(array(low), array(high), shape, dtype, key, to_stream(s));
}

/** Generate binary variables with probability to be true equal to p */
MLX_API array bernoulli(
    const array& p,
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
MLX_API array bernoulli(
    const array& p,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

template <typename T>
array bernoulli(
    T p,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bernoulli(array(p), key, s);
}

template <typename T>
array bernoulli(
    T p,
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return bernoulli(array(p), shape, key, s);
}

MLX_API array bernoulli(
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array truncated_normal(
    const array& lower,
    const array& upper,
    const Shape& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array truncated_normal(
    const array& lower,
    const array& upper,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array gumbel(
    const Shape& shape,
    Dtype dtype = float32,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array categorical(
    const array& logits,
    int axis,
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array categorical(
    const array& logits_,
    int axis,
    int num_samples,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

MLX_API array categorical(
    const array& logits,
    int axis = -1,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

/** Generate samples from the laplace distribution. */
MLX_API array laplace(
    const Shape& shape,
    Dtype dtype,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});
inline array laplace(
    const Shape& shape,
    const float loc,
    const float scale,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return laplace(shape, float32, loc, scale, key, s);
}
inline array laplace(
    const Shape& shape,
    const Dtype dtype,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return laplace(shape, dtype, 0.0, 1.0, key, s);
}
inline array laplace(
    const Shape& shape,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {}) {
  return laplace(shape, float32, 0.0, 1.0, key, s);
}

/* Randomly permute the elements of x along the given axis. */
MLX_API array permutation(
    const array& x,
    int axis = 0,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

/* A random permutation of `arange(x)` */
MLX_API array permutation(
    int x,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::random
