// Copyright © 2023 Apple Inc.

#include <cmath>
#include <sstream>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/utils.h"

namespace mlx::core::random {

KeySequence::KeySequence(uint64_t seed) : key_(key(seed)) {}

void KeySequence::seed(uint64_t seed) {
  key_ = key((seed));
}

array KeySequence::next() {
  auto out = split(key_);
  key_ = out.first;
  return out.second;
}

void seed(uint64_t seed) {
  KeySequence::default_().seed(seed);
}

array key(uint64_t seed) {
  uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  uint32_t k2 = static_cast<uint32_t>(seed);
  return array({k1, k2});
}

array bits(
    const std::vector<int>& shape,
    int width /* 4 */,
    const std::optional<array>& key_ /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  auto key = key_ ? *key_ : KeySequence::default_().next();
  if (key.dtype() != uint32) {
    std::ostringstream msg;
    msg << "Expected key type uint32 but received " << key.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (key.shape() != std::vector<int>{2}) {
    std::ostringstream msg;
    msg << "Expected key shape (2) but received " << key.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto get_dtype = [width]() {
    switch (width) {
      case 4:
        return uint32;
      case 2:
        return uint16;
      case 1:
        return uint8;
      default:
        std::ostringstream msg;
        msg << "[bits] Bit width must be in {1, 2, 4} but got " << width << ".";
        throw std::invalid_argument(msg.str());
    }
  };
  return array(
      shape,
      get_dtype(),
      std::make_unique<RandomBits>(to_stream(s), shape, width),
      {key});
}

std::pair<array, array> split(const array& key, StreamOrDevice s /* = {} */) {
  auto stream = to_stream(s);
  auto out = mlx::core::split(random::split(key, 2, stream), 2, stream);
  return {reshape(out[0], {2}, stream), reshape(out[1], {2}, stream)};
}

array split(const array& key, int num, StreamOrDevice s /* = {} */) {
  return bits({num, 2}, 4, key, s);
}

// Get the next representable value below 1.0 for half precision
// floating point types (fp16, bf16)
template <typename T>
T below_one() {
  T f = T(1.0);
  uint16_t* m = (uint16_t*)&f;
  *m -= 1;
  return f;
}

array uniform(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!is_floating_point(dtype) && !is_complex(dtype)) {
    throw std::invalid_argument(
        "Can only generate uniform numbers with real floating point type.");
  }

  auto stream = to_stream(s);
  auto range = subtract(high, low, stream);
  auto out_shape = broadcast_shapes(shape, range.shape());
  if (out_shape != shape) {
    std::ostringstream msg;
    msg << "Cannot generate random values of shape " << shape
        << " from broadcasted shape " << out_shape << ".";
    throw std::invalid_argument(msg.str());
  }
  // Get random values between [0, nextafter(maxval, 0.0f)] since samples must
  // be in [low, high)
  auto get_limits = [&dtype]() {
    switch (dtype) {
      case float32:
        return std::make_pair(
            array(std::nextafter(1.0f, 0.0f), float32),
            array(std::numeric_limits<uint32_t>::max(), float32));
      case float16:
        return std::make_pair(
            array(below_one<float16_t>(), float16),
            array(std::numeric_limits<uint16_t>::max(), float32));
      case bfloat16:
        return std::make_pair(
            array(below_one<bfloat16_t>(), bfloat16),
            array(std::numeric_limits<uint16_t>::max(), float32));
      default:
        throw std::runtime_error("[uniform] Unsupported type.");
    }
  };

  auto [upper, maxval] = get_limits();
  auto out = bits(shape, size_of(dtype), key, stream);
  out = astype(divide(out, maxval, stream), dtype, stream);
  out = minimum(out, upper, stream);
  return add(multiply(range, out, stream), low, stream);
}

array uniform(
    const std::vector<int>& shape,
    Dtype dtype,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  return uniform(
      array(0.0, dtype), array(1.0, dtype), shape, dtype, key, to_stream(s));
}

array normal(
    const std::vector<int>& shape,
    Dtype dtype,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  auto stream = to_stream(s);
  auto low = array(std::nextafter(-1.0f, 0.0f), dtype);
  auto high = array(1.0f, dtype);
  auto samples = uniform(low, high, shape, dtype, key, stream);
  return multiply(
      array(std::sqrt(2.0), dtype), erfinv(samples, stream), stream);
}

array randint(
    const array& low,
    const array& high,
    const std::vector<int>& shape,
    Dtype dtype /* = int32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!is_integral(dtype)) {
    throw std::invalid_argument(
        "[randint] randint only accepts integer dtypes and bool.");
  }
  auto u = uniform(low, high, shape, float32, key, s);
  return astype(maximum(u, low, s), dtype, s);
}

array bernoulli(
    const array& p,
    const std::vector<int>& shape,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!is_floating_point(p.dtype())) {
    throw std::invalid_argument(
        "[bernoulli] bernoulli probability `p` must be a float type.");
  }
  auto res = uniform(shape, p.dtype(), key, s);
  res = less(res, p, s);
  if (res.shape() != shape) {
    throw std::invalid_argument(
        "[bernoulli] shape of `p` is incompatible with argument `shape`.");
  }
  return res;
}

array bernoulli(
    const array& p,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  return bernoulli(p, p.shape(), key, s);
}

array bernoulli(
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  return bernoulli(array(0.5f), key, s);
}

array truncated_normal(
    const array& lower,
    const array& upper,
    const std::vector<int>& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  // Same as
  // https://jax.readthedocs.io/en/latest/_modules/jax/_src/random.html#truncated_normal

  if (!is_floating_point(dtype)) {
    throw std::invalid_argument(
        "[trunc_normal] trunc_normal only accepts floating point dtypes.");
  }

  auto sqrt2 = array(std::sqrt(2.0), dtype);
  auto lower_t = astype(lower, dtype, s);
  auto upper_t = astype(upper, dtype, s);
  auto a = erf(divide(lower_t, sqrt2, s), s);
  auto b = erf(divide(upper_t, sqrt2, s), s);
  auto u = uniform(a, b, shape, dtype, key, s);
  auto out = multiply(sqrt2, erfinv(u, s), s);

  // Clip in bouds
  return maximum(minimum(upper_t, out, s), lower_t, s);
}

array truncated_normal(
    const array& lower,
    const array& upper,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  auto shape = broadcast_shapes(lower.shape(), upper.shape());
  return truncated_normal(lower, upper, shape, dtype, key, s);
}

array gumbel(
    const std::vector<int>& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  // -log(-log(uniform(shape)))
  return negative(
      log(negative(log(uniform(shape, dtype, key, s), s), s), s), s);
}

int get_valid_axis(int axis, int ndim) {
  int ax = axis < 0 ? axis + ndim : axis;
  if (ax < 0 || ax >= ndim) {
    std::ostringstream msg;
    msg << "[categorical] Invalid axis " << axis << " for logits with " << ndim
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  return ax;
}

array categorical_impl(
    const array& logits,
    int axis,
    const std::vector<int>& shape,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s) {
  auto gumbel_shape = shape;
  auto offset = axis + shape.size() - logits.ndim() + 1;
  gumbel_shape.insert(gumbel_shape.begin() + offset, logits.shape(axis));
  auto g = gumbel(gumbel_shape, float32, key, s);
  return argmax(add(g, logits, s), offset, false, s);
}

array categorical(
    const array& logits,
    int axis,
    const std::vector<int>& shape,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  // Validate and normalize axis
  axis = get_valid_axis(axis, logits.ndim());

  // Check that shape broadcasts with reduce(logits, axis)
  auto reduced_shape = logits.shape();
  reduced_shape.erase(reduced_shape.begin() + axis);
  if (broadcast_shapes(shape, reduced_shape) != shape) {
    std::ostringstream msg;
    msg << "[categorical] Requested shape " << shape
        << " is not broadcast compatable with reduced logits shape"
        << reduced_shape << ".";
    throw std::invalid_argument(msg.str());
  }

  return categorical_impl(logits, axis, shape, key, s);
}

array categorical(
    const array& logits_,
    int axis,
    int num_samples,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  axis = get_valid_axis(axis, logits_.ndim());
  auto logits = expand_dims(logits_, -1);
  auto shape = logits.shape();
  shape.erase(shape.begin() + axis);
  shape.back() = num_samples;
  return categorical_impl(logits, axis, shape, key, s);
}

array categorical(
    const array& logits,
    int axis /* = -1 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  axis = get_valid_axis(axis, logits.ndim());
  auto shape = logits.shape();
  shape.erase(shape.begin() + axis);
  return categorical_impl(logits, axis, shape, key, s);
}

} // namespace mlx::core::random
