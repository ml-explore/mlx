// Copyright © 2023-2024 Apple Inc.

#include <cmath>
#include <sstream>

#include "mlx/linalg.h"
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
    const Shape& shape,
    int width /* 4 */,
    const std::optional<array>& key_ /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  auto key = key_ ? *key_ : KeySequence::default_().next();
  if (key.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[bits] Expected key type uint32 but received " << key.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (key.shape() != Shape{2}) {
    std::ostringstream msg;
    msg << "[bits] Expected key shape (2) but received " << key.shape() << ".";
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
      std::make_shared<RandomBits>(to_stream(s), shape, width),
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
    const Shape& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!issubdtype(dtype, floating)) {
    throw std::invalid_argument(
        "[uniform] Can only generate uniform numbers with real "
        "floating point type.");
  }

  auto stream = to_stream(s);
  auto lo = astype(low, dtype, stream);
  auto hi = astype(high, dtype, stream);
  auto range = subtract(hi, lo, stream);
  auto out_shape = broadcast_shapes(shape, range.shape());
  if (out_shape != shape) {
    std::ostringstream msg;
    msg << "[uniform] Cannot generate random values of shape " << shape
        << " from broadcasted shape " << out_shape << ".";
    throw std::invalid_argument(msg.str());
  }

  // Get random values between [0, nextafter(1.0, 0.0)] since samples must
  // be in [low, high)
  auto get_upper = [&dtype]() {
    switch (dtype) {
      case float32:
        return array(std::nextafter(1.0f, 0.0f), float32);
      case float16:
        return array(below_one<float16_t>(), float32);
      case bfloat16:
        return array(below_one<bfloat16_t>(), float32);
      default:
        throw std::runtime_error("[uniform] Unsupported type.");
    }
  };

  auto upper = get_upper();
  auto maxval = array(std::numeric_limits<uint32_t>::max(), float32);
  auto out = bits(shape, size_of(float32), key, stream);
  out = divide(out, maxval, stream);
  out = astype(minimum(out, upper, stream), dtype, stream);
  return add(multiply(range, out, stream), lo, stream);
}

array uniform(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  return uniform(
      array(0.0, dtype), array(1.0, dtype), shape, dtype, key, to_stream(s));
}

inline array complex_normal(
    Shape shape,
    const std::optional<array>& loc,
    const std::optional<array>& scale,
    const std::optional<array>& key,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  auto low = array(std::nextafter(-1.0f, 0.0f), float32);
  auto high = array(1.0f, float32);
  shape.push_back(2);
  auto samples =
      erfinv(uniform(low, high, shape, float32, key, stream), stream);
  samples = squeeze(view(samples, complex64, stream), -1, stream);
  if (scale.has_value()) {
    samples = multiply(*scale, samples, stream);
  }
  if (loc.has_value()) {
    samples = add(*loc, samples, stream);
  }
  return samples;
}

array normal(
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& loc,
    const std::optional<array>& scale,
    const std::optional<array>& key,
    StreamOrDevice s /* = {} */) {
  if (dtype == complex64) {
    return complex_normal(shape, loc, scale, key, s);
  } else if (!issubdtype(dtype, floating)) {
    throw std::invalid_argument(
        "[normal] Can only generate uniform numbers with "
        "floating point type.");
  }

  auto stream = to_stream(s);
  auto low = array(std::nextafter(-1.0f, 0.0f), float32);
  auto high = array(1.0f, float32);
  auto samples = uniform(low, high, shape, float32, key, stream);
  auto applied_scale = array(std::sqrt(2.0), dtype);
  if (scale.has_value()) {
    applied_scale =
        multiply(applied_scale, astype(*scale, dtype, stream), stream);
  }
  samples = astype(erfinv(samples, stream), dtype, stream);
  samples = multiply(applied_scale, samples, stream);
  if (loc.has_value()) {
    samples = add(astype(*loc, dtype, stream), samples, stream);
  }
  return samples;
}

array multivariate_normal(
    const array& mean,
    const array& cov,
    const Shape& shape,
    Dtype dtype,
    const std::optional<array>& key /* = nullopt */,
    StreamOrDevice s) {
  auto stream = to_stream(s);

  if (dtype != float32) {
    throw std::invalid_argument("[multivariate_normal] dtype must be float32.");
  }

  if (mean.ndim() < 1) {
    throw std::invalid_argument(
        "[multivariate_normal] mean must have at least one dimension.");
  }

  if (cov.ndim() < 2) {
    throw std::invalid_argument(
        "[multivariate_normal] cov must have at least two dimensions.");
  }

  auto n = mean.shape(-1);

  // Check shapes compatibility of mean and cov
  if (cov.shape(-1) != cov.shape(-2)) {
    throw std::invalid_argument(
        "[multivariate_normal] last two dimensions of cov must be equal.");
  }
  if (n != cov.shape(-1)) {
    throw std::invalid_argument(
        "[multivariate_normal] mean and cov must have compatible shapes.");
  }

  // Compute output shape
  auto truncated_mean_shape =
      Shape(mean.shape().begin(), mean.shape().end() - 1);
  auto truncated_cov_shape = Shape(cov.shape().begin(), cov.shape().end() - 2);
  auto output_shape =
      broadcast_shapes(truncated_cov_shape, truncated_mean_shape);
  output_shape = broadcast_shapes(output_shape, shape);
  output_shape.push_back(n);

  // Compute the square-root of the covariance matrix, using the SVD
  auto covariance = astype(cov, float32, stream);
  auto SVD = linalg::svd(covariance, true, stream);
  auto std = astype(
      matmul(
          multiply(
              SVD[0], expand_dims(sqrt(SVD[1], stream), -2, stream), stream),
          SVD[2],
          stream),
      dtype,
      stream);

  // Generate standard the samples
  auto standard_normal = normal(output_shape, dtype, 0.0, 1.0, key, stream);
  auto scaled_out = squeeze(
      matmul(expand_dims(standard_normal, -2, stream), std, stream),
      -2,
      stream);
  return add(mean, scaled_out, stream);
}

array randint(
    const array& low,
    const array& high,
    const Shape& shape,
    Dtype dtype /* = int32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (issubdtype(dtype, inexact)) {
    throw std::invalid_argument(
        "[randint] randint only accepts integer dtypes and bool.");
  }
  auto stream = to_stream(s);

  // Sampling through `uniform()` rounds a float32 draw to the target integer
  // type. float32 has a 24-bit mantissa, so once |low| or |high| reaches 2^24
  // the rounded value can land on the excluded `high`, skip valid integers
  // entirely, or -- when the whole interval sits between two consecutive
  // representable floats -- collapse every draw in the interval to a single
  // constant, regardless of how narrow the interval is. Sample directly in
  // integer space instead: draw raw random bits and reduce them modulo the
  // (integer) range, which is exact for any low/high representable in int64,
  // independent of magnitude.
  auto lo = astype(low, int64, stream);
  auto hi = astype(high, int64, stream);
  auto range = subtract(hi, lo, stream);
  auto out_shape = broadcast_shapes(shape, range.shape());
  if (out_shape != shape) {
    std::ostringstream msg;
    msg << "[randint] Cannot generate random values of shape " << shape
        << " from broadcasted shape " << out_shape << ".";
    throw std::invalid_argument(msg.str());
  }
  // The width of an int64 interval does not itself fit in int64: `[-2^62,
  // 2^62)` is 2^63 wide and `[int64_min, int64_max)` is 2^64 - 1 wide, so the
  // subtraction above wraps negative for any interval wider than int64_max.
  // Clamping that with a signed `maximum(range, 1)` reads the wrapped value as
  // negative and yields 1, collapsing every draw onto `low` -- the same defect
  // this patch fixes at 2^24, relocated to a wider magnitude. The wrapped bit
  // pattern IS the correct unsigned width, so reinterpret it as uint64 and do
  // the empty-interval test on the signed values, where the comparison is
  // still meaningful. Reported by @axiom-of-choice on #3936.
  auto empty = less_equal(hi, lo, stream);
  auto safe_range = where(
      empty,
      array(uint64_t(1), uint64),
      astype(range, uint64, stream),
      stream);

  array raw(0);
  if (dtype == int64 || dtype == uint64) {
    // The output dtype can represent ranges beyond what one uint32 draw (32
    // bits of entropy) can cover, so combine two independent draws into 64
    // bits. Explicitly split any caller-provided key so the two draws don't
    // repeat the same bits.
    array lo_bits(0), hi_bits(0);
    if (key) {
      auto [k1, k2] = split(*key, stream);
      lo_bits = bits(shape, 4, k1, stream);
      hi_bits = bits(shape, 4, k2, stream);
    } else {
      lo_bits = bits(shape, 4, std::nullopt, stream);
      hi_bits = bits(shape, 4, std::nullopt, stream);
    }
    raw = bitwise_or(
        left_shift(
            astype(hi_bits, uint64, stream),
            array(uint64_t(32), uint64),
            stream),
        astype(lo_bits, uint64, stream),
        stream);
  } else {
    // Every other supported dtype's own domain spans at most 2^32 values, so
    // one uint32 draw (32 bits of entropy) always has enough range.
    raw = astype(bits(shape, 4, key, stream), uint64, stream);
  }

  // Reducing unsigned avoids numpy's negative-dividend `remainder` (which
  // returns `raw % r + r` and skews the low end of the interval by
  // `2^63 mod r`); an unsigned reduction is uniform on [0, r) by construction.
  auto offset = remainder(raw, safe_range, stream);
  // Add in uint64 so the wrap for full-width intervals is well defined rather
  // than signed overflow; the resulting bit pattern is the correct value once
  // reinterpreted as the (possibly signed) target dtype.
  return astype(
      add(astype(lo, uint64, stream), offset, stream), dtype, stream);
}

array bernoulli(
    const array& p,
    const Shape& shape,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!issubdtype(p.dtype(), floating)) {
    throw std::invalid_argument(
        "[bernoulli] bernoulli probability `p` must be a float type.");
  }

  // Place p on the scale [0, nexthigher(UINT32_MAX)] so that if p >= 1.0 we
  // get all true and if p <= 0.0 we get all false
  auto upper = array(
      std::nextafter(
          static_cast<float>(std::numeric_limits<uint32_t>::max()),
          std::numeric_limits<float>::max()),
      float32);
  auto res = less(bits(shape, key, s), multiply(p, upper, s), s);
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
    const Shape& shape,
    Dtype dtype /* = float32 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  // Same as
  // https://jax.readthedocs.io/en/latest/_modules/jax/_src/random.html#truncated_normal

  if (!issubdtype(dtype, floating)) {
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

  // Clip in bounds
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
    const Shape& shape,
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
    const Shape& shape,
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
    const Shape& shape,
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
        << " is not broadcast compatible with reduced logits shape"
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

array laplace(
    const Shape& shape,
    Dtype dtype,
    const float loc /* = 0.0 */,
    const float scale /* = 1.0 */,
    const std::optional<array>& key /*= nullopt */,
    StreamOrDevice s /* = {} */) {
  if (!issubdtype(dtype, floating)) {
    throw std::invalid_argument(
        "[laplace] Can only generate uniform numbers with real"
        "floating point type.");
  }

  auto stream = to_stream(s);
  auto low = array(std::nextafter(-1.0f, 0.0f), float32);
  auto high = array(1.0f, float32);
  auto samples = uniform(low, high, shape, float32, key, stream);
  // Use inverse CDF to generate Laplacian noise
  samples = multiply(
      sign(samples, stream),
      log1p(
          multiply(array(-1.0f, dtype), abs(samples, stream), stream), stream),
      stream);
  samples = astype(samples, dtype, stream);

  if (scale != 1.0) {
    samples = multiply(array(scale, dtype), samples, stream);
  }
  if (loc != 0.0) {
    samples = add(array(loc, dtype), samples, stream);
  }
  return samples;
}

array permutation(
    const array& x,
    int axis /* = 0 */,
    const std::optional<array>& key /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return take(x, permutation(x.shape(axis), key, s), axis, s);
}

array permutation(
    int x,
    const std::optional<array>& key /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return argsort(bits({x}, key, s), s);
}

} // namespace mlx::core::random
