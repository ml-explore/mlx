// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
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

// Chunks the output along axis 0 across K sub-keys, writing each
// via slice_update + eval to free the fp32 transients between chunks.
// Peak memory ~ (1 + 2/K) * output. Caller must guard with
// !detail::in_tracing(); the per-chunk eval is illegal under tracing.
template <typename ProcessChunk>
static array chunked_fp32_then_cast(
    const Shape& shape,
    Dtype dtype,
    const array& key,
    size_t K,
    const Stream& stream,
    ProcessChunk&& process_chunk) {
  auto subkeys = random::split(key, static_cast<int>(K), stream);
  auto out = zeros(shape, dtype, stream);
  Shape strides_one(shape.size(), 1);
  int base_dim = shape[0] / static_cast<int>(K);
  int remainder = shape[0] % static_cast<int>(K);
  int cursor = 0;
  for (size_t i = 0; i < K; ++i) {
    int this_dim = base_dim + (static_cast<int>(i) < remainder ? 1 : 0);
    Shape chunk_shape = shape;
    chunk_shape[0] = this_dim;
    auto subkey_i = reshape(
        slice(
            subkeys,
            Shape{static_cast<int>(i), 0},
            Shape{static_cast<int>(i) + 1, 2},
            Shape{1, 1},
            stream),
        Shape{2},
        stream);
    auto chunk_out = process_chunk(chunk_shape, subkey_i);
    Shape start = Shape(shape.size(), 0);
    start[0] = cursor;
    Shape stop = shape;
    stop[0] = cursor + this_dim;
    out = slice_update(out, chunk_out, start, stop, strides_one, stream);
    eval(out);
    cursor += this_dim;
  }
  return out;
}

// Target ~256 MB fp32 transient per chunk. Clamped to [4, 256] and
// never larger than shape[0].
static inline size_t pick_chunk_count(
    size_t bits_bytes_fp32,
    int first_dim) {
  constexpr size_t kChunkBytes = 256ULL * 1024 * 1024;
  size_t K = (bits_bytes_fp32 + kChunkBytes - 1) / kChunkBytes;
  K = std::clamp(K, size_t{4}, size_t{256});
  return std::min(K, static_cast<size_t>(first_dim));
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

  // Fused per-thread uniform for half-precision GPU outputs.
  // Bit-identical to vanilla; cuts peak from 3x to 1x output.
  // Skipped under transforms (no vmap rule); falls through to vanilla.
  bool half = (dtype == bfloat16 || dtype == float16);
  size_t total = 1;
  for (auto d : shape) {
    total *= static_cast<size_t>(d);
  }
  bool scalar_lohi = (lo.size() == 1) && (hi.size() == 1);
  bool single_key = !key || (key->shape() == Shape{2});
  bool gpu_stream = (stream.device.type == Device::gpu);
  if (half && (total % 2 == 0) && scalar_lohi && single_key && gpu_stream &&
      !detail::in_tracing()) {
    auto eff_key = key ? *key : KeySequence::default_().next();
    // .item() forces evaluation, so no explicit eval needed here.
    float lo_f = astype(low, float32, stream).item<float>();
    float hi_f = astype(high, float32, stream).item<float>();
    return array(
        shape,
        dtype,
        std::make_shared<RandomUniform>(stream, shape, dtype, lo_f, hi_f),
        {eff_key});
  }

  auto maxval = array(std::numeric_limits<uint32_t>::max(), float32);

  // Chunked fp32-then-cast for large half-precision uniform. Sub-key
  // derivation differs from vanilla so output bytes change (precedent:
  // #904); per-element distribution preserved. fp32 excluded (vanilla
  // already at 1x peak); transforms skipped (per-chunk eval illegal).
  size_t bits_bytes = total * size_of(float32);
  constexpr size_t kChunkBytes = 256ULL * 1024 * 1024;
  if (half && gpu_stream && scalar_lohi && single_key &&
      bits_bytes >= 2 * kChunkBytes && !shape.empty() && shape[0] >= 4 &&
      !detail::in_tracing()) {
    auto eff_key = key ? *key : KeySequence::default_().next();
    size_t K = pick_chunk_count(bits_bytes, shape[0]);
    return chunked_fp32_then_cast(
        shape, dtype, eff_key, K, stream,
        [&](const Shape& chunk_shape, const array& subkey_i) {
          auto bits_i = bits(chunk_shape, size_of(float32), subkey_i, stream);
          auto fp_i = divide(bits_i, maxval, stream);
          auto clipped = astype(minimum(fp_i, upper, stream), dtype, stream);
          return add(multiply(range, clipped, stream), lo, stream);
        });
  }

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
  // Keep normal() on the fp32 sampling path: native bf16 sampling produces
  // only ~382 unique values per 100K samples vs ~2254 for fp32-then-cast
  // (PR #2361).
  auto low = array(std::nextafter(-1.0f, 0.0f), float32);
  auto high = array(1.0f, float32);

  // Chunked normal pipeline; same dispatch rationale as the uniform
  // chunked path above. Lets (46341, 46341) bf16 normal fit on 16 GB.
  bool chunkable_dtype = (dtype == bfloat16 || dtype == float16);
  bool gpu_stream = (stream.device.type == Device::gpu);
  bool single_key = !key || (key->shape() == Shape{2});
  size_t total = 1;
  for (auto d : shape) {
    total *= static_cast<size_t>(d);
  }
  size_t bits_bytes = total * size_of(float32);
  constexpr size_t kChunkBytes = 256ULL * 1024 * 1024;
  if (chunkable_dtype && gpu_stream && single_key &&
      bits_bytes >= 2 * kChunkBytes && !shape.empty() && shape[0] >= 4 &&
      !detail::in_tracing()) {
    auto eff_key = key ? *key : KeySequence::default_().next();
    size_t K = pick_chunk_count(bits_bytes, shape[0]);
    auto applied_scale = array(std::sqrt(2.0), dtype);
    if (scale.has_value()) {
      applied_scale =
          multiply(applied_scale, astype(*scale, dtype, stream), stream);
    }
    array loc_dt = loc.has_value() ? astype(*loc, dtype, stream)
                                   : array(0.0, dtype);
    bool has_loc = loc.has_value();
    return chunked_fp32_then_cast(
        shape, dtype, eff_key, K, stream,
        [&](const Shape& chunk_shape, const array& subkey_i) {
          auto u = uniform(low, high, chunk_shape, float32, subkey_i, stream);
          auto chunk_dt = astype(erfinv(u, stream), dtype, stream);
          chunk_dt = multiply(applied_scale, chunk_dt, stream);
          if (has_loc) {
            chunk_dt = add(loc_dt, chunk_dt, stream);
          }
          return chunk_dt;
        });
  }

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
  auto u = uniform(low, high, shape, float32, key, s);
  return astype(maximum(u, low, s), dtype, s);
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
