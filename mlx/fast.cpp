// Copyright © 2023-2024 Apple Inc.

#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (i < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, tangents);
  std::vector<array> jvp_outs;
  for (int i = 0, j = 0; i < jvps.size(); ++i) {
    if (i < argnums.size() && i == argnums[j]) {
      jvp_outs.push_back(jvps[i]);
      j++;
    }
  }
  return jvp_outs;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    float base,
    float scale,
    int offset,
    StreamOrDevice s /* = {} */) {
  if (x.ndim() != 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have 3 dimensions but got input with " << x.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (traditional && x.shape(-1) != dims) {
    throw std::invalid_argument(
        "[rope] Does not support partial traditional application.");
  }

  auto fallback = [dims, traditional, base, scale, offset, s](
                      const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto t = x.dtype();
    auto N = x.shape(1) + offset;
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto positions = multiply(arange(offset, N, t, s), array(scale, t), s);
    auto freqs = negative(arange(0, half_dims, t, s), s);
    freqs = exp(multiply(freqs, array(std::log(base) / half_dims, t), s), s);
    auto theta =
        multiply(expand_dims(positions, 1, s), expand_dims(freqs, 0, s), s);
    auto coss = cos(theta, s);
    auto sins = sin(theta, s);

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0}, x.shape(), {1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 1}, x.shape(), {1, 1, 2}, s);
      std::vector<array> outs;
      outs.push_back(subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
      outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      for (auto& o : outs) {
        o = expand_dims(o, 3, s);
      }
      return std::vector<array>{reshape(concatenate(outs, 3, s), x.shape(), s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, half_dims}, out_s, s);

      std::vector<array> outs;
      outs.push_back(subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
      outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{concatenate(outs, 2, s)};
    }
  };
  auto stream = to_stream(s);
  if (stream.device == Device::gpu && x.shape(-1) == dims) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_unique<RoPE>(
            stream, fallback, dims, traditional, base, scale, offset),
        {x});
  }
  return fallback({x})[0];
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      offset_ == a_other.offset_);
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask,
    StreamOrDevice s) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Q, K must have matching last dims (d_k aka 'head_dim');
  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // K, V must have matching number of heads (n_kv_heads);
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type({queries, keys, values});
  if (!is_floating_point(final_type) || is_complex(final_type)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  auto out_shape =
      std::vector<int>({q.shape(0), q.shape(1), q.shape(2), v.shape(-1)});

  /* generic implementation for use cases that Metal implementation does not
   * support. For non-supported cases listed below, use MLX primitives:
   * * CPU implementation
   * * batch size > 1
   * * query sequence length > 1
   * * non-null mask
   * * dtype is not fp32 or fp16
   */
  bool needs_mask = mask.has_value();
  auto fallback = [scale, needs_mask, final_type, n_q_heads, n_kv_heads, &s](
                      const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    int B = q.shape(0);
    int L = q.shape(2);
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = reshape(q, {B, n_kv_heads, n_repeats, L, -1}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (needs_mask) {
      scores = add(scores, inputs[3], s);
    }
    scores = astype(
        softmax(astype(scores, float32, s), std::vector<int>{-1}, s),
        final_type,
        s);
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = reshape(out, {B, n_q_heads, L, -1}, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  constexpr const int supported_head_dim = 128;
  const size_t query_head_dim = q.shape(-1);
  const size_t query_sequence_length = q.shape(2);
  bool implementation_supports_use_case = batch_dim == 1 &&
      query_sequence_length == 1 && !mask.has_value() &&
      query_head_dim == supported_head_dim && final_type != bfloat16 &&
      stream.device == Device::gpu;
  // TODO, update routing conditions post further tuning
  implementation_supports_use_case &= false;
  if (implementation_supports_use_case) {
    auto out = array(
        out_shape,
        final_type,
        std::make_unique<ScaledDotProductAttention>(
            stream, fallback, scale, false),
        {q, k, v});
    return out;
  }

  if (mask.has_value()) {
    return fallback({q, k, v, mask.value()})[0];
  } else {
    return fallback({q, k, v})[0];
  }
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other =
      static_cast<const ScaledDotProductAttention&>(other);
  return needs_mask_ == a_other.needs_mask_ && scale_ == a_other.scale_;
}

} // namespace mlx::core::fast
