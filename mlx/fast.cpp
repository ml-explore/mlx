// Copyright © 2023-2024 Apple Inc.
#include <cassert>
#include <iostream>
#include <numeric>
#include <regex>

#include "mlx/backend/common/compiled.h"
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
    if (j < argnums.size() && i == argnums[j]) {
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
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rms_norm(
    const array& x,
    const array& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (weight.ndim() != 1) {
    std::ostringstream msg;
    msg << "[rms_norm] weight must have 1 dimension but has " << weight.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  auto out_type = result_type(x, weight);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [eps, out_type, s](const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);
    x = multiply(
        x,
        rsqrt(
            add(mean(square(x, s), -1, /* keepdims */ true, s),
                array(eps, float32),
                s),
            s),
        s);
    x = astype(x, out_type, s);
    return std::vector<array>{multiply(inputs[1], x, s)};
  };
  if (s.device == Device::gpu) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), astype(weight, out_type, s)});
  }
  return fallback({x, weight})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    // df/dx
    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    vjps.push_back(
        sum(multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[layer_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (weight.has_value() && (*weight).ndim() != 1) {
    std::ostringstream msg;
    msg << "[layer_norm] weight must have 1 dimension but has "
        << (*weight).ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (bias.has_value() && (*bias).ndim() != 1) {
    std::ostringstream msg;
    msg << "[layer_norm] bias must have 1 dimension but has " << (*bias).ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  auto out_type = (weight.has_value())
      ? ((bias.has_value()) ? result_type(x, *weight, *bias)
                            : result_type(x, *weight))
      : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[layer_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  bool has_weight = weight.has_value();
  bool has_bias = bias.has_value();
  auto fallback = [has_weight, has_bias, eps, out_type, s](
                      const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);

    // Should I not be smart here and leave the double mean to simplify()?
    auto mu = mean(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto mu2 = square(mu, s);
    auto x2 = mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto v = subtract(x2, mu2, s);

    x = multiply(subtract(x, mu, s), rsqrt(add(v, array(eps, float32), s), s));
    x = astype(x, out_type, s);

    // If the LN is affine then transform x according to the weight and bias
    if (has_weight) {
      x = multiply(x, inputs[1], s);
    }
    if (has_bias) {
      x = add(x, inputs[2], s);
    }

    return std::vector<array>{x};
  };

  auto passed_weight =
      astype((weight.has_value()) ? *weight : array(1, out_type), out_type);
  auto passed_bias =
      astype((bias.has_value()) ? *bias : array(0, out_type), out_type);

  if (s.device == Device::gpu) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<LayerNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight, passed_bias});
  }
  return fallback({x, passed_weight, passed_bias})[0];
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& b = inputs[2];
    auto& g = inputs[3];

    std::vector<array> vjps;

    auto norm = number_of_elements(x, {-1}, true, x.dtype(), s);
    auto sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto mu = multiply(sumx, norm, s);
    auto mu2 = multiply(sumx2, norm, s);
    auto var = subtract(mu2, square(mu, s), s);
    auto n = rsqrt(add(var, array(eps, x.dtype()), s));
    auto n3 = power(n, array(3, x.dtype()), s);
    auto x_c = subtract(x, mu, s);

    // df/dx
    auto wg = multiply(w, g, s);
    auto sumwg =
        multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
    auto sumwgxc = multiply(
        sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
        norm,
        s);
    auto t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
    auto t2 = multiply(subtract(wg, sumwg, s), n, s);
    vjps.push_back(subtract(t2, t1, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x_c, n, s), s), axes, /* keepdims= */ false, s));
    }

    // df/db
    if (b.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(g, axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape(), primals[2].shape()},
      {primals[0].dtype(), primals[1].dtype(), primals[2].dtype()},
      std::make_shared<LayerNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& a_other = static_cast<const LayerNorm&>(other);
  return eps_ == a_other.eps_;
}

bool LayerNormVJP::is_equivalent(const Primitive& other) const {
  const LayerNormVJP& a_other = static_cast<const LayerNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    int offset,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (inputs.size() == 2 &&
      (inputs[1].ndim() != 1 || inputs[1].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[1].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, offset, forward, s](
                      std::vector<array> inputs) {
    auto& shape = inputs[0].shape();
    int ndim = shape.size();
    auto x = reshape(inputs[0], {-1, shape[ndim - 2], shape[ndim - 1]}, s);
    auto t = x.dtype();
    auto N = x.shape(1) + offset;
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto positions = multiply(arange(offset, N, t, s), array(scale, t), s);

    auto default_inv_freqs = [&inputs, &s, &t, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, t, s),
              array(std::log(base) / half_dims, t),
              s),
          s);
    };

    auto inv_freqs =
        inputs.size() == 2 ? reciprocal(inputs[1], s) : default_inv_freqs();
    auto theta =
        multiply(expand_dims(positions, 1, s), expand_dims(inv_freqs, 0, s), s);
    auto coss = cos(theta, s);
    auto sins = sin(theta, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 =
          slice(x, {0, 0, 0}, {x.shape(0), x.shape(1), dims}, {1, 1, 2}, s);
      auto x2 =
          slice(x, {0, 0, 1}, {x.shape(0), x.shape(1), dims}, {1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, 3, s);
      }
      auto out = concatenate(outs, 3, s);
      if (dims < x.shape(-1)) {
        out = reshape(out, {x.shape(0), x.shape(1), dims});
        out = concatenate({out, slice(x, {0, 0, dims}, x.shape(), s)}, 2, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, 2, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (stream.device == Device::gpu) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, offset, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument(
          "[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      offset,
      true,
      s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   offset = offset_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{rope(
        std::move(inputs),
        dims,
        traditional,
        base,
        scale,
        offset,
        !forward,
        s)};
  };

  auto inputs = cotangents;
  if (primals.size() == 2) {
    inputs.push_back(primals[1]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, offset_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      offset_ == a_other.offset_ && forward_ == a_other.forward_);
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask,
    const std::optional<int> memory_efficient_threshold,
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

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  /* generic implementation for use cases that Metal implementation does not
   * support. For non-supported cases listed below, use MLX primitives:
   * * CPU implementation
   * * batch size > 1 for decoding or causal attention
   * * query sequence length > 1 for decoding
   * * query sequence length > 16 && non-null mask (causal attention)
   * * non-null mask
   * * dtype is not fp32 or fp16
   */

  int threshold = 1e6;
  if (memory_efficient_threshold.has_value()) {
    threshold = std::max(1, memory_efficient_threshold.value());
  }

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
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = reshape(out, {B, n_q_heads, L, -1}, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  const size_t value_head_dim = v.shape(-1);
  const size_t query_head_dim = q.shape(-1);
  const size_t query_sequence_length = q.shape(2);

  bool implementation_supports_use_case = query_head_dim == value_head_dim;

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128;
  const bool sdpa_full_supported_head_dim =
      query_head_dim == 64 || query_head_dim == 128;

  const bool supports_sdpa_full = query_sequence_length >= threshold &&
      !mask.has_value() && sdpa_full_supported_head_dim &&
      n_q_heads == n_kv_heads && final_type != bfloat16 &&
      stream.device == Device::gpu;

  const bool supports_sdpa_vector = query_sequence_length == 1 &&
      !mask.has_value() && sdpa_vector_supported_head_dim &&
      stream.device == Device::gpu;

  implementation_supports_use_case &=
      supports_sdpa_full || supports_sdpa_vector;

  if (implementation_supports_use_case) {
    auto out_shape =
        std::vector<int>({q.shape(0), q.shape(1), q.shape(2), v.shape(-1)});
    return array(
        std::move(out_shape),
        final_type,
        std::make_shared<ScaledDotProductAttention>(
            stream, fallback, scale, /*needs_mask=*/false, /*quantized=*/false),
        {q, k, v});
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
  return needs_mask_ == a_other.needs_mask_ && scale_ == a_other.scale_ &&
      quantized_ == a_other.quantized_;
}

array quantized_scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& key_scales,
    const array& key_biases,
    const array& values,
    const array& value_scales,
    const array& value_biases,
    const float scale,
    const std::optional<array>& mask,
    const int group_size,
    const int bits,
    StreamOrDevice s) {
  int el_per_int = 32 / bits;
  int out_dim = values.shape(-1) * el_per_int;

  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  auto out_shape = std::vector<int>(
      {queries.shape(0), queries.shape(1), queries.shape(2), out_dim});
  auto stream = to_stream(s);
  bool needs_mask = mask.has_value();
  auto fallback =
      [scale, needs_mask, n_q_heads, n_kv_heads, group_size, bits, &s](
          const std::vector<array>& inputs) -> std::vector<array> {
    int n_repeats = n_q_heads / n_kv_heads;

    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);

    auto k = inputs[1];
    auto k_scales = inputs[2];
    auto k_biases = inputs[3];
    auto v = inputs[4];
    auto v_scales = inputs[5];
    auto v_biases = inputs[6];

    int B = q.shape(0);
    int L = q.shape(2);

    if (n_repeats > 1) {
      q = reshape(q, {B, n_kv_heads, n_repeats, L, -1}, s);
      k = expand_dims(k, 2, s);
      k_scales = expand_dims(k_scales, 2, s);
      k_biases = expand_dims(k_biases, 2, s);
      v = expand_dims(v, 2, s);
      v_scales = expand_dims(v_scales, 2, s);
      v_biases = expand_dims(v_biases, 2, s);
    }

    array scores = quantized_matmul(
        q,
        k,
        k_scales,
        k_biases,
        /*transpose=*/true,
        /*group_size=*/group_size,
        /*bits=*/bits,
        s);
    if (needs_mask) {
      scores = add(scores, inputs[7], s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    array out = quantized_matmul(
        scores,
        v,
        v_scales,
        v_biases,
        /*transpose=*/false,
        /*group_size=*/group_size,
        /*bits=*/bits,
        s);
    if (n_repeats > 1) {
      out = reshape(out, {B, n_q_heads, L, -1}, s);
    }
    return std::vector<array>{out};
  };

  int L = queries.shape(2);
  if (L > 1) {
    if (needs_mask) {
      return fallback(
          {queries,
           keys,
           key_scales,
           key_biases,
           values,
           value_scales,
           value_biases,
           mask.value()})[0];
    } else {
      return fallback(
          {queries,
           keys,
           key_scales,
           key_biases,
           values,
           value_scales,
           value_biases})[0];
    }
  } else {
    return array(
        std::move(out_shape),
        queries.dtype(),
        std::make_shared<ScaledDotProductAttention>(
            stream,
            fallback,
            scale,
            /*needs_mask=*/false,
            /*quantized=*/true,
            group_size,
            bits),
        {queries,
         keys,
         key_scales,
         key_biases,
         values,
         value_scales,
         value_biases});
  }
}

array pack_and_quantize(
    array& packed_w,
    const array& scales,
    const array& biases,
    int group_size,
    int bits,
    const Stream& s) {
  int el_per_int = 32 / bits;
  array zero(0, packed_w.dtype());
  array n_bins((1 << bits) - 1, packed_w.dtype()); // 2**bits - 1
  array shifts = power(array(2, uint32), arange(0, 32, bits, uint32, s), s);
  packed_w = astype(
      clip(
          round(divide(subtract(packed_w, biases, s), scales, s), s),
          zero,
          n_bins,
          s),
      uint32,
      s);
  packed_w = reshape(packed_w, {packed_w.shape(0), -1, el_per_int}, s);
  packed_w = sum(
      multiply(packed_w, shifts, s), /* axis= */ 2, /* keepdims= */ false, s);
  return packed_w;
}

std::tuple<array, array, array>
affine_quantize(const array& w, int group_size, int bits, StreamOrDevice s_) {
  auto s = to_stream(s_);

  if (group_size != 32 && group_size != 64 && group_size != 128) {
    std::ostringstream msg;
    msg << "[quantize] The requested group size " << group_size
        << " is not supported. The supported group sizes are 64 and 128.";
    throw std::invalid_argument(msg.str());
  }

  if (bits != 2 && bits != 4 && bits != 8) {
    std::ostringstream msg;
    msg << "[quantize] The requested number of bits " << bits
        << " is not supported. The supported bits are 2, 4 and 8.";
    throw std::invalid_argument(msg.str());
  }

  if (w.ndim() < 2) {
    std::ostringstream msg;
    msg << "[quantize] The matrix to be quantized must have at least 2 dimension "
        << "but it has only " << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  if ((w.shape(-1) % group_size) != 0) {
    std::ostringstream msg;
    msg << "[quantize] The last dimension of the matrix needs to be divisible by "
        << "the quantization group size " << group_size
        << ". However the provided " << " matrix has shape " << w.shape();
    throw std::invalid_argument(msg.str());
  }

  int el_per_int = 32 / bits;

  auto fallback = [group_size, bits, el_per_int, s](
                      const std::vector<array>& inputs) -> std::vector<array> {
    auto& w = inputs[0];
    auto wshape = w.shape();
    wshape.back() = -1;

    array zero(0, w.dtype());
    array n_bins((1 << bits) - 1, w.dtype()); // 2**bits - 1
    array eps(1e-7, w.dtype());

    array packed_w = reshape(w, {-1, w.shape(-1) / group_size, group_size}, s);

    array w_max = max(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
    array w_min = min(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
    array mask = greater(abs(w_min, s), abs(w_max, s), s);
    array scales =
        maximum(divide(subtract(w_max, w_min, s), n_bins, s), eps, s);
    scales = where(mask, scales, negative(scales, s), s);
    array edge = where(mask, w_min, w_max, s);
    array q0 = round(divide(edge, scales, s), s);
    scales = where(not_equal(q0, zero, s), divide(edge, q0, s), scales);
    array biases = where(equal(q0, zero, s), zero, edge, s);

    packed_w = pack_and_quantize(packed_w, scales, biases, group_size, bits, s);
    return {
        reshape(packed_w, wshape, s),
        reshape(scales, wshape, s),
        reshape(biases, wshape, s),
    };
  };

  std::vector<array> outputs;
  if (s.device == Device::gpu) {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) / el_per_int;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size;
    outputs = array::make_arrays(
        {wq_shape, sshape, sshape},
        {uint32, w.dtype(), w.dtype()},
        std::make_shared<AffineQuantize>(s, fallback, group_size, bits, false),
        {w});
  } else {
    outputs = fallback({w});
  }
  return {outputs[0], outputs[1], outputs[2]};
}

array affine_quantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size,
    int bits,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int el_per_int = 32 / bits;
  auto fallback = [group_size, bits, el_per_int, s](
                      const std::vector<array>& inputs) -> std::vector<array> {
    auto& w = inputs[0];
    auto scales = expand_dims(inputs[1], -1, s);
    auto biases = expand_dims(inputs[2], -1, s);

    auto wshape = w.shape();
    wshape.back() = -1;

    array packed_w = reshape(w, {-1, w.shape(-1) / group_size, group_size}, s);
    packed_w = pack_and_quantize(packed_w, scales, biases, group_size, bits, s);
    return {reshape(packed_w, wshape, s)};
  };

  if (s.device == Device::gpu) {
    auto out_shape = w.shape();
    out_shape.back() = w.shape(-1) / el_per_int;
    return array(
        out_shape,
        uint32,
        std::make_shared<AffineQuantize>(s, fallback, group_size, bits, false),
        {w, scales, biases});
  }
  return fallback({w, scales, biases})[0];
}

array affine_dequantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size,
    int bits,
    StreamOrDevice s_) {
  if (bits <= 0) {
    std::ostringstream msg;
    msg << "[dequantize] Invalid value for bits: " << bits;
    throw std::invalid_argument(msg.str());
  }
  if (group_size <= 0) {
    std::ostringstream msg;
    msg << "[dequantize] Invalid value for group_size: " << group_size;
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() < 2 || scales.ndim() < 2 || biases.ndim() < 2) {
    std::ostringstream msg;
    msg << "[quantize] The matrix to be quantized must have at least 2 dimension "
        << "but it has only " << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto wshape = w.shape();
  auto sshape = scales.shape();
  auto bshape = biases.shape();
  wshape.back() = -1;
  sshape.back() = -1;
  bshape.back() = -1;

  if (wshape != sshape || wshape != bshape) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales and biases does not match the matrix");
  }

  if (w.dtype() != uint32) {
    throw std::invalid_argument(
        "[dequantize] The matrix should be given as a uint32");
  }

  // Packing into uint32
  int el_per_int = 32 / bits;

  if (w.shape(-1) * el_per_int != scales.shape(-1) * group_size) {
    std::ostringstream msg;
    msg << "[dequantize] Shape of scales and biases does not match the matrix "
        << "given the quantization parameters. Provided matrix of shape "
        << w.shape() << " and scales/biases of shape " << scales.shape()
        << " with group_size=" << group_size << " and bits=" << bits << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);

  auto fallback =
      [&wshape, &sshape, &scales, &biases, group_size, bits, el_per_int, s](
          const std::vector<array>& inputs) -> std::vector<array> {
    auto& w = inputs[0];
    auto& scales = inputs[1];
    auto& biases = inputs[2];
    std::vector<array> parts;
    for (int start = 0; start < 32; start += bits) {
      int shift_left = 32 - (start + bits);
      int shift_right = shift_left + start;

      parts.push_back(expand_dims(
          right_shift(
              left_shift(w, array(32 - (start + bits), uint32), s),
              array(32 - bits, uint32),
              s),
          -1,
          s));
    }
    array w_full = concatenate(parts, -1, s);

    // Dequantize
    wshape.push_back(group_size);
    w_full = reshape(w_full, wshape, s);
    w_full = multiply(w_full, expand_dims(scales, -1, s), s);
    w_full = add(w_full, expand_dims(biases, -1, s), s);
    w_full = reshape(w_full, sshape, s);

    return {w_full};
  };

  if (s.device == Device::gpu) {
    auto out_shape = w.shape();
    out_shape.back() = w.shape(-1) * el_per_int;
    return array(
        out_shape,
        scales.dtype(),
        std::make_shared<AffineQuantize>(s, fallback, group_size, bits, true),
        {w, scales, biases});
  }
  return fallback({w, scales, biases})[0];
}

std::string write_signature(
    std::string func_name,
    const std::string& header,
    const std::string& source,
    const std::vector<std::string>& input_names,
    const std::vector<array>& inputs,
    const std::vector<std::string>& output_names,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<std::pair<std::string, TemplateArg>>& template_args,
    const std::vector<std::string>& attributes,
    const std::vector<CustomKernelShapeInfo>& shape_infos,
    bool atomic_outputs) {
  std::string kernel_source;
  kernel_source.reserve(header.size() + source.size() + 16384);
  kernel_source += header;
  // Auto-generate a function signature based on `template_args`
  // and the dtype/shape of the arrays passed as `inputs`.
  if (!template_args.empty()) {
    kernel_source += "template <";
    int i = 0;
    for (const auto& [name, arg] : template_args) {
      std::string param_type;
      if (std::holds_alternative<int>(arg)) {
        param_type = "int";
      } else if (std::holds_alternative<bool>(arg)) {
        param_type = "bool";
      } else if (std::holds_alternative<Dtype>(arg)) {
        param_type = "typename";
      }
      if (i > 0) {
        kernel_source += ", ";
      }
      kernel_source += param_type;
      kernel_source += " ";
      kernel_source += name;
      i++;
    }
    kernel_source += ">\n";
  }
  kernel_source += "[[kernel]] void ";
  kernel_source += func_name;
  kernel_source += "(\n";

  int index = 0;
  constexpr int max_constant_array_size = 8;
  // Add inputs
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& name = input_names[i];
    const auto& arr = inputs[i];
    auto dtype = get_type_string(arr.dtype());
    std::string location =
        arr.size() < max_constant_array_size ? "constant" : "device";
    std::string ref = arr.ndim() == 0 ? "&" : "*";
    kernel_source += "  const ";
    kernel_source += location;
    kernel_source += " ";
    kernel_source += dtype;
    kernel_source += ref;
    kernel_source += " ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]],\n";
    index++;
    // Add input shape, strides and ndim if present in the source
    if (arr.ndim() > 0) {
      if (shape_infos[i].shape) {
        kernel_source +=
            ("  const constant int* " + name + "_shape [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (shape_infos[i].strides) {
        kernel_source +=
            ("  const constant size_t* " + name + "_strides [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
      if (shape_infos[i].ndim) {
        kernel_source +=
            ("  const constant int& " + name + "_ndim [[buffer(" +
             std::to_string(index) + ")]],\n");
        index++;
      }
    }
  }
  // Add outputs
  for (int i = 0; i < output_names.size(); ++i) {
    const auto& name = output_names[i];
    const auto& dtype = output_dtypes[i];
    kernel_source += "  device ";
    auto type_string = get_type_string(dtype);
    if (atomic_outputs) {
      kernel_source += "atomic<";
    }
    kernel_source += type_string;
    if (atomic_outputs) {
      kernel_source += ">";
    }
    kernel_source += "* ";
    kernel_source += name;
    kernel_source += " [[buffer(";
    kernel_source += std::to_string(index);
    kernel_source += ")]]";
    if (index < inputs.size() + output_names.size() - 1 ||
        attributes.size() > 0) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }

  index = 0;
  for (const auto& attr : attributes) {
    kernel_source += attr;
    if (index < attributes.size() - 1) {
      kernel_source += ",\n";
    } else {
      kernel_source += ") {\n";
    }
    index++;
  }
  kernel_source += source;
  kernel_source += "\n}\n";
  return kernel_source;
}

std::string write_template(
    const std::vector<std::pair<std::string, TemplateArg>>& template_args) {
  std::ostringstream template_def;
  template_def << "<";
  int i = 0;
  for (const auto& [name, arg] : template_args) {
    if (i > 0) {
      template_def << ", ";
    }
    if (std::holds_alternative<int>(arg)) {
      template_def << std::get<int>(arg);
    } else if (std::holds_alternative<bool>(arg)) {
      template_def << std::get<bool>(arg);
    } else if (std::holds_alternative<Dtype>(arg)) {
      template_def << get_type_string(std::get<Dtype>(arg));
    }
    i++;
  }
  template_def << ">";
  return template_def.str();
}

MetalKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header /* = "" */,
    bool ensure_row_contiguous /* = true */,
    bool atomic_outputs /* = false */) {
  if (output_names.empty()) {
    throw std::invalid_argument(
        "[metal_kernel] Must specify at least one output.");
  }
  std::vector<CustomKernelShapeInfo> shape_infos;
  for (auto& n : input_names) {
    CustomKernelShapeInfo shape_info;
    shape_info.shape = source.find(n + "_shape") != std::string::npos;
    shape_info.strides = source.find(n + "_strides") != std::string::npos;
    shape_info.ndim = source.find(n + "_ndim") != std::string::npos;
    shape_infos.push_back(shape_info);
  }
  const std::vector<std::pair<std::string, std::string>> metal_attributes = {
      {"dispatch_quadgroups_per_threadgroup", "uint"},
      {"dispatch_simdgroups_per_threadgroup", "uint"},
      {"dispatch_threads_per_threadgroup", "uint3"},
      {"grid_origin", "uint3"},
      {"grid_size", "uint3"},
      {"quadgroup_index_in_threadgroup", "uint"},
      {"quadgroups_per_threadgroup", "uint"},
      {"simdgroup_index_in_threadgroup", "uint"},
      {"simdgroups_per_threadgroup", "uint"},
      {"thread_execution_width", "uint"},
      {"thread_index_in_quadgroup", "uint"},
      {"thread_index_in_simdgroup", "uint"},
      {"thread_index_in_threadgroup", "uint"},
      {"thread_position_in_grid", "uint3"},
      {"thread_position_in_threadgroup", "uint3"},
      {"threadgroup_position_in_grid", "uint3"},
      {"threadgroups_per_grid", "uint3"},
      {"threads_per_grid", "uint3"},
      {"threads_per_simdgroup", "uint"},
      {"threads_per_threadgroup", "uint3"},
  };

  std::vector<std::string> attributes;
  for (const auto& [attr, dtype] : metal_attributes) {
    if (source.find(attr) != std::string::npos) {
      attributes.push_back("  " + dtype + " " + attr + " [[" + attr + "]]");
    }
  }

  return [=,
          shape_infos = std::move(shape_infos),
          attributes = std::move(attributes)](
             const std::vector<array>& inputs,
             const std::vector<std::vector<int>>& output_shapes,
             const std::vector<Dtype>& output_dtypes,
             std::tuple<int, int, int> grid,
             std::tuple<int, int, int> threadgroup,
             const std::vector<std::pair<std::string, TemplateArg>>&
                 template_args = {},
             std::optional<float> init_value = std::nullopt,
             bool verbose = false,
             StreamOrDevice s_ = {}) {
    if (inputs.size() != input_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `inputs` to have size "
          << input_names.size() << " but got size " << inputs.size() << "."
          << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_shapes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_shapes` to have size "
          << output_names.size() << " but got size " << output_shapes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }
    if (output_dtypes.size() != output_names.size()) {
      std::ostringstream msg;
      msg << "[metal_kernel] Expected `output_dtypes` to have size "
          << output_names.size() << " but got size " << output_dtypes.size()
          << "." << std::endl;
      throw std::invalid_argument(msg.str());
    }

    auto s = to_stream(s_);
    if (s.device != Device::gpu) {
      throw std::invalid_argument("[metal_kernel] Only supports the GPU.");
    }

    std::ostringstream func_name;
    std::string template_def = "";
    std::string hash_key = "";
    if (!template_args.empty()) {
      std::regex disallowed_chars("\\<|\\>|(, )");
      template_def = write_template(template_args);
      hash_key = std::regex_replace(template_def, disallowed_chars, "_");
      hash_key.pop_back();
    }
    func_name << "custom_kernel_" << name << hash_key;
    std::string kernel_name = func_name.str();

    std::string kernel_source = write_signature(
        kernel_name,
        header,
        source,
        input_names,
        inputs,
        output_names,
        output_dtypes,
        template_args,
        attributes,
        shape_infos,
        atomic_outputs);

    if (!template_args.empty()) {
      template_def = kernel_name + template_def;
      kernel_source += "\ntemplate [[host_name(\"";
      kernel_source += kernel_name;
      kernel_source += "\")]] [[kernel]] decltype(";
      kernel_source += template_def;
      kernel_source += ") ";
      kernel_source += template_def;
      kernel_source += ";\n";
    }

    if (verbose) {
      std::cout << "Generated source code for `" << name << "`:" << std::endl
                << "```" << std::endl
                << kernel_source << std::endl
                << "```" << std::endl;
    }

    return array::make_arrays(
        std::move(output_shapes),
        std::move(output_dtypes),
        std::make_shared<CustomKernel>(
            s,
            std::move(kernel_name),
            std::move(kernel_source),
            grid,
            threadgroup,
            shape_infos,
            ensure_row_contiguous,
            init_value),
        std::move(inputs));
  };
}

} // namespace mlx::core::fast
