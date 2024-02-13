// Copyright © 2023-2024 Apple Inc.

#include "mlx/extensions.h"
#include "mlx/transforms.h"

namespace mlx::core::ext {

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
  // TODO change to condition for using custom prim
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

} // namespace mlx::core::ext
