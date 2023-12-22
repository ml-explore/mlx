// Copyright © 2023 Apple Inc.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

std::tuple<array, array, int> vmap_binary_op(
    const std::vector<array>& inputs,
    const std::vector<int>& axes,
    const Stream& stream) {
  assert(inputs.size() == 2);
  assert(axes.size() == 2);

  auto a = inputs[0];
  auto b = inputs[1];
  int ndim = std::max(a.ndim() + (axes[0] == -1), b.ndim() + (axes[1] == -1));

  auto expand_dims = [stream, ndim](auto in) {
    auto shape = in.shape();
    shape.insert(shape.begin(), ndim - shape.size(), 1);
    return reshape(in, shape, stream);
  };

  int to_ax = (ndim - a.ndim()) + axes[0];
  int from_ax = (ndim - b.ndim()) + axes[1];
  a = expand_dims(a);
  b = expand_dims(b);

  if (from_ax != to_ax) {
    std::vector<int> tdims(b.ndim());
    std::iota(tdims.begin(), tdims.end(), 0);
    tdims.erase(tdims.begin() + from_ax);
    tdims.insert(tdims.begin() + to_ax, from_ax);
    b = transpose(b, tdims, stream);
  }
  return {a, b, to_ax};
}

} // namespace

array Primitive::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::invalid_argument("Primitive's jvp not implemented.");
};

std::vector<array> Primitive::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  throw std::invalid_argument("Primitive's vjp not implemented.");
};

std::pair<array, int> Primitive::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::invalid_argument("Primitive's vmap not implemented.");
};

std::vector<array> Abs::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Abs::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(tangents[0], sign(primals[0], stream()), stream());
}

std::pair<array, int> Abs::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {abs(inputs[0], stream()), axes[0]};
}

array Add::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return tangents.size() > 1 ? add(tangents[0], tangents[1], stream())
                             : tangents[0];
}

std::vector<array> Add::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  if (argnums.size() == 1) {
    return {cotan};
  } else {
    return {cotan, cotan};
  }
}

std::pair<array, int> Add::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {add(a, b, stream()), to_ax};
}

bool Arange::is_equivalent(const Primitive& other) const {
  const Arange& a_other = static_cast<const Arange&>(other);
  return (
      start_ == a_other.start_ && stop_ == a_other.stop_ &&
      step_ == a_other.step_);
}

std::vector<array> ArcCos::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcCos::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  array denom = negative(rsqrt(t, stream()), stream());
  return multiply(tangents[0], denom, stream());
}

std::pair<array, int> ArcCos::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arccos(inputs[0], stream()), axes[0]};
}

std::vector<array> ArcCosh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcCosh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(square(primals[0], stream()), one, stream());
  return multiply(tangents[0], rsqrt(t, stream()), stream());
}

std::pair<array, int> ArcCosh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arccosh(inputs[0], stream()), axes[0]};
}

std::vector<array> ArcSin::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcSin::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  return multiply(tangents[0], rsqrt(t, stream()), stream());
}

std::pair<array, int> ArcSin::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arcsin(inputs[0], stream()), axes[0]};
}

std::vector<array> ArcSinh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcSinh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = add(square(primals[0], stream()), one, stream());
  return multiply(tangents[0], rsqrt(t, stream()), stream());
}

std::pair<array, int> ArcSinh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arcsinh(inputs[0], stream()), axes[0]};
}

std::vector<array> ArcTan::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcTan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = add(one, square(primals[0], stream()), stream());
  return divide(tangents[0], t, stream());
}

std::pair<array, int> ArcTan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arctan(inputs[0], stream()), axes[0]};
}

std::vector<array> ArcTanh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ArcTanh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  return divide(tangents[0], t, stream());
}

std::pair<array, int> ArcTanh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {arctanh(inputs[0], stream()), axes[0]};
}

std::pair<array, int> ArgPartition::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  return {
      argpartition(inputs[0], axis_ + (axes[0] <= axis_), stream()), axes[0]};
}

bool ArgPartition::is_equivalent(const Primitive& other) const {
  const ArgPartition& r_other = static_cast<const ArgPartition&>(other);
  return axis_ == r_other.axis_ && kth_ == r_other.kth_;
}

bool ArgReduce::is_equivalent(const Primitive& other) const {
  const ArgReduce& r_other = static_cast<const ArgReduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axis_ == r_other.axis_;
}

std::pair<array, int> ArgSort::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  return {argsort(inputs[0], axis_ + (axes[0] <= axis_), stream()), axes[0]};
}

bool ArgSort::is_equivalent(const Primitive& other) const {
  const ArgSort& r_other = static_cast<const ArgSort&>(other);
  return axis_ == r_other.axis_;
}

std::vector<array> AsType::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  if (cotan.dtype() != dtype_) {
    throw std::invalid_argument(
        "[astype] Type of cotangent does not much primal output type.");
  }
  return {astype(cotan, primals[0].dtype(), stream())};
}

array AsType::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return astype(tangents[0], dtype_, stream());
}

std::pair<array, int> AsType::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {astype(inputs[0], dtype_, stream()), axes[0]};
}

bool AsType::is_equivalent(const Primitive& other) const {
  const AsType& a_other = static_cast<const AsType&>(other);
  return dtype_ == a_other.dtype_;
}

std::vector<array> AsStrided::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1);

  // Extract the sizes and cast them to ints
  int grad_size = primals[0].size();
  int cotan_size = cotan.size();

  // Make a flat container to hold the gradients
  auto grad = zeros_like(primals[0], stream());
  grad = reshape(grad, {grad_size}, stream());

  // Create the indices that map output to input
  auto idx = arange(grad_size, stream());
  idx = as_strided(idx, shape_, strides_, offset_, stream());
  idx = reshape(idx, {cotan_size}, stream());

  // Reshape the cotangent for use with scatter
  auto flat_cotan = reshape(cotan, {cotan_size, 1}, stream());

  // Finally accumulate the gradients and reshape them to look like the input
  grad = scatter_add(grad, idx, flat_cotan, 0, stream());
  grad = reshape(grad, primals[0].shape(), stream());

  return {grad};
}

array AsStrided::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);

  return as_strided(tangents[0], shape_, strides_, offset_, stream());
}

bool AsStrided::is_equivalent(const Primitive& other) const {
  const AsStrided& a_other = static_cast<const AsStrided&>(other);
  return shape_ == a_other.shape_ && strides_ == a_other.strides_ &&
      offset_ == a_other.offset_;
}

std::vector<array> Broadcast::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1);

  // Reduce cotan to the shape of the primal
  auto& shape = primals[0].shape();
  int diff = cotan.ndim() - shape.size();
  std::vector<int> reduce_axes;
  for (int i = 0; i < cotan.ndim(); ++i) {
    if (i < diff) {
      reduce_axes.push_back(i);
    } else if (shape[i - diff] != cotan.shape(i)) {
      reduce_axes.push_back(i);
    }
  }
  return {reshape(sum(cotan, reduce_axes, true, stream()), shape, stream())};
}

array Broadcast::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1);
  return broadcast_to(tangents[0], shape_, stream());
}

std::pair<array, int> Broadcast::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto ax = axes[0];
  auto in_shape = inputs[0].shape();
  int diff = shape_.size() - inputs[0].ndim() + 1;
  assert(diff >= 0);
  in_shape.insert(in_shape.begin(), diff, 1);
  ax += diff;
  shape_.insert(shape_.begin() + ax, in_shape[ax]);
  auto in = reshape(inputs[0], in_shape, stream());
  return {broadcast_to(in, shape_, stream()), ax};
}

bool Broadcast::is_equivalent(const Primitive& other) const {
  const Broadcast& b_other = static_cast<const Broadcast&>(other);
  return shape_ == b_other.shape_;
}

std::vector<array> Ceil::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Ceil::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return zeros_like(primals[0], stream());
}

std::pair<array, int> Ceil::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {ceil(inputs[0], stream()), axes[0]};
}

std::vector<array> Concatenate::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<int> start(cotan.ndim(), 0);
  std::vector<int> stop = cotan.shape();

  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : primals) {
    sizes.push_back(p.shape(axis_));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  std::vector<array> grads;
  for (auto i : argnums) {
    start[axis_] = sizes[i];
    stop[axis_] = sizes[i + 1];
    grads.push_back(slice(cotan, start, stop, stream()));
  }
  return grads;
}

array Concatenate::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<int> argidx(argnums.size());
  std::iota(argidx.begin(), argidx.end(), 0);
  std::sort(argidx.begin(), argidx.end(), [&argnums](int a, int b) {
    return argnums[a] < argnums[b];
  });

  std::vector<array> vals;
  for (int i = 0, j = 0; i < primals.size(); ++i) {
    if (j < argnums.size() && argnums[argidx[j]] == i) {
      vals.push_back(tangents[argidx[j++]]);
    } else {
      vals.push_back(zeros_like(primals[i], stream()));
    }
  }
  return concatenate(vals, axis_, stream());
}

std::pair<array, int> Concatenate::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  std::vector<array> t_inputs;
  // Find the first vmapped input
  int i = 0;
  for (; i < axes.size(); i++) {
    t_inputs.push_back(inputs[i]);
    if (axes[i] >= 0) {
      break;
    }
  }
  auto out_ax = axes[i++];
  // Move vmap axes to the same spot.
  for (; i < axes.size(); ++i) {
    if (out_ax != axes[i] && axes[i] >= 0) {
      t_inputs.push_back(moveaxis(inputs[i], axes[i], out_ax, stream()));
    } else {
      t_inputs.push_back(inputs[i]);
    }
  }
  auto axis = axis_ + (axis_ >= out_ax);
  return {concatenate(t_inputs, axis, stream()), out_ax};
}

bool Concatenate::is_equivalent(const Primitive& other) const {
  const Concatenate& c_other = static_cast<const Concatenate&>(other);
  return axis_ == c_other.axis_;
}

std::vector<array> Convolution::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  std::vector<array> grads;

  // Collect info
  auto& in = primals[0];
  auto& wt = primals[1];

  int N = in.shape(0);
  int O = wt.shape(0);

  // Resolve Padded input shapes and strides
  std::vector<int> padding_starts(in.ndim(), 0);
  std::vector<int> padding_ends = in.shape();
  std::vector<int> in_padded_shape = in.shape();

  // padded shape
  for (int i = 1; i < in.ndim() - 1; i++) {
    in_padded_shape[i] += 2 * padding_[i - 1];
    padding_ends[i] += padding_[i - 1];
    padding_starts[i] += padding_[i - 1];
  }

  // padded strides (contiguous)
  std::vector<size_t> in_padded_strides(in.ndim(), 1);
  for (int i = in.ndim() - 2; i >= 0; --i) {
    in_padded_strides[i] = in_padded_strides[i + 1] * in_padded_shape[i + 1];
  }

  // Resolve strided patches

  // patches are shaped as
  // (batch_dim, out_spatial_dims, weight_spatial_dims, in_channels)
  std::vector<int> patches_shape{
      cotan.shape().begin(), cotan.shape().end() - 1};
  patches_shape.insert(
      patches_shape.end(), wt.shape().begin() + 1, wt.shape().end());

  // Resolve patch strides
  int n_spatial_dim = in.ndim() - 2;
  std::vector<size_t> patches_strides(patches_shape.size(), 1);
  patches_strides[0] = in_padded_strides[0];
  for (int i = 1; i < n_spatial_dim + 1; i++) {
    patches_strides[i] = in_padded_strides[i] * kernel_strides_[i - 1];
  }
  for (int i = 1; i < in.ndim(); i++) {
    patches_strides[n_spatial_dim + i] = in_padded_strides[i];
  }

  // Reshape cotan and weights for gemm
  auto cotan_reshaped = reshape(cotan, {-1, O}, stream());
  auto weight_reshaped = reshape(wt, {O, -1}, stream());

  for (int a : argnums) {
    // Grads for input
    if (a == 0) {
      // Gemm with cotan to get patches
      auto grad_patches = matmul(cotan_reshaped, weight_reshaped, stream());

      // Prepare base grad array to accumulate on
      int in_padded_size = in_padded_strides[0] * in_padded_shape[0];
      auto grad = zeros(
          {
              in_padded_size,
          },
          in.dtype(),
          stream());

      // Create index map
      int patches_size = grad_patches.size();
      auto idx = arange(in_padded_size, stream());
      idx = as_strided(idx, patches_shape, patches_strides, 0, stream());
      idx = reshape(idx, {patches_size}, stream());

      // Flatten patches and scatter
      auto flat_patches = reshape(grad_patches, {patches_size, 1}, stream());
      grad = scatter_add(grad, idx, flat_patches, 0, stream());

      // Reshape and slice away padding
      grad = reshape(grad, in_padded_shape, stream());
      grad = slice(grad, padding_starts, padding_ends, stream());

      grads.push_back(grad);
    }
    // Grads for weight
    else if (a == 1) {
      // Make patches from in
      std::vector<int> padded_axes(in.ndim() - 2, 0);
      std::iota(padded_axes.begin(), padded_axes.end(), 1);
      auto in_padded = pad(
          in, padded_axes, padding_, padding_, array(0, in.dtype()), stream());
      auto in_patches =
          as_strided(in_padded, patches_shape, patches_strides, 0, stream());
      in_patches = reshape(in_patches, {cotan_reshaped.shape(0), -1}, stream());

      auto grad = matmul(
          transpose(cotan_reshaped, {1, 0}, stream()), in_patches, stream());
      grad = reshape(grad, wt.shape(), stream());
      grads.push_back(grad);
    }
  }

  return grads;
}

bool Convolution::is_equivalent(const Primitive& other) const {
  const Convolution& c_other = static_cast<const Convolution&>(other);
  return padding_ == c_other.padding_ &&
      kernel_strides_ == c_other.kernel_strides_ &&
      kernel_dilation_ == c_other.kernel_dilation_ &&
      input_dilation_ == c_other.input_dilation_;
}

std::vector<array> Copy::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {cotan};
}

array Copy::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents[0];
}

std::pair<array, int> Copy::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {copy(inputs[0], stream()), axes[0]};
}

std::vector<array> Cos::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Cos::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(
      tangents[0], negative(sin(primals[0], stream()), stream()), stream());
}

std::pair<array, int> Cos::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {cos(inputs[0], stream()), axes[0]};
}

std::vector<array> Cosh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Cosh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(tangents[0], sinh(primals[0], stream()), stream());
}

std::pair<array, int> Cosh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {cosh(inputs[0], stream()), axes[0]};
}

std::vector<array> Divide::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(divide(cotan, primals[1], stream()));
    } else {
      vjps.push_back(negative(
          divide(
              multiply(cotan, primals[0], stream()),
              square(primals[1], stream()),
              stream()),
          stream()));
    }
  }
  return vjps;
}

array Divide::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    if (arg == 0) {
      return divide(tangents[i], primals[1], stream());
    } else {
      return negative(
          divide(
              multiply(tangents[i], primals[0], stream()),
              square(primals[1], stream()),
              stream()),
          stream());
    }
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> Divide::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {divide(a, b, stream()), to_ax};
}

std::vector<array> Remainder::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(cotan);
    } else {
      auto x_over_y = divide(primals[0], primals[1], stream());
      x_over_y = floor(x_over_y, stream());
      vjps.push_back(negative(multiply(x_over_y, cotan, stream()), stream()));
    }
  }
  return vjps;
}

array Remainder::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    if (arg == 0) {
      return tangents[i];
    } else {
      auto x_over_y = divide(primals[0], primals[1], stream());
      x_over_y = floor(x_over_y, stream());
      return negative(multiply(x_over_y, tangents[i], stream()), stream());
    }
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> Remainder::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {remainder(a, b, stream()), to_ax};
}

std::pair<array, int> Equal::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {equal(a, b, stream()), axes[0]};
}

std::vector<array> Equal::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array Equal::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::vector<array> Erf::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Erf::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(M_2_SQRTPI, dtype), tangents[0], stream());
  return multiply(
      scale,
      exp(negative(square(primals[0], stream()), stream()), stream()),
      stream());
}

std::pair<array, int> Erf::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {erf(inputs[0], stream()), axes[0]};
}

std::vector<array> ErfInv::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array ErfInv::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(1.0 / M_2_SQRTPI, dtype), tangents[0], stream());
  return multiply(
      scale,
      exp(square(erfinv(primals[0], stream()), stream()), stream()),
      stream());
}

std::pair<array, int> ErfInv::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {erfinv(inputs[0], stream()), axes[0]};
}

std::vector<array> Exp::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Exp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(tangents[0], exp(primals[0], stream()), stream());
}

std::pair<array, int> Exp::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {exp(inputs[0], stream()), axes[0]};
}

bool FFT::is_equivalent(const Primitive& other) const {
  const FFT& r_other = static_cast<const FFT&>(other);
  return axes_ == r_other.axes_ && inverse_ == r_other.inverse_ &&
      real_ == r_other.real_;
}

std::pair<array, int> FFT::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& in = inputs[0];
  int ax = axes[0];
  auto fft_axes = axes_;
  auto out_shape = in.shape();
  for (auto& fft_ax : fft_axes) {
    if (fft_ax >= ax) {
      fft_ax++;
    }
    if (real_) {
      auto n = out_shape[fft_ax];
      out_shape[fft_ax] = inverse_ ? 2 * (n - 1) : n / 2 + 1;
    }
  }
  return {
      array(
          out_shape,
          real_ && inverse_ ? float32 : complex64,
          std::make_unique<FFT>(stream(), fft_axes, inverse_, real_),
          {in}),
      ax};
}

std::vector<array> FFT::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto& in = primals[0];
  std::vector<int> axes(axes_.begin(), axes_.end());
  if (real_ && inverse_) {
    auto out = fft::fftn(cotan, axes, stream());
    auto start = std::vector<int>(out.ndim(), 0);
    auto stop = in.shape();
    out = slice(out, start, stop, stream());
    auto mask_shape = out.shape();
    mask_shape[axes_.back()] -= 2;
    auto mask = full(mask_shape, 2.0f, stream());
    auto pad_shape = out.shape();
    pad_shape[axes_.back()] = 1;
    auto pad = full(pad_shape, 1.0f, stream());
    mask = concatenate({pad, mask, pad}, axes_.back(), stream());
    return {multiply(mask, out, stream())};
  } else if (real_) {
    std::vector<int> n;
    for (auto ax : axes_) {
      n.push_back(in.shape()[ax]);
    }
    return {astype(fft::fftn(cotan, n, axes, stream()), in.dtype(), stream())};
  } else if (inverse_) {
    return {fft::ifftn(cotan, axes, stream())};
  } else {
    return {fft::fftn(cotan, axes, stream())};
  }
}

array FFT::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto& tan = tangents[0];
  if (real_ & inverse_) {
    return fft::irfftn(tan, stream());
  } else if (real_) {
    return fft::rfftn(tan, stream());
  } else if (inverse_) {
    return fft::ifftn(tan, stream());
  } else {
    return fft::fftn(tan, stream());
  }
}

std::vector<array> Floor::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Floor::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return zeros_like(primals[0], stream());
}

std::pair<array, int> Floor::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {floor(inputs[0], stream()), axes[0]};
}

std::vector<array> Full::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(cotan, primals[0], stream())};
}

array Full::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents[0];
}

std::pair<array, int> Full::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto& in = inputs[0];
  auto out =
      array(in.shape(), in.dtype(), std::make_unique<Full>(stream()), {in});
  return {out, axes[0]};
}

std::pair<array, int> Gather::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& src = inputs[0];
  std::vector<array> indices(inputs.begin() + 1, inputs.end());
  auto gather_axes = axes_;
  auto slice_sizes = slice_sizes_;
  auto src_vmapped = axes[0] >= 0;
  auto indices_vmapped =
      std::any_of(axes.begin() + 1, axes.end(), [](int a) { return a >= 0; });
  auto out_ax =
      *std::find_if(axes.begin(), axes.end(), [](int a) { return a >= 0; });

  // Reorder all the index arrays so the vmap axis is in the same spot.
  for (int i = 1; i < axes.size(); ++i) {
    if (out_ax != axes[i] && axes[i] >= 0) {
      indices[i - 1] = moveaxis(indices[i - 1], axes[i], out_ax, stream());
    }
  }

  if (src_vmapped) {
    int max_dims = 0;
    for (auto& idx : indices) {
      max_dims = std::max(static_cast<int>(idx.ndim()), max_dims);
    }
    auto new_ax_loc =
        std::find_if(gather_axes.begin(), gather_axes.end(), [&out_ax](int a) {
          return a >= out_ax;
        });
    for (; new_ax_loc < gather_axes.end(); new_ax_loc++) {
      (*new_ax_loc)++;
    }
    if (indices_vmapped) {
      // Make a new index array for the vmapped dimension
      // Reshape it so it broadcasts with other index arrays
      // Update gather axes and slice sizes accordingly
      auto shape = std::vector<int>(max_dims - out_ax, 1);
      auto vmap_inds = arange(0, src.shape(out_ax), stream());
      shape[0] = vmap_inds.shape(0);
      vmap_inds = reshape(vmap_inds, shape, stream());
      slice_sizes.insert(slice_sizes.begin() + out_ax, 1);
      auto new_ax_idx = new_ax_loc - gather_axes.begin();
      gather_axes.insert(new_ax_loc, out_ax);
      indices.insert(indices.begin() + new_ax_idx, vmap_inds);
    } else {
      slice_sizes.insert(slice_sizes.begin() + axes[0], src.shape(axes[0]));
      out_ax = max_dims + axes[0];
    }
  }
  return {gather(src, indices, gather_axes, slice_sizes, stream()), out_ax};
}

std::vector<array> Gather::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[gather] Cannot calculate VJP with respect to indices.");
  }
  auto src = zeros_like(primals[0], stream());
  std::vector<array> inds(primals.begin() + 1, primals.end());
  return {scatter_add(src, inds, cotan, axes_, stream())};
}

array Gather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[gather] Cannot calculate JVP with respect to indices.");
  }
  std::vector<array> inds(primals.begin() + 1, primals.end());
  return gather(tangents[0], inds, axes_, slice_sizes_, stream());
}

bool Gather::is_equivalent(const Primitive& other) const {
  const Gather& g_other = static_cast<const Gather&>(other);
  return axes_ == g_other.axes_ && slice_sizes_ == g_other.slice_sizes_;
}

std::pair<array, int> Greater::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {greater(a, b, stream()), axes[0]};
}

std::vector<array> Greater::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array Greater::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::pair<array, int> GreaterEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {greater_equal(a, b, stream()), axes[0]};
}

std::vector<array> GreaterEqual::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array GreaterEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::pair<array, int> Less::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {less(a, b, stream()), axes[0]};
}

std::vector<array> Less::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array Less::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::pair<array, int> LessEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {less_equal(a, b, stream()), axes[0]};
}

std::vector<array> LessEqual::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array LessEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::vector<array> Log::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Log::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto out = divide(tangents[0], primals[0], stream());
  if (base_ != Base::e) {
    auto scale = 1 / std::log(base_ == Base::ten ? 10.0f : 2.0f);
    out = multiply(array(scale, out.dtype()), out, stream());
  }
  return out;
}

std::pair<array, int> Log::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto& in = inputs[0];
  return {
      array(
          in.shape(), in.dtype(), std::make_unique<Log>(stream(), base_), {in}),
      axes[0]};
}

std::vector<array> Log1p::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Log1p::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  return divide(
      tangents[0], add(array(1.0f, dtype), primals[0], stream()), stream());
}

std::pair<array, int> Log1p::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {log1p(inputs[0], stream()), axes[0]};
}

std::vector<array> LogicalNot::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array LogicalNot::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return zeros_like(tangents[0], stream());
}

std::pair<array, int> LogicalNot::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {logical_not(inputs[0], stream()), axes[0]};
}

std::vector<array> LogAddExp::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  auto a = primals[0];
  auto b = primals[1];
  auto s = sigmoid(subtract(a, b, stream()), stream());
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(
        cotan,
        arg == 0 ? s : subtract(array(1.0f, s.dtype()), s, stream()),
        stream()));
  }
  return vjps;
}

array LogAddExp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto a = primals[0];
  auto b = primals[1];
  auto s = sigmoid(subtract(a, b, stream()), stream());
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    return multiply(
        tangents[i],
        arg == 0 ? s : subtract(array(1.0f, s.dtype()), s, stream()),
        stream());
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> LogAddExp::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {logaddexp(a, b, stream()), to_ax};
}

std::vector<array> Matmul::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  std::vector<int> reorder(cotan.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  std::iter_swap(reorder.end() - 1, reorder.end() - 2);
  for (auto arg : argnums) {
    if (arg == 0) {
      // M X N * (K X N).T -> M X K
      vjps.push_back(
          matmul(cotan, transpose(primals[1], reorder, stream()), stream()));
    } else {
      // (M X K).T * M X N -> K X N
      vjps.push_back(
          matmul(transpose(primals[0], reorder, stream()), cotan, stream()));
    }
  }
  return vjps;
}

std::pair<array, int> Matmul::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {array(1.0), 0};
}

std::vector<array> Maximum::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask =
        (arg == 0) ? greater(a, b, stream()) : less_equal(a, b, stream());
    vjps.push_back(multiply(cotan, mask, stream()));
  }
  return vjps;
}

array Maximum::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    auto mask =
        (arg == 0) ? greater(a, b, stream()) : less_equal(a, b, stream());
    return multiply(tangents[i], mask, stream());
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> Maximum::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {maximum(a, b, stream()), to_ax};
}

std::vector<array> Minimum::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask =
        (arg == 0) ? less(a, b, stream()) : greater_equal(a, b, stream());
    vjps.push_back(multiply(cotan, mask, stream()));
  }
  return vjps;
}

array Minimum::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto& a = primals[0];
  auto& b = primals[1];
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    auto mask =
        (arg == 0) ? less(a, b, stream()) : greater_equal(a, b, stream());
    return multiply(tangents[i], mask, stream());
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> Minimum::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {minimum(a, b, stream()), to_ax};
}

array Multiply::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto arg = argnums[0];
  auto jvp = multiply(tangents[0], primals[1 - arg], stream());
  if (argnums.size() > 1) {
    arg = argnums[1];
    jvp = add(jvp, multiply(tangents[1], primals[1 - arg], stream()), stream());
  }
  return jvp;
}

std::vector<array> Multiply::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(primals[1 - arg], cotan, stream()));
  }
  return vjps;
}

std::pair<array, int> Multiply::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {multiply(a, b, stream()), to_ax};
}

std::vector<array> Negative::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Negative::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return negative(tangents[0], stream());
}

std::pair<array, int> Negative::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {negative(inputs[0], stream()), axes[0]};
}

std::pair<array, int> NotEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {not_equal(a, b, stream()), axes[0]};
}

std::vector<array> NotEqual::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

array NotEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return zeros(shape, bool_, stream());
}

std::vector<array> Pad::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  std::vector<int> start(cotan.ndim(), 0);
  std::vector<int> stop = cotan.shape();

  for (auto i : axes_) {
    start[i] = low_pad_size_[i];
    stop[i] -= high_pad_size_[i];
  }

  auto out = slice(cotan, start, stop, stream());

  return {out};
}

array Pad::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  return pad(
      tangents[0],
      axes_,
      low_pad_size_,
      high_pad_size_,
      array(0, tangents[0].dtype()),
      stream());
}

std::pair<array, int> Pad::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("Pad vmap is NYI.");
}

bool Pad::is_equivalent(const Primitive& other) const {
  const Pad& p_other = static_cast<const Pad&>(other);
  return (
      p_other.axes_ == axes_ && p_other.low_pad_size_ == low_pad_size_ &&
      p_other.high_pad_size_ == high_pad_size_);
}

std::vector<array> Partition::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Partition::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto sort_idx = argpartition(primals[0], kth_, axis_, stream());
  auto out = take_along_axis(tangents[0], sort_idx, axis_, stream());
  return out;
}

std::pair<array, int> Partition::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  return {partition(inputs[0], axis_ + (axes[0] <= axis_), stream()), axes[0]};
}

bool Partition::is_equivalent(const Primitive& other) const {
  const Partition& r_other = static_cast<const Partition&>(other);
  return axis_ == r_other.axis_ && kth_ == r_other.kth_;
}

std::vector<array> Power::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(multiply(
          power(
              primals[0],
              subtract(primals[1], array(1, primals[0].dtype()), stream()),
              stream()),
          primals[1],
          stream()));
    } else {
      vjps.push_back(multiply(
          log(primals[0], stream()),
          power(primals[0], primals[1], stream()),
          stream()));
    }
    vjps.back() = multiply(cotan, vjps.back(), stream());
  }
  return vjps;
}

array Power::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp = vjp(primals, tangents[0], {argnums[0]})[0];
  if (argnums.size() > 1) {
    jvp = add(jvp, vjp(primals, tangents[1], {argnums[1]})[0], stream());
  }
  return jvp;
}

std::pair<array, int> Power::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {power(a, b, stream()), to_ax};
}

std::pair<array, int> QuantizedMatmul::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("QuantizedMatmul::vmap NYI");
}

std::vector<array> QuantizedMatmul::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  throw std::runtime_error("QuantizedMatmul::vjp NYI");
}

array QuantizedMatmul::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("QuantizedMatmul::vjp NYI");
}

bool QuantizedMatmul::is_equivalent(const Primitive& other) const {
  const QuantizedMatmul& qm_other = static_cast<const QuantizedMatmul&>(other);
  return group_size_ == qm_other.group_size_ && bits_ == qm_other.bits_;
}

std::pair<array, int> RandomBits::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  // The last dimension of the key is always a key pair
  auto key = inputs[0];
  auto kax = axes[0];
  if (kax == key.ndim() - 1) {
    std::vector<int> reorder(key.ndim());
    std::iota(reorder.begin(), reorder.end(), 0);
    std::swap(reorder[kax], reorder[kax - 1]);
    key = transpose(key, reorder, stream());
    kax--;
  }

  auto shape = shape_;
  shape.insert(shape.begin() + kax, key.shape()[kax]);

  auto get_dtype = [width = width_]() {
    switch (width) {
      case 1:
        return uint8;
      case 2:
        return uint16;
      default:
        return uint32;
    }
  };

  auto out = array(
      shape,
      get_dtype(),
      std::make_unique<RandomBits>(stream(), shape, width_),
      {key});
  return {out, kax};
}

bool RandomBits::is_equivalent(const Primitive& other) const {
  const RandomBits& r_other = static_cast<const RandomBits&>(other);
  return shape_ == r_other.shape_;
}

std::pair<array, int> Reshape::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  // Transpose the input so that the vmap dim is first.
  auto& in = inputs[0];
  auto ax = axes[0];
  std::vector<int> reorder(in.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  reorder.erase(reorder.begin() + ax);
  reorder.insert(reorder.begin(), ax);
  // Insert the vmap dim into the shape at the beginning.
  auto out = transpose(in, reorder, stream());
  shape_.insert(shape_.begin(), in.shape()[ax]);
  // Reshape the transposed input to the new shape.
  return {reshape(out, shape_, stream()), 0};
}

std::vector<array> Reshape::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return {reshape(cotan, primals[0].shape(), stream())};
}

array Reshape::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return reshape(tangents[0], shape_, stream());
}

bool Reshape::is_equivalent(const Primitive& other) const {
  const Reshape& r_other = static_cast<const Reshape&>(other);
  return shape_ == r_other.shape_;
}

std::vector<array> Reduce::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  auto in = primals[0];

  std::vector<int> shape = in.shape();
  for (auto ax : axes_) {
    shape[ax] = 1;
  }

  if (reduce_type_ == Reduce::Sum) {
    return {
        broadcast_to(reshape(cotan, shape, stream()), in.shape(), stream())};
  } else if (reduce_type_ == Reduce::Prod) {
    auto s = stream();
    auto prod_grad_single_axis =
        [&s](const array& x, const array& cotan, int axis) {
          auto p1 = cumprod(x, axis, /*reverse=*/false, /*inclusive=*/false, s);
          auto p2 = cumprod(x, axis, /*reverse=*/true, /*inclusive=*/false, s);
          auto exclusive_prod = multiply(p1, p2, s);
          return multiply(exclusive_prod, cotan, s);
        };

    // To compute a numerically stable gradient for prod we need an exclusive
    // product of all elements in axes_ . To achieve that we move axes_ to the
    // last dim and perform two exclusive cumprods. Afterwards we move
    // everything back to the original axes.
    if (axes_.size() > 1) {
      std::vector<int> transpose_to;
      std::vector<int> transpose_back;
      std::vector<int> shape_flat;
      {
        // Find the transpose needed to move axes_ to the back and the shape
        // except the reduced over axes.
        int j = 0;
        for (int i = 0; i < in.ndim(); i++) {
          if (j < axes_.size() && axes_[j] == i) {
            j++;
          } else {
            transpose_to.push_back(i);
            shape_flat.push_back(in.shape(i));
          }
        }
        for (auto ax : axes_) {
          transpose_to.push_back(ax);
        }
        shape_flat.push_back(-1);
        transpose_back.resize(transpose_to.size());
        for (int i = 0; i < transpose_to.size(); i++) {
          transpose_back[transpose_to[i]] = i;
        }
      }

      // Move axes to the back
      auto x = transpose(in, transpose_to, s);
      // Keep the shape in order to reshape back to the original
      auto shape_to = x.shape();

      // Flatten and compute the gradient
      x = reshape(x, shape_flat, stream());
      auto grad = prod_grad_single_axis(x, reshape(cotan, shape_flat, s), -1);

      // Reshape and transpose to the original shape
      grad = reshape(grad, shape_to, s);
      grad = transpose(grad, transpose_back, s);

      return {grad};
    } else {
      return {prod_grad_single_axis(in, reshape(cotan, shape, s), axes_[0])};
    }

  } else if (reduce_type_ == Reduce::Min || reduce_type_ == Reduce::Max) {
    array (*op)(const array&, const std::vector<int>&, bool, StreamOrDevice);

    if (reduce_type_ == Reduce::Min) {
      op = min;
    } else {
      op = max;
    }

    auto out = op(in, axes_, true, stream());
    auto mask = equal(in, out, stream());
    auto normalizer = sum(mask, axes_, true, stream());
    auto cotan_reshape = reshape(cotan, shape, stream());
    cotan_reshape = divide(cotan_reshape, normalizer, stream());
    return {multiply(cotan_reshape, mask, stream())};
  }

  else {
    throw std::runtime_error("Reduce type VJP not yet implemented.");
  }
}

std::pair<array, int> Reduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  // TODO implement
  return {array(1.0f), axes[0]};
}

bool Reduce::is_equivalent(const Primitive& other) const {
  const Reduce& r_other = static_cast<const Reduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axes_ == r_other.axes_;
}

std::vector<array> Round::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Round::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return zeros_like(primals[0], stream());
}

std::pair<array, int> Round::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {round(inputs[0], stream()), axes[0]};
}

std::pair<array, int> Scan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& in = inputs[0];
  auto axis = axes[0];

  auto out_dtype =
      (in.dtype() == bool_ && reduce_type_ == Scan::Sum) ? int32 : in.dtype();
  return {
      array(
          in.shape(),
          out_dtype,
          std::make_unique<Scan>(
              stream(),
              reduce_type_,
              axis_ + (axis <= axis_),
              reverse_,
              inclusive_),
          {in}),
      axis};
}

std::vector<array> Scan::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return {cumsum(cotan, axis_, !reverse_, inclusive_, stream())};
  } else if (reduce_type_ == Scan::Prod) {
    // TODO: Make it numerically stable when we introduce where()
    auto prod = cumprod(primals[0], axis_, reverse_, inclusive_, stream());
    auto partial_grads = multiply(prod, cotan, stream());
    auto accum_grads =
        cumsum(partial_grads, axis_, !reverse_, inclusive_, stream());
    return {divide(accum_grads, primals[0], stream())};
  } else {
    // Can probably be implemented by equals and then cummax to make the mask
    throw std::runtime_error("VJP is not implemented for cumulative min/max");
  }
}

array Scan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(tangents.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return cumsum(tangents[0], axis_, reverse_, inclusive_, stream());
  } else {
    throw std::runtime_error(
        "JVP is not implemented for cumulative prod/min/max");
  }
}

bool Scan::is_equivalent(const Primitive& other) const {
  const Scan& s_other = static_cast<const Scan&>(other);
  return (
      reduce_type_ == s_other.reduce_type_ && axis_ == s_other.axis_ &&
      reverse_ == s_other.reverse_ && inclusive_ == s_other.inclusive_);
}

bool Scatter::is_equivalent(const Primitive& other) const {
  const Scatter& s_other = static_cast<const Scatter&>(other);
  return reduce_type_ == s_other.reduce_type_ && axes_ == s_other.axes_;
}

std::vector<array> Sigmoid::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sigmoid::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto s = sigmoid(primals[0], stream());
  auto sprime =
      multiply(s, subtract(array(1.0f, s.dtype()), s, stream()), stream());
  return multiply(tangents[0], sprime, stream());
}

std::pair<array, int> Sigmoid::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {sigmoid(inputs[0], stream()), axes[0]};
}

std::vector<array> Sign::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sign::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return zeros(primals[0].shape(), primals[0].dtype(), stream());
}

std::pair<array, int> Sign::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {sign(inputs[0], stream()), axes[0]};
}

std::vector<array> Sin::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sin::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(tangents[0], cos(primals[0], stream()), stream());
}

std::pair<array, int> Sin::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {sin(inputs[0], stream()), axes[0]};
}

std::vector<array> Sinh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sinh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return multiply(tangents[0], cosh(primals[0], stream()), stream());
}

std::pair<array, int> Sinh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {sinh(inputs[0], stream()), axes[0]};
}

std::pair<array, int> Slice::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto start = start_indices_;
  auto stop = end_indices_;
  auto strides = strides_;
  auto ax = axes[0];
  auto& input = inputs[0];
  start.insert(start.begin() + ax, 0);
  stop.insert(stop.begin() + ax, input.shape(ax));
  strides.insert(strides.begin() + ax, 1);
  return {slice(input, start, stop, strides, stream()), ax};
}

std::vector<array> Slice::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  // Check inputs
  assert(primals.size() == 1);

  std::vector<array> inds;
  std::vector<int> ind_axes;
  std::vector<array> single_inds;
  std::vector<int> single_ind_axes;
  for (int i = 0; i < start_indices_.size(); ++i) {
    auto start = start_indices_[i];
    auto end = end_indices_[i];
    auto stride = strides_[i];
    if (start == 0 && stride == 1) {
      continue;
    }
    if (stride == 1) {
      single_inds.push_back(array(start));
      single_ind_axes.push_back(i);
    } else {
      inds.push_back(arange(start, end, stride, stream()));
      ind_axes.push_back(i);
    }
  }

  // Transpose and reshape cotan
  auto cotan_ = cotan;
  if (!ind_axes.empty()) {
    std::vector<int> cotan_shape;
    for (auto ax : ind_axes) {
      cotan_shape.push_back(cotan.shape(ax));
    }
    std::vector<int> cotan_axes(ind_axes);
    for (int j = 0, i = 0; i < cotan.ndim(); ++i) {
      if (j < ind_axes.size() && ind_axes[j] == i) {
        cotan_shape.push_back(1);
        j++;
      } else {
        cotan_shape.push_back(cotan.shape(i));
        cotan_axes.push_back(i);
      }
    }
    cotan_ =
        reshape(transpose(cotan_, cotan_axes, stream()), cotan_shape, stream());
  }

  // Make indices broadcastable
  std::vector<int> inds_shape(inds.size(), 1);
  for (int i = 0; i < inds.size(); ++i) {
    inds_shape[i] = inds[i].size();
    inds[i] = reshape(inds[i], inds_shape, stream());
    inds_shape[i] = 1;
  }

  // Concatenate all the indices and axes
  inds.insert(inds.end(), single_inds.begin(), single_inds.end());
  ind_axes.insert(
      ind_axes.end(), single_ind_axes.begin(), single_ind_axes.end());

  return {scatter_add(
      zeros_like(primals[0], stream()), inds, cotan_, ind_axes, stream())};
}

array Slice::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // Check inputs
  assert(primals.size() == 1);
  return slice(tangents[0], start_indices_, end_indices_, strides_, stream());
}

bool Slice::is_equivalent(const Primitive& other) const {
  const Slice& s_other = static_cast<const Slice&>(other);
  return (
      start_indices_ == s_other.start_indices_ &&
      end_indices_ == s_other.end_indices_ && strides_ == s_other.strides_);
}

std::pair<array, int> Softmax::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  std::vector<int> softmax_axes;

  // We are vectorizing over an axis other than the last one so keep the
  // softmax axis unchanged
  if (axes[0] < inputs[0].ndim() - 1) {
    softmax_axes.push_back(-1);
  } else {
    softmax_axes.push_back(-2);
  }
  return {softmax(inputs[0], softmax_axes, stream()), axes[0]};
}

std::vector<array> Softmax::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Softmax::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto s = softmax(primals[0], std::vector<int>{-1}, stream());
  auto sv = multiply(s, tangents[0], stream());
  return subtract(
      sv, multiply(s, sum(sv, std::vector<int>{-1}, true, stream()), stream()));
}

std::pair<array, int> Sort::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  return {sort(inputs[0], axis_ + (axes[0] <= axis_), stream()), axes[0]};
}

std::vector<array> Sort::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sort::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto sort_idx = argsort(primals[0], axis_, stream());
  auto out = take_along_axis(tangents[0], sort_idx, axis_, stream());
  return out;
}

bool Sort::is_equivalent(const Primitive& other) const {
  const Sort& r_other = static_cast<const Sort&>(other);
  return axis_ == r_other.axis_;
}

std::vector<array> Square::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Square::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return multiply(
      primals[0],
      multiply(array(2, primals[0].dtype()), tangents[0], stream()),
      stream());
}

std::pair<array, int> Square::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {square(inputs[0], stream()), axes[0]};
}

std::vector<array> Sqrt::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Sqrt::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto dtype = primals[0].dtype();
  if (recip_) {
    auto one_over_x_root_x =
        divide(rsqrt(primals[0], stream()), primals[0], stream());
    return multiply(
        multiply(array(-0.5, dtype), tangents[0], stream()),
        one_over_x_root_x,
        stream());
  }
  return divide(
      multiply(array(0.5, dtype), tangents[0], stream()),
      sqrt(primals[0], stream()),
      stream());
}

std::pair<array, int> Sqrt::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  if (recip_)
    return {rsqrt(inputs[0], stream()), axes[0]};

  return {sqrt(inputs[0], stream()), axes[0]};
}

bool Sqrt::is_equivalent(const Primitive& other) const {
  const Sqrt& s_other = static_cast<const Sqrt&>(other);
  return recip_ == s_other.recip_;
}

std::pair<array, int> StopGradient::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {inputs[0], axes[0]};
};

std::vector<array> Subtract::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto vjp = cotan;
    if (arg == 1) {
      vjp = negative(vjp, stream());
    }
    vjps.push_back(vjp);
  }
  return vjps;
}

array Subtract::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto jvp_fun = [&](int i) {
    int arg = argnums[i];
    return arg == 1 ? negative(tangents[i], stream()) : tangents[i];
  };
  auto out = jvp_fun(0);
  if (argnums.size() > 1) {
    out = add(out, jvp_fun(1), stream());
  }
  return out;
}

std::pair<array, int> Subtract::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {subtract(a, b, stream()), to_ax};
}

std::vector<array> Tan::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Tan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cos_sq = square(cos(primals[0], stream()), stream());
  return divide(tangents[0], cos_sq, stream());
}

std::pair<array, int> Tan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {tan(inputs[0], stream()), axes[0]};
}

std::vector<array> Tanh::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  return {jvp(primals, {cotan}, argnums)};
}

array Tanh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cosh_sq = square(cosh(primals[0], stream()), stream());
  return divide(tangents[0], cosh_sq, stream());
}

std::pair<array, int> Tanh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {tanh(inputs[0], stream()), axes[0]};
}

std::vector<array> Transpose::vjp(
    const std::vector<array>& primals,
    const array& cotan,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  std::vector<int> iaxes(axes_.size());
  for (int i = 0; i < axes_.size(); ++i) {
    iaxes[axes_[i]] = i;
  }
  return {transpose(cotan, iaxes, stream())};
}

array Transpose::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return transpose(tangents[0], axes_, stream());
}

std::pair<array, int> Transpose::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto vdim = axes[0];
  for (auto& dim : axes_) {
    if (dim >= vdim) {
      dim++;
    }
  }
  axes_.insert(axes_.begin() + vdim, vdim);
  return {transpose(inputs[0], axes_, stream()), vdim};
}

bool Transpose::is_equivalent(const Primitive& other) const {
  const Transpose& t_other = static_cast<const Transpose&>(other);
  return axes_ == t_other.axes_;
}

} // namespace mlx::core
