// Copyright © 2023-2024 Apple Inc.

// Required for using M_2_SQRTPI in MSVC.
#define _USE_MATH_DEFINES

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/common/utils.h"
#include "mlx/fft.h"
#include "mlx/linalg.h"
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

  if (axes[0] == -1 && axes[1] == -1) {
    return {inputs[0], inputs[1], -1};
  }

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

std::tuple<array, array, array, int> vmap_ternary_op(
    const std::vector<array>& inputs,
    const std::vector<int>& axes,
    const Stream& stream) {
  assert(inputs.size() == 3);
  assert(axes.size() == 3);

  if (axes[0] == -1 && axes[1] == -1 && axes[2] == -1) {
    return {inputs[0], inputs[1], inputs[2], -1};
  }

  auto a = inputs[0];
  auto b = inputs[1];
  auto c = inputs[2];
  int ndim = std::max(
      {a.ndim() + (axes[0] == -1),
       b.ndim() + (axes[1] == -1),
       c.ndim() + (axes[2] == -1)});

  auto expand_dims = [stream, ndim](auto in) {
    auto shape = in.shape();
    shape.insert(shape.begin(), ndim - shape.size(), 1);
    return reshape(in, shape, stream);
  };

  int to_ax = (ndim - a.ndim()) + axes[0];
  int from_ax1 = (ndim - b.ndim()) + axes[1];
  int from_ax2 = (ndim - c.ndim()) + axes[2];
  a = expand_dims(a);
  b = expand_dims(b);
  c = expand_dims(c);

  auto find_tdims = [](auto x, int to_ax, int from_ax) {
    std::vector<int> tdims(x.ndim());
    std::iota(tdims.begin(), tdims.end(), 0);
    tdims.erase(tdims.begin() + from_ax);
    tdims.insert(tdims.begin() + to_ax, from_ax);
    return tdims;
  };

  if (to_ax != from_ax1) {
    std::vector<int> tdims = find_tdims(b, to_ax, from_ax1);
    b = transpose(b, tdims, stream);
  }

  if (to_ax != from_ax2) {
    std::vector<int> tdims = find_tdims(c, to_ax, from_ax2);
    c = transpose(c, tdims, stream);
  }
  return {a, b, c, to_ax};
}

} // namespace

std::vector<array> Primitive::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  std::ostringstream msg;
  msg << "[Primitive::jvp] Not implemented for ";
  print(msg);
  msg << ".";
  throw std::invalid_argument(msg.str());
}

std::vector<array> Primitive::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  std::ostringstream msg;
  msg << "[Primitive::vjp] Not implemented for ";
  print(msg);
  msg << ".";
  throw std::invalid_argument(msg.str());
}

std::pair<std::vector<array>, std::vector<int>> Primitive::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  std::ostringstream msg;
  msg << "[Primitive::vmap] Not implemented for ";
  print(msg);
  msg << ".";
  throw std::invalid_argument(msg.str());
}

std::vector<Shape> Primitive::output_shapes(const std::vector<array>&) {
  std::ostringstream msg;
  msg << "[Primitive::output_shapes] ";
  this->print(msg);
  msg << " cannot infer output shapes.";
  throw std::invalid_argument(msg.str());
}

std::vector<array> Abs::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Abs::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], sign(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Abs::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{abs(inputs[0], stream())}, axes};
}

std::vector<array> Add::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {
      tangents.size() > 1 ? add(tangents[0], tangents[1], stream())
                          : tangents[0]};
}

std::vector<array> Add::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  if (argnums.size() == 1) {
    return cotangents;
  } else {
    return {cotangents[0], cotangents[0]};
  }
}

std::pair<std::vector<array>, std::vector<int>> Add::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{add(a, b, stream())}, {to_ax}};
}

std::vector<array> AddMM::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  auto& cotan = cotangents[0];
  std::vector<int> reorder(cotan.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  std::iter_swap(reorder.end() - 1, reorder.end() - 2);
  for (auto arg : argnums) {
    if (arg == 0) {
      // M X N * (K X N).T -> M X K
      auto cotan_scaled = cotan;
      if (alpha_ != 1.) {
        auto alpha_arr = array(alpha_, cotan.dtype());
        cotan_scaled = (multiply(alpha_arr, cotan_scaled, stream()));
      }
      vjps.push_back(matmul(
          cotan_scaled, transpose(primals[1], reorder, stream()), stream()));
    } else if (arg == 1) {
      // (M X K).T * M X N -> K X N
      auto cotan_scaled = cotan;
      if (alpha_ != 1.) {
        auto alpha_arr = array(alpha_, cotan.dtype());
        cotan_scaled = (multiply(alpha_arr, cotan_scaled, stream()));
      }
      vjps.push_back(matmul(
          transpose(primals[0], reorder, stream()), cotan_scaled, stream()));
    } else {
      auto cotan_scaled = cotan;
      if (beta_ != 1.) {
        auto beta_arr = array(beta_, cotan.dtype());
        cotan_scaled = (multiply(beta_arr, cotan_scaled, stream()));
      }
      vjps.push_back(cotan_scaled);
    }
  }
  return vjps;
}

bool AddMM::is_equivalent(const Primitive& other) const {
  const AddMM& a_other = static_cast<const AddMM&>(other);
  return (alpha_ == a_other.alpha_ && beta_ == a_other.beta_);
}

std::pair<std::vector<array>, std::vector<int>> AddMM::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto maybe_move_ax = [this](auto& arr, auto ax) {
    return ax > 0 ? moveaxis(arr, ax, 0, stream()) : arr;
  };
  auto a = maybe_move_ax(inputs[0], axes[0]);
  auto b = maybe_move_ax(inputs[1], axes[1]);
  auto c = maybe_move_ax(inputs[2], axes[2]);
  return {{addmm(c, a, b, alpha_, beta_, stream())}, {0}};
}

bool Arange::is_equivalent(const Primitive& other) const {
  const Arange& a_other = static_cast<const Arange&>(other);
  return (
      start_ == a_other.start_ && stop_ == a_other.stop_ &&
      step_ == a_other.step_);
}

std::vector<Shape> Arange::output_shapes(const std::vector<array>&) {
  auto real_size = std::ceil((stop_ - start_) / step_);
  return {{std::max(static_cast<int>(real_size), 0)}};
}

std::vector<array> ArcCos::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcCos::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  array denom = negative(rsqrt(t, stream()), stream());
  return {multiply(tangents[0], denom, stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcCos::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arccos(inputs[0], stream())}, axes};
}

std::vector<array> ArcCosh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcCosh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(square(primals[0], stream()), one, stream());
  return {multiply(tangents[0], rsqrt(t, stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcCosh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arccosh(inputs[0], stream())}, axes};
}

std::vector<array> ArcSin::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcSin::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  return {multiply(tangents[0], rsqrt(t, stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcSin::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arcsin(inputs[0], stream())}, axes};
}

std::vector<array> ArcSinh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcSinh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = add(square(primals[0], stream()), one, stream());
  return {multiply(tangents[0], rsqrt(t, stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcSinh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arcsinh(inputs[0], stream())}, axes};
}

std::vector<array> ArcTan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcTan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = add(one, square(primals[0], stream()), stream());
  return {divide(tangents[0], t, stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcTan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arctan(inputs[0], stream())}, axes};
}

std::vector<array> ArcTan2::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcTan2::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(argnums.size() == 2);
  array t =
      add(square(primals[0], stream()), square(primals[1], stream()), stream());
  return {
      divide(tangents[0], t, stream()),
      divide(negative(tangents[1], stream()), t, stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcTan2::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 2);
  assert(axes.size() == 2);
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{arctan2(a, b, stream())}, {to_ax}};
}

std::vector<array> ArcTanh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> ArcTanh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array one = array(1., primals[0].dtype());
  array t = subtract(one, square(primals[0], stream()), stream());
  return {divide(tangents[0], t, stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArcTanh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{arctanh(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> ArgPartition::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {{argpartition(inputs[0], axis_ + axis_left, stream())}, axes};
}

std::vector<array> ArgPartition::vjp(
    const std::vector<array>& primals,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {zeros_like(primals[0], stream())};
}

std::vector<array> ArgPartition::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {zeros_like(tangents[0], stream())};
}

bool ArgPartition::is_equivalent(const Primitive& other) const {
  const ArgPartition& r_other = static_cast<const ArgPartition&>(other);
  return axis_ == r_other.axis_ && kth_ == r_other.kth_;
}

bool ArgReduce::is_equivalent(const Primitive& other) const {
  const ArgReduce& r_other = static_cast<const ArgReduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axis_ == r_other.axis_;
}

std::pair<std::vector<array>, std::vector<int>> ArgReduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  int reduce_ax = axis_ + (axes[0] >= 0 && axis_ >= axes[0]);
  auto& in = inputs[0];
  std::vector<array> out;
  if (reduce_type_ == ArgReduce::ArgMin) {
    out.push_back(argmin(in, reduce_ax, true, stream()));
  } else {
    out.push_back(argmax(in, reduce_ax, true, stream()));
  }
  return {out, axes};
}

std::vector<array> ArgReduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {zeros_like(primals[0], stream())};
}

std::vector<array> ArgReduce::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {zeros_like(tangents[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> ArgSort::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {{argsort(inputs[0], axis_ + axis_left, stream())}, axes};
}

std::vector<Shape> ArgReduce::output_shapes(const std::vector<array>& inputs) {
  auto out_shape = inputs[0].shape();
  out_shape[axis_] = 1;
  return {std::move(out_shape)};
}

bool ArgSort::is_equivalent(const Primitive& other) const {
  const ArgSort& r_other = static_cast<const ArgSort&>(other);
  return axis_ == r_other.axis_;
}

std::vector<array> AsType::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  if (cotangents[0].dtype() != dtype_) {
    throw std::invalid_argument(
        "[astype] Type of cotangents does not match primal output type.");
  }
  return {astype(cotangents[0], primals[0].dtype(), stream())};
}

std::vector<array> AsType::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {astype(tangents[0], dtype_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> AsType::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{astype(inputs[0], dtype_, stream())}, axes};
}

bool AsType::is_equivalent(const Primitive& other) const {
  const AsType& a_other = static_cast<const AsType&>(other);
  return dtype_ == a_other.dtype_;
}

std::vector<array> AsStrided::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(argnums.size() == 1);

  // Extract the sizes and cast them to ints
  int grad_size = primals[0].size();
  int cotangents_size = cotangents[0].size();

  // Make a flat container to hold the gradients
  auto grad = zeros_like(primals[0], stream());
  grad = reshape(grad, {grad_size}, stream());

  // Create the indices that map output to input
  auto idx = arange(grad_size, stream());
  idx = as_strided(idx, shape_, strides_, offset_, stream());
  idx = reshape(idx, {cotangents_size}, stream());

  // Reshape the cotangentsgent for use with scatter
  auto flat_cotangents = reshape(cotangents[0], {cotangents_size, 1}, stream());

  // Finally accumulate the gradients and reshape them to look like the input
  grad = scatter_add(grad, idx, flat_cotangents, 0, stream());
  grad = reshape(grad, primals[0].shape(), stream());

  return {grad};
}

std::vector<array> AsStrided::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);

  return {as_strided(tangents[0], shape_, strides_, offset_, stream())};
}

bool AsStrided::is_equivalent(const Primitive& other) const {
  const AsStrided& a_other = static_cast<const AsStrided&>(other);
  return shape_ == a_other.shape_ && strides_ == a_other.strides_ &&
      offset_ == a_other.offset_;
}

bool BitwiseBinary::is_equivalent(const Primitive& other) const {
  const BitwiseBinary& a_other = static_cast<const BitwiseBinary&>(other);
  return op_ == a_other.op_;
}

void BitwiseBinary::print(std::ostream& os) {
  switch (op_) {
    case BitwiseBinary::And:
      os << "BitwiseAnd";
      break;
    case BitwiseBinary::Or:
      os << "BitwiseOr";
      break;
    case BitwiseBinary::Xor:
      os << "BitwiseXor";
      break;
    case BitwiseBinary::LeftShift:
      os << "LeftShift";
      break;
    case BitwiseBinary::RightShift:
      os << "RightShift";
      break;
  }
}

std::pair<std::vector<array>, std::vector<int>> BitwiseBinary::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {
      {array(
          a.shape(),
          a.dtype(),
          std::make_shared<BitwiseBinary>(stream(), op_),
          {a, b})},
      {to_ax}};
}

std::vector<array> BitwiseBinary::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(tangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> BitwiseBinary::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Broadcast::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  // Reduce cotangents to the shape of the primal
  auto& shape = primals[0].shape();
  auto& cotan = cotangents[0];
  int diff = cotan.ndim() - shape.size();
  std::vector<int> squeeze_axes(diff);
  std::iota(squeeze_axes.begin(), squeeze_axes.end(), 0);
  auto reduce_axes = squeeze_axes;
  for (int i = diff; i < cotan.ndim(); ++i) {
    if (shape[i - diff] != cotan.shape(i)) {
      reduce_axes.push_back(i);
    }
  }
  auto out =
      squeeze(sum(cotan, reduce_axes, true, stream()), squeeze_axes, stream());
  return {out};
}

std::vector<array> Broadcast::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {broadcast_to(tangents[0], shape_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Broadcast::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto in = inputs[0];
  if (ax >= 0) {
    int diff = shape_.size() - in.ndim() + 1;
    assert(diff >= 0);
    shape_.insert(shape_.begin() + ax + diff, in.shape(ax));
    ax += diff;
  }
  return {{broadcast_to(in, shape_, stream())}, {ax}};
}

bool Broadcast::is_equivalent(const Primitive& other) const {
  const Broadcast& b_other = static_cast<const Broadcast&>(other);
  return shape_ == b_other.shape_;
}

Shape Broadcast::output_shape(const std::vector<array>& inputs) {
  auto shape = inputs[0].shape();
  for (int i = 1; i < inputs.size(); ++i) {
    shape = broadcast_shapes(shape, inputs[i].shape());
  }
  return shape;
}

std::vector<Shape> Broadcast::output_shapes(const std::vector<array>& inputs) {
  if (inputs.size() < 2) {
    if (broadcast_shapes(inputs[0].shape(), shape_) != shape_) {
      throw std::invalid_argument(
          "[Broadcast] Unable to infer broadcast shape");
    }
    return {shape_};
  }
  return {output_shape(inputs)};
};

std::pair<std::vector<array>, std::vector<int>> BroadcastAxes::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::invalid_argument("[BroadcastAxes] VMAP NYI");
}

bool BroadcastAxes::is_equivalent(const Primitive& other) const {
  const auto& b_other = static_cast<const BroadcastAxes&>(other);
  return ignore_axes_ == b_other.ignore_axes_;
}

Shape BroadcastAxes::output_shape(
    const std::vector<array>& inputs,
    const std::vector<int>& ignore_axes) {
  auto shape = Shape{};
  for (auto& in : inputs) {
    auto in_shape = in.shape();
    for (auto it = ignore_axes.rbegin(); it != ignore_axes.rend(); ++it) {
      in_shape.erase(in_shape.begin() + in.ndim() + *it);
    }
    shape = broadcast_shapes(shape, in_shape);
  }
  int dims = ignore_axes.size() + shape.size();
  for (auto ax : ignore_axes) {
    shape.insert(shape.begin() + dims + ax, inputs[0].shape(ax));
  }
  return shape;
}

std::vector<Shape> BroadcastAxes::output_shapes(
    const std::vector<array>& inputs) {
  return {output_shape(inputs, ignore_axes_)};
}

std::vector<array> Ceil::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Ceil::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Ceil::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{ceil(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Cholesky::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::cholesky(a, upper_, stream())}, {ax}};
}

std::pair<std::vector<array>, std::vector<int>> Eigh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  bool needs_move = axes[0] >= (inputs[0].ndim() - 2);
  auto a = needs_move ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  auto ax = needs_move ? 0 : axes[0];

  std::vector<array> outputs;
  if (compute_eigenvectors_) {
    auto [values, vectors] = linalg::eigh(a, uplo_, stream());
    outputs = {values, vectors};
  } else {
    outputs = {linalg::eigvalsh(a, uplo_, stream())};
  }

  return {outputs, std::vector<int>(outputs.size(), ax)};
}

std::vector<Shape> Eigh::output_shapes(const std::vector<array>& inputs) {
  auto shape = inputs[0].shape();
  shape.pop_back(); // Remove last dimension for eigenvalues
  if (compute_eigenvectors_) {
    return {
        std::move(shape), inputs[0].shape()}; // Eigenvalues and eigenvectors
  } else {
    return {std::move(shape)}; // Only eigenvalues
  }
}

bool Eigh::is_equivalent(const Primitive& other) const {
  auto& e_other = static_cast<const Eigh&>(other);
  return uplo_ == e_other.uplo_ &&
      compute_eigenvectors_ == e_other.compute_eigenvectors_;
}

std::vector<array> Concatenate::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& cotan = cotangents[0];
  Shape start(cotan.ndim(), 0);
  Shape stop = cotan.shape();

  Shape sizes;
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

std::vector<array> Concatenate::jvp(
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
  return {concatenate(vals, axis_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Concatenate::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  int out_ax = -1;
  int first_vmap = -1;

  // Find the first vmapped input
  for (int i = 0; i < axes.size(); i++) {
    if (axes[i] >= 0) {
      out_ax = axes[i];
      first_vmap = i;
      break;
    }
  }

  // No vmap, should we even be in here?
  if (out_ax < 0) {
    return {{concatenate(inputs, axis_, stream())}, {out_ax}};
  }

  // Make sure vmapped arrays have all vmapped axes in the same location and
  // expand non-vmapped arrays to be compatible with the vmapped ones.
  std::vector<array> t_inputs;
  int N = inputs[first_vmap].shape(out_ax);
  int axis = axis_ + (axis_ >= out_ax);
  auto cat_shape = inputs[first_vmap].shape();
  for (int i = 0; i < axes.size(); i++) {
    if (axes[i] >= 0) {
      if (out_ax != axes[i]) {
        t_inputs.push_back(moveaxis(inputs[i], axes[i], out_ax, stream()));
      } else {
        t_inputs.push_back(inputs[i]);
      }
    } else {
      cat_shape[axis] = inputs[i].shape(axis_);
      t_inputs.push_back(broadcast_to(
          expand_dims(inputs[i], out_ax, stream()), cat_shape, stream()));
    }
  }

  return {{concatenate(t_inputs, axis, stream())}, {out_ax}};
}

bool Concatenate::is_equivalent(const Primitive& other) const {
  const Concatenate& c_other = static_cast<const Concatenate&>(other);
  return axis_ == c_other.axis_;
}

std::vector<Shape> Concatenate::output_shapes(
    const std::vector<array>& inputs) {
  auto shape = inputs[0].shape();
  for (int i = 1; i < inputs.size(); ++i) {
    shape[axis_] += inputs[i].shape(axis_);
  }
  return {std::move(shape)};
}

std::pair<std::vector<array>, std::vector<int>> Conjugate::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{conjugate(inputs[0], stream())}, axes};
}

std::vector<array> Contiguous::vjp(
    const std::vector<array>&,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {cotangents};
}

std::vector<array> Contiguous::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {tangents};
}

std::pair<std::vector<array>, std::vector<int>> Contiguous::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{contiguous(inputs[0], allow_col_major_, stream())}, axes};
}

bool Contiguous::is_equivalent(const Primitive& other) const {
  const Contiguous& c_other = static_cast<const Contiguous&>(other);
  return allow_col_major_ == c_other.allow_col_major_;
}

array conv_weight_backward_patches(
    const array& in,
    const array& wt,
    const array& cotan,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding,
    StreamOrDevice s) {
  // Resolve Padded input shapes and strides
  Shape padding_starts(in.ndim(), 0);
  auto padding_ends = in.shape();
  auto in_padded_shape = in.shape();

  // padded shape
  for (int i = 1; i < in.ndim() - 1; i++) {
    in_padded_shape[i] += 2 * padding[i - 1];
    padding_ends[i] += padding[i - 1];
    padding_starts[i] += padding[i - 1];
  }

  // padded strides (contiguous)
  Strides in_padded_strides(in.ndim(), 1);
  for (int i = in.ndim() - 2; i >= 0; --i) {
    in_padded_strides[i] = in_padded_strides[i + 1] * in_padded_shape[i + 1];
  }

  // Pad input
  std::vector<int> padded_axes(in.ndim() - 2, 0);
  std::iota(padded_axes.begin(), padded_axes.end(), 1);
  Shape padding_(padding.begin(), padding.end());
  auto in_padded = pad(
      in, padded_axes, padding_, padding_, array(0, in.dtype()), "constant", s);

  // Resolve strided patches

  // patches are shaped as
  // (batch_dim, out_spatial_dims, weight_spatial_dims, in_channels)
  Shape patches_shape{cotan.shape().begin(), cotan.shape().end() - 1};
  patches_shape.insert(
      patches_shape.end(), wt.shape().begin() + 1, wt.shape().end());

  // Resolve patch strides
  int n_spatial_dim = in.ndim() - 2;
  Strides patches_strides(patches_shape.size(), 1);
  patches_strides[0] = in_padded_strides[0];
  for (int i = 1; i < n_spatial_dim + 1; i++) {
    patches_strides[i] = in_padded_strides[i] * kernel_strides[i - 1];
  }
  for (int i = 1; i < in.ndim(); i++) {
    patches_strides[n_spatial_dim + i] = in_padded_strides[i];
  }

  // Make patches from in
  auto in_patches = as_strided(in_padded, patches_shape, patches_strides, 0, s);

  // Prepare for matmul
  int O = wt.shape(0);
  auto cotan_mat = reshape(cotan, {-1, O}, s);
  in_patches = reshape(in_patches, {cotan_mat.shape(0), -1}, s);

  auto grad = matmul(transpose(cotan_mat, {1, 0}, s), in_patches, s);
  grad = reshape(grad, wt.shape(), s);
  return grad;
}

std::vector<array> Convolution::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  std::vector<array> grads;

  // Collect info
  auto& in = primals[0];
  auto& wt = primals[1];
  auto& cotan = cotangents[0];

  auto group_transpose =
      [this](const array& x, int group_dim, int ax_a, int ax_b) {
        if (groups_ > 1) {
          auto shape = x.shape();
          if (group_dim < 0) {
            group_dim += shape.size();
          }
          shape.insert(shape.begin() + group_dim, groups_);
          shape[group_dim + 1] = shape[group_dim + 1] / groups_;
          auto x_trans = swapaxes(
              reshape(x, std::move(shape), stream()), ax_a, ax_b, stream());
          return flatten(x_trans, group_dim, group_dim + 1, stream());
        } else {
          return swapaxes(x, 0, -1, stream());
        }
      };

  for (int a : argnums) {
    // Grads for input
    if (a == 0) {
      std::vector<int> padding_lo = padding_;
      std::vector<int> padding_hi = padding_;

      for (int i = 0; i < padding_lo.size(); ++i) {
        int wt_size = 1 + kernel_dilation_[i] * (wt.shape(1 + i) - 1);
        padding_lo[i] = wt_size - padding_[i] - 1;

        int in_size = 1 + input_dilation_[i] * (in.shape(1 + i) - 1);
        int out_size = 1 + kernel_strides_[i] * (cotan.shape(1 + i) - 1);
        padding_hi[i] = in_size - out_size + padding_[i];
      }

      // Check for negative padding
      bool has_neg_padding = false;
      for (auto& pd : padding_lo) {
        has_neg_padding |= (pd < 0);
      }
      for (auto& pd : padding_hi) {
        has_neg_padding |= (pd < 0);
      }

      auto padding_lo_ = std::vector<int>(padding_lo);
      auto padding_hi_ = std::vector<int>(padding_hi);

      // Use negative padding on the gradient output
      if (has_neg_padding) {
        for (auto& p : padding_lo_) {
          p = std::max(0, p);
        }
        for (auto& p : padding_hi_) {
          p = std::max(0, p);
        }
      }

      auto wt_trans = group_transpose(wt, 0, 1, -1);
      auto grad = conv_general(
          /* const array& input = */ cotan,
          /* const array& weight = */ wt_trans,
          /* std::vector<int> stride = */ input_dilation_,
          /* std::vector<int> padding_lo = */ padding_lo,
          /* std::vector<int> padding_hi = */ padding_hi,
          /* std::vector<int> kernel_dilation = */ kernel_dilation_,
          /* std::vector<int> input_dilation = */ kernel_strides_,
          /* int groups = */ groups_,
          /* bool flip = */ !flip_,
          stream());

      // Handle negative padding
      if (has_neg_padding) {
        Shape starts(grad.ndim(), 0);
        auto stops = grad.shape();

        for (int i = 0; i < grad.ndim() - 2; i++) {
          if (padding_lo[i] < 0) {
            starts[i + 1] -= padding_lo[i];
            padding_lo[i] = 0;
          }

          if (padding_hi[i] < 0) {
            stops[i + 1] += padding_hi[i];
            padding_hi[i] = 0;
          }
        }

        grad = slice(grad, std::move(starts), std::move(stops), stream());
      }

      grads.push_back(grad);
    }
    // Grads for weight
    else if (a == 1) {
      bool no_dilation = true;

      for (int i = 0; i < input_dilation_.size(); i++) {
        no_dilation &= (input_dilation_[i] == 1) && (kernel_dilation_[i] == 1);
      }

      if (no_dilation && !flip_ && groups_ == 1) {
        auto grad = conv_weight_backward_patches(
            in, wt, cotan, kernel_strides_, padding_, stream());
        grads.push_back(grad);
      } else {
        if (flip_) {
          auto padding = padding_;
          for (int i = 0; i < padding.size(); i++) {
            int wt_size = 1 + kernel_dilation_[i] * (wt.shape(1 + i) - 1);
            padding[i] = wt_size - padding_[i] - 1;
          }

          auto cotan_trans = group_transpose(cotan, -1, 0, -1);
          auto in_trans = swapaxes(in, 0, -1, stream());

          auto grad_trans = conv_general(
              /* const array& input = */ cotan_trans,
              /* const array& weight = */ in_trans,
              /* std::vector<int> stride = */ kernel_dilation_,
              /* std::vector<int> padding_lo = */ padding,
              /* std::vector<int> padding_hi = */ padding,
              /* std::vector<int> kernel_dilation = */ input_dilation_,
              /* std::vector<int> input_dilation = */ kernel_strides_,
              /* int groups = */ groups_,
              /* bool flip = */ false,
              stream());
          if (groups_ > 1) {
            grads.push_back(group_transpose(grad_trans, -1, 0, -2));
          } else {
            grads.push_back(grad_trans);
          }
        } else {
          std::vector<int> padding_lo = padding_;
          std::vector<int> padding_hi = padding_;

          for (int i = 0; i < padding_hi.size(); ++i) {
            int in_size = 1 + input_dilation_[i] * (in.shape(1 + i) - 1);
            int out_size = 1 + kernel_strides_[i] * (cotan.shape(1 + i) - 1);
            int wt_size = 1 + kernel_dilation_[i] * (wt.shape(1 + i) - 1);
            padding_hi[i] = out_size - in_size + wt_size - padding_[i] - 1;
          }
          auto cotan_trans = swapaxes(cotan, 0, -1, stream());
          auto in_trans = group_transpose(in, -1, 0, -1);

          auto grad_trans = conv_general(
              /* const array& input = */ in_trans,
              /* const array& weight = */ cotan_trans,
              /* std::vector<int> stride = */ kernel_dilation_,
              /* std::vector<int> padding_lo = */ padding_lo,
              /* std::vector<int> padding_hi = */ padding_hi,
              /* std::vector<int> kernel_dilation = */ kernel_strides_,
              /* std::vector<int> input_dilation = */ input_dilation_,
              /* int groups = */ groups_,
              /* bool flip = */ false,
              stream());
          grads.push_back(swapaxes(grad_trans, 0, -1, stream()));
        }
      }
    }
  }

  return grads;
}

bool Convolution::is_equivalent(const Primitive& other) const {
  const Convolution& c_other = static_cast<const Convolution&>(other);
  return padding_ == c_other.padding_ &&
      kernel_strides_ == c_other.kernel_strides_ &&
      kernel_dilation_ == c_other.kernel_dilation_ &&
      input_dilation_ == c_other.input_dilation_ &&
      groups_ == c_other.groups_ && flip_ == c_other.flip_;
}

std::vector<array> Copy::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return cotangents;
}

std::vector<array> Copy::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents;
}

std::pair<std::vector<array>, std::vector<int>> Copy::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{copy(inputs[0], stream())}, axes};
}

std::vector<array> Cos::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return {jvp(primals, cotangents, argnums)};
}

std::vector<array> Cos::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(
      tangents[0], negative(sin(primals[0], stream()), stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Cos::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{cos(inputs[0], stream())}, axes};
}

std::vector<array> Cosh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Cosh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], sinh(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Cosh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{cosh(inputs[0], stream())}, axes};
}

std::vector<array> CustomTransforms::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  // Extract the inputs to the VJP function
  std::vector<array> inputs(primals.begin(), primals.end() - num_outputs_);

  // Compute all the vjps
  auto all_vjps = vjp_fun_(inputs, cotangents, outputs);
  for (const auto& cot : cotangents) {
    all_vjps.emplace_back(cot);
  }

  // Select the vjps requested
  std::vector<array> vjps;
  vjps.reserve(argnums.size());
  for (auto arg : argnums) {
    vjps.push_back(all_vjps[arg]);
  }

  return vjps;
}

std::vector<array> CustomTransforms::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // Extract the inputs to the JVP function
  std::vector<array> inputs(primals.begin(), primals.end() - num_outputs_);

  // Compute the jvps
  return jvp_fun_(inputs, tangents, argnums);
}

std::pair<std::vector<array>, std::vector<int>> CustomTransforms::vmap(
    const std::vector<array>& inputs_,
    const std::vector<int>& axes_) {
  // Extract the inputs to the vmap function
  std::vector<array> inputs(inputs_.begin(), inputs_.end() - num_outputs_);
  std::vector<int> axes(axes_.begin(), axes_.end() - num_outputs_);
  return vmap_fun_(inputs, axes);
}

std::vector<array> Depends::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  std::vector<array> vjps;

  for (auto arg : argnums) {
    if (arg < cotangents.size()) {
      vjps.push_back(cotangents[arg]);
    } else {
      vjps.push_back(zeros_like(primals[arg]));
    }
  }
  return vjps;
}

std::vector<array> Divide::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(divide(cotangents[0], primals[1], stream()));
    } else {
      vjps.push_back(negative(
          divide(
              multiply(cotangents[0], primals[0], stream()),
              square(primals[1], stream()),
              stream()),
          stream()));
    }
  }
  return vjps;
}

std::vector<array> DivMod::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> DivMod::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> DivMod::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {divmod(a, b, stream()), {to_ax}};
}

std::vector<array> Divide::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Divide::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{divide(a, b, stream())}, {to_ax}};
}

std::vector<array> Remainder::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(cotangents[0]);
    } else {
      auto x_over_y = divide(primals[0], primals[1], stream());
      x_over_y = floor(x_over_y, stream());
      vjps.push_back(
          negative(multiply(x_over_y, cotangents[0], stream()), stream()));
    }
  }
  return vjps;
}

std::vector<array> Remainder::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Remainder::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{remainder(a, b, stream())}, {to_ax}};
}

std::pair<std::vector<array>, std::vector<int>> Equal::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{equal(a, b, stream())}, {to_ax}};
}

std::vector<array> Equal::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Equal::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Erf::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Erf::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(M_2_SQRTPI, dtype), tangents[0], stream());
  return {multiply(
      scale,
      exp(negative(square(primals[0], stream()), stream()), stream()),
      stream())};
}

std::pair<std::vector<array>, std::vector<int>> Erf::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{erf(inputs[0], stream())}, axes};
}

std::vector<array> ErfInv::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto dtype = primals[0].dtype();
  auto scale =
      multiply(array(1.0 / M_2_SQRTPI, dtype), cotangents[0], stream());
  return {
      multiply(scale, exp(square(outputs[0], stream()), stream()), stream())};
}

std::vector<array> ErfInv::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  auto scale = multiply(array(1.0 / M_2_SQRTPI, dtype), tangents[0], stream());
  return {multiply(
      scale,
      exp(square(erfinv(primals[0], stream()), stream()), stream()),
      stream())};
}

std::pair<std::vector<array>, std::vector<int>> ErfInv::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{erfinv(inputs[0], stream())}, axes};
}

std::vector<array> Exp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return {multiply(cotangents[0], outputs[0], stream())};
}

std::vector<array> Exp::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], exp(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Exp::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{exp(inputs[0], stream())}, axes};
}

std::vector<array> Expm1::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  return {multiply(
      cotangents[0],
      add(outputs[0], array(1.0f, outputs[0].dtype()), stream()),
      stream())};
}

std::vector<array> Expm1::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], exp(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Expm1::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{expm1(inputs[0], stream())}, axes};
}

std::vector<array> ExpandDims::vjp(
    const std::vector<array>&,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {squeeze(cotangents[0], axes_, stream())};
}

std::vector<array> ExpandDims::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {expand_dims(tangents[0], axes_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> ExpandDims::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto expand_axes = axes_;
  for (auto& s : expand_axes) {
    if (s >= axes[0]) {
      s++;
    } else {
      ax++;
    }
  }
  return {{expand_dims(inputs[0], std::move(expand_axes), stream())}, {ax}};
}

bool ExpandDims::is_equivalent(const Primitive& other) const {
  const ExpandDims& a_other = static_cast<const ExpandDims&>(other);
  return (axes_ == a_other.axes_);
}

Shape ExpandDims::output_shape(
    const array& input,
    const std::vector<int>& axes) {
  auto shape = input.shape();
  for (auto ax : axes) {
    shape.insert(shape.begin() + ax, 1);
  }
  return shape;
}

std::vector<Shape> ExpandDims::output_shapes(const std::vector<array>& inputs) {
  return {ExpandDims::output_shape(inputs[0], axes_)};
}

std::vector<array> Flatten::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  auto& in = primals[0];
  Shape unflatten_shape(
      in.shape().begin() + start_axis_, in.shape().begin() + end_axis_ + 1);
  return {unflatten(
      cotangents[0], start_axis_, std::move(unflatten_shape), stream())};
}

std::vector<array> Flatten::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {flatten(tangents[0], start_axis_, end_axis_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Flatten::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto start_axis = start_axis_;
  auto end_axis = end_axis_;
  if (ax < start_axis) {
    start_axis++;
    end_axis++;
  } else {
    ax -= (end_axis - start_axis);
  }
  return {{flatten(inputs[0], start_axis, end_axis, stream())}, {ax}};
}

bool Flatten::is_equivalent(const Primitive& other) const {
  const Flatten& a_other = static_cast<const Flatten&>(other);
  return start_axis_ == a_other.start_axis_ && end_axis_ == a_other.end_axis_;
}

Shape Flatten::output_shape(const array& input, int start_axis, int end_axis) {
  Shape shape = input.shape();
  auto flat_size = input.shape(start_axis);
  for (int ax = start_axis + 1; ax <= end_axis; ++ax) {
    flat_size *= input.shape(ax);
  }
  shape.erase(shape.begin() + start_axis + 1, shape.begin() + end_axis + 1);
  shape[start_axis] = flat_size;
  return shape;
}

std::vector<Shape> Flatten::output_shapes(const std::vector<array>& inputs) {
  return {Flatten::output_shape(inputs[0], start_axis_, end_axis_)};
}

bool FFT::is_equivalent(const Primitive& other) const {
  const FFT& r_other = static_cast<const FFT&>(other);
  return axes_ == r_other.axes_ && inverse_ == r_other.inverse_ &&
      real_ == r_other.real_;
}

std::vector<array> Unflatten::vjp(
    const std::vector<array>&,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {flatten(cotangents[0], axis_, axis_ + shape_.size() - 1, stream())};
}

std::vector<array> Unflatten::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {unflatten(tangents[0], axis_, shape_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Unflatten::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto axis = axis_;
  if (ax <= axis_) {
    axis++;
  } else {
    ax += (shape_.size() - 1);
  }
  return {{unflatten(inputs[0], axis, shape_, stream())}, {ax}};
}

bool Unflatten::is_equivalent(const Primitive& other) const {
  const auto& a_other = static_cast<const Unflatten&>(other);
  return axis_ == a_other.axis_ && shape_ == a_other.shape_;
}

Shape Unflatten::output_shape(
    const array& input,
    int axis,
    const Shape& shape) {
  Shape out_shape = input.shape();
  out_shape[axis] = shape[0];
  out_shape.insert(
      out_shape.begin() + axis + 1, shape.begin() + 1, shape.end());
  return out_shape;
}

std::vector<Shape> Unflatten::output_shapes(const std::vector<array>& inputs) {
  return {Unflatten::output_shape(inputs[0], axis_, shape_)};
}

std::pair<std::vector<array>, std::vector<int>> FFT::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& in = inputs[0];
  int ax = axes[0];
  auto fft_axes = axes_;
  auto out_shape = in.shape();
  if (ax >= 0) {
    for (auto& fft_ax : fft_axes) {
      if (fft_ax >= ax) {
        fft_ax++;
      }
      if (real_) {
        auto n = out_shape[fft_ax];
        out_shape[fft_ax] = inverse_ ? 2 * (n - 1) : n / 2 + 1;
      }
    }
  }
  return {
      {array(
          out_shape,
          real_ && inverse_ ? float32 : complex64,
          std::make_shared<FFT>(stream(), fft_axes, inverse_, real_),
          {in})},
      {ax}};
}

std::vector<array> FFT::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto& in = primals[0];
  std::vector<int> axes(axes_.begin(), axes_.end());
  if (real_ && inverse_) {
    auto out = fft::fftn(cotangents[0], axes, stream());
    auto start = Shape(out.ndim(), 0);
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
    Shape n;
    for (auto ax : axes_) {
      n.push_back(in.shape()[ax]);
    }
    return {astype(
        fft::fftn(cotangents[0], n, axes, stream()), in.dtype(), stream())};
  } else if (inverse_) {
    return {fft::ifftn(cotangents[0], axes, stream())};
  } else {
    return {fft::fftn(cotangents[0], axes, stream())};
  }
}

std::vector<array> FFT::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto& tan = tangents[0];
  if (real_ & inverse_) {
    return {fft::irfftn(tan, stream())};
  } else if (real_) {
    return {fft::rfftn(tan, stream())};
  } else if (inverse_) {
    return {fft::ifftn(tan, stream())};
  } else {
    return {fft::fftn(tan, stream())};
  }
}

std::vector<array> Floor::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Floor::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Floor::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{floor(inputs[0], stream())}, axes};
}

std::vector<array> Full::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(cotangents[0], primals[0], stream())};
}

std::vector<array> Full::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return tangents;
}

std::pair<std::vector<array>, std::vector<int>> Full::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto& in = inputs[0];
  auto out =
      array(in.shape(), in.dtype(), std::make_shared<Full>(stream()), {in});
  return {{out}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Gather::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& src = inputs[0];
  std::vector<array> indices(inputs.begin() + 1, inputs.end());
  auto gather_axes = axes_;
  auto slice_sizes = slice_sizes_;
  auto src_vmapped = axes[0] >= 0;
  auto ind_vmap_ax_ptr =
      std::find_if(axes.begin() + 1, axes.end(), [](int a) { return a >= 0; });
  int out_ax = -1;
  bool indices_vmapped = (ind_vmap_ax_ptr != axes.end());
  if (indices_vmapped) {
    out_ax = *ind_vmap_ax_ptr;
  } else if (src_vmapped) {
    out_ax = axes[0];
  }

  // Reorder all the index arrays so the vmap axis is in the same spot.
  if (indices_vmapped) {
    for (int i = 1; i < axes.size(); ++i) {
      if (out_ax != axes[i] && axes[i] >= 0) {
        indices[i - 1] = moveaxis(indices[i - 1], axes[i], out_ax, stream());
      } else if (axes[i] < 0) {
        indices[i - 1] = expand_dims(indices[i - 1], out_ax, stream());
      }
    }
  }

  int idx_dims = indices.empty() ? 0 : indices[0].ndim();

  if (src_vmapped) {
    for (auto& ax : gather_axes) {
      if (ax >= axes[0]) {
        ax++;
      }
    }
    if (indices_vmapped) {
      // Make a new index array for the vmapped dimension
      auto vmap_inds =
          arange(static_cast<ShapeElem>(0), src.shape(axes[0]), stream());
      // Reshape it so it broadcasts with other index arrays
      {
        auto shape = Shape(idx_dims, 1);
        shape[out_ax] = vmap_inds.size();
        vmap_inds = reshape(vmap_inds, std::move(shape), stream());
      }
      // Update gather axes and slice sizes accordingly
      slice_sizes.insert(slice_sizes.begin() + axes[0], 1);
      gather_axes.push_back(axes[0]);
      indices.push_back(vmap_inds);
    } else {
      slice_sizes.insert(slice_sizes.begin() + out_ax, src.shape(out_ax));
      out_ax += idx_dims;
    }
  }
  auto out = gather(src, indices, gather_axes, slice_sizes, stream());
  if (src_vmapped && indices_vmapped) {
    out = squeeze(out, idx_dims + axes[0], stream());
  }
  return {{out}, {out_ax}};
}

std::vector<array> Gather::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (int argnum : argnums) {
    if (argnum > 0) {
      // Grads w.r.t. indices are zero
      vjps.push_back(
          zeros(primals[argnum].shape(), primals[argnum].dtype(), stream()));
    } else {
      auto src = zeros_like(primals[0], stream());
      std::vector<array> inds(primals.begin() + 1, primals.end());
      vjps.push_back(scatter_add(src, inds, cotangents[0], axes_, stream()));
    }
  }
  return vjps;
}

std::vector<array> Gather::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[gather] Cannot calculate JVP with respect to indices.");
  }
  std::vector<array> inds(primals.begin() + 1, primals.end());
  return {gather(tangents[0], inds, axes_, slice_sizes_, stream())};
}

bool Gather::is_equivalent(const Primitive& other) const {
  const Gather& g_other = static_cast<const Gather&>(other);
  return axes_ == g_other.axes_ && slice_sizes_ == g_other.slice_sizes_;
}

std::vector<Shape> Gather::output_shapes(const std::vector<array>& inputs) {
  Shape out_shape;
  if (inputs.size() > 1) {
    out_shape = inputs[1].shape();
  }
  out_shape.insert(out_shape.end(), slice_sizes_.begin(), slice_sizes_.end());
  return {std::move(out_shape)};
}

std::pair<std::vector<array>, std::vector<int>> Greater::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{greater(a, b, stream())}, {to_ax}};
}

std::vector<array> Greater::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Greater::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> GreaterEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{greater_equal(a, b, stream())}, {to_ax}};
}

std::vector<array> GreaterEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> GreaterEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Imag::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(
      array(complex64_t{0.0f, -1.0f}, primals[0].dtype()),
      cotangents[0],
      stream())};
}

std::vector<array> Imag::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {imag(tangents[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Imag::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{imag(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Less::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{less(a, b, stream())}, {to_ax}};
}

std::vector<array> Less::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> Less::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> LessEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{less_equal(a, b, stream())}, {to_ax}};
}

std::vector<array> LessEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> LessEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Log::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Log::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Log::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto& in = inputs[0];
  return {
      {array(
          in.shape(),
          in.dtype(),
          std::make_shared<Log>(stream(), base_),
          {in})},
      axes};
}

std::vector<array> Log1p::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Log1p::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto dtype = primals[0].dtype();
  return {divide(
      tangents[0], add(array(1.0f, dtype), primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Log1p::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{log1p(inputs[0], stream())}, axes};
}

std::vector<array> LogicalNot::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> LogicalNot::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(tangents[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> LogicalNot::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{logical_not(inputs[0], stream())}, axes};
}

std::vector<array> LogicalAnd::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(cotangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> LogicalAnd::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(argnums.size() <= 2);
  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> LogicalAnd::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 2);
  assert(axes.size() == 2);

  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{logical_and(a, b, stream())}, {to_ax}};
}

std::vector<array> LogicalOr::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 2);
  std::vector<array> vjps = {zeros_like(cotangents[0], stream())};
  if (argnums.size() > 1) {
    vjps.push_back(vjps.back());
  }
  return vjps;
}

std::vector<array> LogicalOr::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 2);
  assert(argnums.size() <= 2);

  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> LogicalOr::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 2);
  assert(axes.size() == 2);

  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{logical_or(a, b, stream())}, {to_ax}};
}

std::vector<array> LogAddExp::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto a = primals[0];
  auto b = primals[1];
  auto s = sigmoid(subtract(a, b, stream()), stream());
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(
        cotangents[0],
        arg == 0 ? s : subtract(array(1.0f, s.dtype()), s, stream()),
        stream()));
  }
  return vjps;
}

std::vector<array> LogAddExp::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> LogAddExp::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{logaddexp(a, b, stream())}, {to_ax}};
}

std::vector<array> Matmul::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  auto& cotan = cotangents[0];
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

std::pair<std::vector<array>, std::vector<int>> Matmul::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto maybe_move_ax = [this](auto& arr, auto ax) {
    return ax > 0 ? moveaxis(arr, ax, 0, stream()) : arr;
  };
  auto a = maybe_move_ax(inputs[0], axes[0]);
  auto b = maybe_move_ax(inputs[1], axes[1]);
  return {{matmul(a, b, stream())}, {0}};
}

std::vector<Shape> Matmul::output_shapes(const std::vector<array>& inputs) {
  auto out_shape = inputs[0].shape();
  out_shape.back() = inputs[1].shape(-1);
  return {std::move(out_shape)};
}

std::vector<array> Maximum::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask =
        (arg == 0) ? greater(a, b, stream()) : less_equal(a, b, stream());
    vjps.push_back(multiply(cotangents[0], mask, stream()));
  }
  return {vjps};
}

std::vector<array> Maximum::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Maximum::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{maximum(a, b, stream())}, {to_ax}};
}

std::vector<array> Minimum::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto& a = primals[0];
  auto& b = primals[1];
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto mask =
        (arg == 0) ? less(a, b, stream()) : greater_equal(a, b, stream());
    vjps.push_back(multiply(cotangents[0], mask, stream()));
  }
  return vjps;
}

std::vector<array> Minimum::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Minimum::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{minimum(a, b, stream())}, {to_ax}};
}

std::vector<array> Multiply::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto arg = argnums[0];
  auto jvp = multiply(tangents[0], primals[1 - arg], stream());
  if (argnums.size() > 1) {
    arg = argnums[1];
    jvp = add(jvp, multiply(tangents[1], primals[1 - arg], stream()), stream());
  }
  return {jvp};
}

std::vector<array> Multiply::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(multiply(primals[1 - arg], cotangents[0], stream()));
  }
  return vjps;
}

std::pair<std::vector<array>, std::vector<int>> Multiply::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{multiply(a, b, stream())}, {to_ax}};
}

std::vector<array> Select::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 3);
  assert(tangents.size() == 3);

  auto jvp_fun = [&](int i) {
    int arg = argnums[i];

    if (arg == 0) {
      return zeros_like(primals[0], stream());
    } else if (arg == 1) {
      return multiply(
          astype(primals[0], tangents[1].dtype(), stream()),
          tangents[1],
          stream());
    } else {
      return multiply(
          astype(
              logical_not(primals[0], stream()), tangents[2].dtype(), stream()),
          tangents[2],
          stream());
    }
  };

  array jvp = jvp_fun(argnums[0]);
  for (int i = 1; i < argnums.size(); i++) {
    jvp = add(jvp, jvp_fun(argnums[i]));
  }
  return {jvp};
}

std::vector<array> Select::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 3);
  assert(cotangents.size() == 1);

  std::vector<array> vjps;
  for (auto arg : argnums) {
    if (arg == 0) {
      vjps.push_back(zeros_like(primals[0], stream()));
    } else if (arg == 1) {
      vjps.push_back(multiply(
          astype(primals[0], cotangents[0].dtype(), stream()),
          cotangents[0],
          stream()));
    } else if (arg == 2) {
      vjps.push_back(multiply(
          astype(
              logical_not(primals[0], stream()),
              cotangents[0].dtype(),
              stream()),
          cotangents[0],
          stream()));
    }
  }
  return vjps;
}

std::pair<std::vector<array>, std::vector<int>> Select::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, c, to_ax] = vmap_ternary_op(inputs, axes, stream());
  return {{where(a, b, c, stream())}, {to_ax}};
}

std::vector<array> Negative::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Negative::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {negative(tangents[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Negative::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{negative(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> NotEqual::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{not_equal(a, b, stream())}, axes};
}

std::vector<array> NotEqual::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    vjps.push_back(zeros_like(primals[arg], stream()));
  }
  return vjps;
}

std::vector<array> NotEqual::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto shape = broadcast_shapes(primals[0].shape(), primals[1].shape());
  return {zeros(shape, bool_, stream())};
}

std::vector<array> Pad::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  auto& cotan = cotangents[0];
  Shape start(cotan.ndim(), 0);
  auto stop = cotan.shape();

  for (auto i : axes_) {
    start[i] = low_pad_size_[i];
    stop[i] -= high_pad_size_[i];
  }

  auto out = slice(cotan, start, stop, stream());

  return {out};
}

std::vector<array> Pad::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(argnums.size() == 1 && argnums[0] == 0);

  return {
      pad(tangents[0],
          axes_,
          low_pad_size_,
          high_pad_size_,
          array(0, tangents[0].dtype()),
          "constant",
          stream())};
}

std::pair<std::vector<array>, std::vector<int>> Pad::vmap(
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
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  auto sort_idx = argpartition(primals[0], kth_, axis_, stream());
  return {put_along_axis(
      zeros_like(primals[0], stream()),
      sort_idx,
      cotangents[0],
      axis_,
      stream())};
}

std::vector<array> Partition::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto sort_idx = argpartition(primals[0], kth_, axis_, stream());
  auto out = take_along_axis(tangents[0], sort_idx, axis_, stream());
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Partition::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {{partition(inputs[0], axis_ + axis_left, stream())}, axes};
}

bool Partition::is_equivalent(const Primitive& other) const {
  const Partition& r_other = static_cast<const Partition&>(other);
  return axis_ == r_other.axis_ && kth_ == r_other.kth_;
}

std::vector<array> Power::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
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
      auto& exp = outputs[0];
      auto exp_vjp = multiply(log(primals[0], stream()), outputs[0], stream());
      // 0 * log 0 -> 0
      vjps.push_back(where(exp, exp_vjp, array(0.0f, exp.dtype()), stream()));
    }
    vjps.back() = multiply(cotangents[0], vjps.back(), stream());
  }
  return vjps;
}

std::vector<array> Power::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  auto output = power(primals[0], primals[1], stream());
  auto grads = vjp(primals, tangents, argnums, {output});
  if (argnums.size() > 1) {
    return {add(grads[0], grads[1], stream())};
  } else {
    return grads;
  }
}

std::pair<std::vector<array>, std::vector<int>> Power::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{power(a, b, stream())}, {to_ax}};
}

std::pair<std::vector<array>, std::vector<int>> QuantizedMatmul::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("QuantizedMatmul::vmap NYI");
}

std::vector<array> QuantizedMatmul::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;

  // We rely on the fact that w is always 2D so transpose is simple
  for (auto arg : argnums) {
    // gradient wrt to x
    if (arg == 0) {
      vjps.push_back(quantized_matmul(
          cotangents[0],
          primals[1],
          primals[2],
          primals[3],
          !transpose_,
          group_size_,
          bits_,
          stream()));
    }

    // gradient wrt to w_q, scales or biases
    else {
      throw std::runtime_error(
          "QuantizedMatmul::vjp no gradient wrt the quantized matrix yet.");
    }
  }
  return vjps;
}

std::vector<array> QuantizedMatmul::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("QuantizedMatmul::jvp NYI");
}

bool QuantizedMatmul::is_equivalent(const Primitive& other) const {
  const QuantizedMatmul& qm_other = static_cast<const QuantizedMatmul&>(other);
  return group_size_ == qm_other.group_size_ && bits_ == qm_other.bits_ &&
      transpose_ == qm_other.transpose_;
}

std::vector<Shape> QuantizedMatmul::output_shapes(
    const std::vector<array>& inputs) {
  auto& w = inputs[1];
  int w_outer_dims = (transpose_) ? w.shape(-2) : w.shape(-1) * 32 / bits_;
  auto out_shape = inputs[0].shape();
  out_shape.back() = w_outer_dims;
  return {std::move(out_shape)};
}

std::pair<std::vector<array>, std::vector<int>> GatherQMM::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("GatherQMM::vmap NYI");
}

std::vector<array> GatherQMM::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;

  auto& cotan = cotangents[0];

  auto& x = primals[0];
  auto& w = primals[1];
  auto& scales = primals[2];
  auto& biases = primals[3];
  auto& lhs_indices = primals[4];
  auto& rhs_indices = primals[5];

  for (auto arg : argnums) {
    // gradient wrt to x
    if (arg == 0) {
      vjps.push_back(reshape(
          scatter_add(
              flatten(zeros_like(x, stream()), 0, -3, stream()),
              lhs_indices,
              expand_dims(
                  gather_qmm(
                      cotan,
                      w,
                      scales,
                      biases,
                      std::nullopt,
                      rhs_indices,
                      !transpose_,
                      group_size_,
                      bits_,
                      stream()),
                  -3,
                  stream()),
              0,
              stream()),
          x.shape(),
          stream()));
    }

    // gradient wrt to the indices is undefined
    else if (arg > 3) {
      throw std::runtime_error(
          "GatherQMM::vjp cannot compute the gradient wrt the indices.");
    }

    // gradient wrt to w_q, scales or biases
    else {
      throw std::runtime_error(
          "GatherQMM::vjp no gradient wrt the quantized matrix yet.");
    }
  }
  return vjps;
}

std::vector<array> GatherQMM::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("GatherQMM::jvp NYI");
}

bool GatherQMM::is_equivalent(const Primitive& other) const {
  const GatherQMM& qm_other = static_cast<const GatherQMM&>(other);
  return group_size_ == qm_other.group_size_ && bits_ == qm_other.bits_ &&
      transpose_ == qm_other.transpose_;
}

std::pair<std::vector<array>, std::vector<int>> RandomBits::vmap(
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
  if (kax >= 0) {
    shape.insert(shape.begin() + kax, key.shape()[kax]);
  }

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
      std::make_shared<RandomBits>(stream(), shape, width_),
      {key});
  return {{out}, {kax}};
}

bool RandomBits::is_equivalent(const Primitive& other) const {
  const RandomBits& r_other = static_cast<const RandomBits&>(other);
  return shape_ == r_other.shape_;
}

std::vector<array> Real::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {astype(cotangents[0], primals[0].dtype(), stream())};
}

std::vector<array> Real::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {real(tangents[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Real::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{real(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Reshape::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  // Transpose the input so that the vmap dim is first.
  auto& in = inputs[0];
  auto ax = axes[0];
  if (ax >= 0) {
    std::vector<int> reorder(in.ndim());
    std::iota(reorder.begin(), reorder.end(), 0);
    reorder.erase(reorder.begin() + ax);
    reorder.insert(reorder.begin(), ax);
    // Insert the vmap dim into the shape at the beginning.
    auto out = transpose(in, reorder, stream());
    shape_.insert(shape_.begin(), in.shape()[ax]);
    // Reshape the transposed input to the new shape.
    return {{reshape(out, shape_, stream())}, {0}};
  } else {
    return {{reshape(in, shape_, stream())}, {ax}};
  }
}

std::vector<array> Reshape::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return {reshape(cotangents[0], primals[0].shape(), stream())};
}

std::vector<array> Reshape::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  assert(argnums[0] == 0);
  return {reshape(tangents[0], shape_, stream())};
}

bool Reshape::is_equivalent(const Primitive& other) const {
  const Reshape& r_other = static_cast<const Reshape&>(other);
  return shape_ == r_other.shape_;
}

std::vector<array> Reduce::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto in = primals[0];

  auto shape = in.shape();
  for (auto ax : axes_) {
    shape[ax] = 1;
  }
  auto& cotan = cotangents[0];
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
      Shape shape_flat;
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
    auto out = outputs[0];
    if (out.ndim() != in.ndim()) {
      out = expand_dims(out, axes_, stream());
    }
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

std::pair<std::vector<array>, std::vector<int>> Reduce::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto reduce_axes = axes_;
  if (ax >= 0) {
    for (auto& rax : reduce_axes) {
      if (rax >= ax) {
        rax++;
      }
    }
  }
  auto& in = inputs[0];
  std::vector<array> out;
  switch (reduce_type_) {
    case Reduce::And:
      out.push_back(all(in, reduce_axes, true, stream()));
      break;
    case Reduce::Or:
      out.push_back(any(in, reduce_axes, true, stream()));
      break;
    case Reduce::Sum:
      out.push_back(sum(in, reduce_axes, true, stream()));
      break;
    case Reduce::Prod:
      out.push_back(prod(in, reduce_axes, true, stream()));
      break;
    case Reduce::Min:
      out.push_back(min(in, reduce_axes, true, stream()));
      break;
    case Reduce::Max:
      out.push_back(max(in, reduce_axes, true, stream()));
      break;
  }
  return {out, axes};
}

bool Reduce::is_equivalent(const Primitive& other) const {
  const Reduce& r_other = static_cast<const Reduce&>(other);
  return reduce_type_ == r_other.reduce_type_ && axes_ == r_other.axes_;
}

std::vector<Shape> Reduce::output_shapes(const std::vector<array>& inputs) {
  auto out_shape = inputs[0].shape();
  for (auto i : axes_) {
    out_shape[i] = 1;
  }
  return {std::move(out_shape)};
}

std::vector<array> Round::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Round::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros_like(primals[0], stream())};
}

std::pair<std::vector<array>, std::vector<int>> Round::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{round(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Scan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto& in = inputs[0];
  auto out_dtype =
      (in.dtype() == bool_ && reduce_type_ == Scan::Sum) ? int32 : in.dtype();
  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {
      {array(
          in.shape(),
          out_dtype,
          std::make_shared<Scan>(
              stream(), reduce_type_, axis_ + axis_left, reverse_, inclusive_),
          {in})},
      axes};
}

std::vector<array> Scan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return {cumsum(cotangents[0], axis_, !reverse_, inclusive_, stream())};
  } else if (reduce_type_ == Scan::Prod) {
    auto in = primals[0];
    // Find the location of the first 0 and set it to 1:
    // - A: Exclusive cumprod
    // - B: Inclusive cumprod
    // - Find the location that is 0 in A and not zero B
    // Compute the gradient by:
    // - Compute the regular gradient for everything before the first zero
    // - Set the first zero to 1 and redo the computation, use this for the
    //   gradient of the first zero
    // - Everything after the first zero has a gradient of 0

    // Get inclusive and exclusive cum prods
    auto cprod_exclusive = cumprod(in, axis_, reverse_, !inclusive_, stream());
    auto cprod_inclusive = outputs[0];
    if (!inclusive_) {
      std::swap(cprod_exclusive, cprod_inclusive);
    }

    // Make the mask for the first zero
    auto z = array(0, in.dtype());
    auto eq_zero = equal(cprod_inclusive, z, stream());
    auto first_zero =
        logical_and(eq_zero, not_equal(cprod_exclusive, z, stream()), stream());

    auto to_partial_grad = [this, &cotangents](const array& arr) {
      return cumsum(
          multiply(arr, cotangents[0], stream()),
          axis_,
          !reverse_,
          inclusive_,
          stream());
    };

    auto cprod_with_one = cumprod(
        where(first_zero, array(1, in.dtype()), in, stream()),
        axis_,
        reverse_,
        inclusive_,
        stream());
    auto grad_with_one = to_partial_grad(cprod_with_one);
    auto grad = divide(to_partial_grad(outputs[0]), in, stream());
    return {where(
        first_zero,
        grad_with_one,
        where(eq_zero, z, grad, stream()),
        stream())};
  } else {
    // Can probably be implemented by equals and then cummax to make the mask
    throw std::runtime_error("VJP is not implemented for cumulative min/max");
  }
}

std::vector<array> Scan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(tangents.size() == 1);
  assert(argnums[0] == 0);

  if (reduce_type_ == Scan::Sum) {
    return {cumsum(tangents[0], axis_, reverse_, inclusive_, stream())};
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

std::vector<array> Scatter::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  switch (reduce_type_) {
    case Scatter::None:
    case Scatter::Sum:
    case Scatter::Max:
    case Scatter::Min:
      break;
    default:
      throw std::runtime_error(
          "[scatter] VJP not implemented for scatter_prod");
  }

  const array& result = outputs[0];
  const array& values = primals[0];
  const array& updates = primals.back();
  const std::vector<array> indices(primals.begin() + 1, primals.end() - 1);

  std::vector<array> vjps;
  for (auto num : argnums) {
    // Gradient wrt to the input array
    if (num == 0) {
      switch (reduce_type_) {
        case Scatter::None:
          // Scatter 0s to the locations that were updated with the updates
          vjps.push_back(scatter(
              cotangents[0],
              indices,
              zeros_like(updates, stream()),
              axes_,
              stream()));
          break;
        case Scatter::Sum:
          // The input array values are kept so they all get gradients
          vjps.push_back(cotangents[0]);
          break;
        case Scatter::Max:
        case Scatter::Min: {
          vjps.push_back(where(
              equal(result, values, stream()),
              cotangents[0],
              array(0, cotangents[0].dtype()),
              stream()));
          break;
        }
        default:
          // Should never reach here
          throw std::invalid_argument("");
      }
    } else if (num == primals.size() - 1) {
      switch (reduce_type_) {
        case Scatter::None:
        case Scatter::Sum: {
          // Gather the values from the cotangent
          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          vjps.push_back(
              gather(cotangents[0], indices, axes_, slice_sizes, stream()));
          break;
        }
        case Scatter::Max:
        case Scatter::Min: {
          auto slice_sizes = cotangents[0].shape();
          for (auto ax : axes_) {
            slice_sizes[ax] = 1;
          }
          auto gathered_cotan =
              gather(cotangents[0], indices, axes_, slice_sizes, stream());
          auto gathered_result =
              gather(result, indices, axes_, slice_sizes, stream());
          vjps.push_back(
              multiply(gathered_cotan, gathered_result == updates, stream()));
          break;
        }
        default: {
          // Should never reach here
          throw std::invalid_argument("");
        }
      }
    } else {
      throw std::invalid_argument(
          "[scatter] Cannot calculate VJP with respect to indices.");
    }
  }
  return vjps;
}

std::vector<array> Scatter::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("[scatter] JVP not yet implemented");
}

std::pair<std::vector<array>, std::vector<int>> Scatter::vmap(
    const std::vector<array>& inputs_,
    const std::vector<int>& vmap_axes) {
  assert(inputs_.size() >= 2);
  assert(inputs_.size() == vmap_axes.size());

  auto inputs = inputs_;

  auto scatter_axes = axes_;
  int src_ax = vmap_axes[0];

  auto vmap_ax_it = std::find_if(
      vmap_axes.begin(), vmap_axes.end(), [](int a) { return a >= 0; });
  auto vmap_ax = *vmap_ax_it;
  if (vmap_ax >= 0) {
    auto vmap_size = inputs[vmap_ax_it - vmap_axes.begin()].shape(vmap_ax);
    if (src_ax < 0) {
      src_ax = 0;
      inputs[0] =
          repeat(expand_dims(inputs[0], 0, stream()), vmap_size, 0, stream());
    }
    for (int i = 1; i < vmap_axes.size() - 1; ++i) {
      // vmap axis for indices goes to 0
      if (vmap_axes[i] >= 0) {
        inputs[i] = moveaxis(inputs[i], vmap_axes[i], 0, stream());
      }
      // insert a vmap axis and repeat
      if (vmap_axes[i] < 0) {
        auto idx_shape = inputs[i].shape();
        inputs[i] =
            repeat(expand_dims(inputs[i], 0, stream()), vmap_size, 0, stream());
      }
      // Adjust non-vmapped index axes to account for the extra vmap dimension.
      if (scatter_axes[i - 1] >= src_ax) {
        scatter_axes[i - 1]++;
      }
    }

    auto vmap_inds = arange(vmap_size, inputs[1].dtype(), stream());
    auto vmap_inds_shape = Shape(inputs[1].ndim(), 1);
    vmap_inds_shape[0] = vmap_inds.size();
    vmap_inds = reshape(vmap_inds, std::move(vmap_inds_shape), stream());
    inputs.insert(
        inputs.end() - 1, broadcast_to(vmap_inds, inputs[1].shape(), stream()));
    scatter_axes.push_back(src_ax);

    // Clone updates along the vmap dimension so they can be applied to each
    // source tensor in the vmap.
    auto& updates = inputs.back();
    if (vmap_axes.back() < 0) {
      updates = expand_dims(
          updates, {0, static_cast<int>(inputs[1].ndim())}, stream());
      updates = repeat(updates, vmap_size, 0, stream());
    } else {
      updates =
          expand_dims(updates, static_cast<int>(inputs[1].ndim()), stream());
      updates = moveaxis(updates, vmap_axes.back(), 0, stream());
    }
  }

  auto& shape = inputs[0].shape();
  auto dtype = inputs[0].dtype();
  auto out = array(
      shape,
      dtype,
      std::make_shared<Scatter>(stream(), reduce_type_, scatter_axes),
      std::move(inputs));

  return {{out}, {src_ax}};
}

std::vector<array> Sigmoid::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto& s = outputs[0];
  auto sprime =
      multiply(s, subtract(array(1.0f, s.dtype()), s, stream()), stream());
  return {multiply(cotangents[0], sprime, stream())};
}

std::vector<array> Sigmoid::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  auto s = sigmoid(primals[0], stream());
  auto sprime =
      multiply(s, subtract(array(1.0f, s.dtype()), s, stream()), stream());
  return {multiply(tangents[0], sprime, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Sigmoid::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{sigmoid(inputs[0], stream())}, axes};
}

std::vector<array> Sign::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sign::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {zeros(primals[0].shape(), primals[0].dtype(), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Sign::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{sign(inputs[0], stream())}, axes};
}

std::vector<array> Sin::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sin::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], cos(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Sin::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{sin(inputs[0], stream())}, axes};
}

std::vector<array> Sinh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sinh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {multiply(tangents[0], cosh(primals[0], stream()), stream())};
}

std::pair<std::vector<array>, std::vector<int>> Sinh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{sinh(inputs[0], stream())}, axes};
}

std::pair<std::vector<array>, std::vector<int>> Slice::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto start = start_indices_;
  auto stop = end_indices_;
  auto strides = strides_;
  auto ax = axes[0];
  auto& input = inputs[0];
  if (ax >= 0) {
    start.insert(start.begin() + ax, 0);
    stop.insert(stop.begin() + ax, input.shape(ax));
    strides.insert(strides.begin() + ax, 1);
  }
  return {{slice(input, start, stop, strides, stream())}, {ax}};
}

std::vector<array> Slice::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  // Check inputs
  assert(primals.size() == 1);
  auto out = zeros_like(primals[0], stream());
  return {slice_update(
      out, cotangents[0], start_indices_, end_indices_, strides_, stream())};
}

std::vector<array> Slice::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // Check inputs
  assert(primals.size() == 1);
  return {slice(tangents[0], start_indices_, end_indices_, strides_, stream())};
}

bool Slice::is_equivalent(const Primitive& other) const {
  const Slice& s_other = static_cast<const Slice&>(other);
  return (
      start_indices_ == s_other.start_indices_ &&
      end_indices_ == s_other.end_indices_ && strides_ == s_other.strides_);
}

std::pair<std::vector<array>, std::vector<int>> SliceUpdate::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 2);
  assert(axes.size() == 2);

  auto start = start_indices_;
  auto stop = end_indices_;
  auto strides = strides_;

  auto src = inputs[0];
  auto upd = inputs[1];

  auto src_ax = axes[0];
  auto upd_ax = axes[1];

  // No vmapping needed
  if (src_ax == -1 && upd_ax == -1) {
    return {{slice_update(src, upd, start, stop, strides, stream())}, {-1}};
  }

  // Broadcast src
  if (src_ax == -1) {
    src = expand_dims(src, upd_ax, stream());
    auto shape = src.shape();
    shape[upd_ax] = upd.shape(upd_ax);
    src = broadcast_to(src, shape, stream());
    src_ax = upd_ax;
  }

  // Broadcast upd
  if (upd_ax == -1) {
    upd = expand_dims(upd, src_ax, stream());
    upd_ax = src_ax;
  }

  if (src_ax != upd_ax) {
    upd = moveaxis(upd, upd_ax, src_ax, stream());
  }

  start.insert(start.begin() + src_ax, 0);
  stop.insert(stop.begin() + src_ax, src.shape(src_ax));
  strides.insert(strides.begin() + src_ax, 1);

  return {{slice_update(src, upd, start, stop, strides, stream())}, {src_ax}};
}

std::vector<array> SliceUpdate::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  // Check inputs
  assert(primals.size() == 2);

  auto& cotan = cotangents[0];
  auto& src = primals[0];
  auto& upd = primals[1];

  std::vector<array> vjps;

  for (int num : argnums) {
    // Vjp for source
    if (num == 0) {
      auto grad = slice_update(
          cotan,
          zeros_like(upd, stream()),
          start_indices_,
          end_indices_,
          strides_,
          stream());

      vjps.push_back(grad);
    }
    // Vjp fpr updates
    else {
      auto grad =
          slice(cotan, start_indices_, end_indices_, strides_, stream());

      vjps.push_back(grad);
    }
  }

  return vjps;
}

std::vector<array> SliceUpdate::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  // Check inputs
  assert(primals.size() == 2);
  return {slice_update(
      tangents[0],
      tangents[1],
      start_indices_,
      end_indices_,
      strides_,
      stream())};
}

bool SliceUpdate::is_equivalent(const Primitive& other) const {
  const SliceUpdate& s_other = static_cast<const SliceUpdate&>(other);
  return (
      start_indices_ == s_other.start_indices_ &&
      end_indices_ == s_other.end_indices_ && strides_ == s_other.strides_);
}

std::pair<std::vector<array>, std::vector<int>> Softmax::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  std::vector<int> softmax_axes;

  // We are vectorizing over an axis other than the last one so keep the
  // softmax axis unchanged
  if (axes[0] >= 0 && axes[0] < inputs[0].ndim() - 1) {
    softmax_axes.push_back(-1);
  } else {
    softmax_axes.push_back(-2);
  }
  return {{softmax(inputs[0], softmax_axes, precise_, stream())}, axes};
}

std::vector<array> Softmax::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(cotangents.size() == 1);
  auto& s = outputs[0];
  auto sv = multiply(s, cotangents[0], stream());
  return {subtract(
      sv,
      multiply(s, sum(sv, std::vector<int>{-1}, true, stream()), stream()))};
}

std::vector<array> Softmax::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto s = softmax(primals[0], std::vector<int>{-1}, precise_, stream());
  auto sv = multiply(s, tangents[0], stream());
  return {subtract(
      sv,
      multiply(s, sum(sv, std::vector<int>{-1}, true, stream()), stream()))};
}

bool Softmax::is_equivalent(const Primitive& other) const {
  const Softmax& s_other = static_cast<const Softmax&>(other);
  return precise_ == s_other.precise_;
}

std::pair<std::vector<array>, std::vector<int>> Sort::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {{sort(inputs[0], axis_ + axis_left, stream())}, axes};
}

std::vector<array> Sort::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Sort::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  auto sort_idx = argsort(primals[0], axis_, stream());
  auto out = take_along_axis(tangents[0], sort_idx, axis_, stream());
  return {out};
}

bool Sort::is_equivalent(const Primitive& other) const {
  const Sort& r_other = static_cast<const Sort&>(other);
  return axis_ == r_other.axis_;
}

std::pair<std::vector<array>, std::vector<int>> Split::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  int axis_left = axes[0] >= 0 && axes[0] <= axis_;
  return {{split(inputs[0], indices_, axis_ + axis_left, stream())}, axes};
}

std::vector<array> Split::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return {concatenate(cotangents, axis_, stream())};
}

std::vector<array> Split::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  return split(tangents[0], indices_, axis_, stream());
}

bool Split::is_equivalent(const Primitive& other) const {
  const Split& s_other = static_cast<const Split&>(other);
  return axis_ == s_other.axis_ && indices_ == s_other.indices_;
}

std::vector<array> Square::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Square::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return {multiply(
      primals[0],
      multiply(array(2, primals[0].dtype()), tangents[0], stream()),
      stream())};
}

std::pair<std::vector<array>, std::vector<int>> Square::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{square(inputs[0], stream())}, axes};
}

std::vector<array> Sqrt::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 1);
  assert(cotangents.size() == 1);
  auto dtype = primals[0].dtype();
  if (recip_) {
    auto one_over_x_root_x = divide(outputs[0], primals[0], stream());
    return {multiply(
        multiply(array(-0.5, dtype), cotangents[0], stream()),
        one_over_x_root_x,
        stream())};
  } else {
    return {divide(
        multiply(array(0.5, dtype), cotangents[0], stream()),
        outputs[0],
        stream())};
  }
}

std::vector<array> Sqrt::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  if (recip_) {
    return vjp(primals, tangents, argnums, {rsqrt(primals[0], stream())});
  } else {
    return vjp(primals, tangents, argnums, {sqrt(primals[0], stream())});
  }
}

std::pair<std::vector<array>, std::vector<int>> Sqrt::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  if (recip_) {
    return {{rsqrt(inputs[0], stream())}, axes};
  }
  return {{sqrt(inputs[0], stream())}, axes};
}

bool Sqrt::is_equivalent(const Primitive& other) const {
  const Sqrt& s_other = static_cast<const Sqrt&>(other);
  return recip_ == s_other.recip_;
}

std::pair<std::vector<array>, std::vector<int>> StopGradient::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  return {{stop_gradient(inputs[0], stream())}, axes};
}

std::vector<array> Subtract::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  for (auto arg : argnums) {
    auto vjp = cotangents[0];
    if (arg == 1) {
      vjp = negative(vjp, stream());
    }
    vjps.push_back(vjp);
  }
  return vjps;
}

std::vector<array> Subtract::jvp(
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
  return {out};
}

std::pair<std::vector<array>, std::vector<int>> Subtract::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto [a, b, to_ax] = vmap_binary_op(inputs, axes, stream());
  return {{subtract(a, b, stream())}, {to_ax}};
}

std::vector<array> Squeeze::vjp(
    const std::vector<array>&,
    const std::vector<array>& cotangents,
    const std::vector<int>&,
    const std::vector<array>&) {
  return {expand_dims(cotangents[0], axes_, stream())};
}

std::vector<array> Squeeze::jvp(
    const std::vector<array>&,
    const std::vector<array>& tangents,
    const std::vector<int>&) {
  return {squeeze(tangents[0], axes_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Squeeze::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0];
  auto squeeze_axes = axes_;
  for (auto& s : squeeze_axes) {
    if (s >= axes[0]) {
      s++;
    } else {
      ax--;
    }
  }
  return {{squeeze(inputs[0], std::move(squeeze_axes), stream())}, {ax}};
}

bool Squeeze::is_equivalent(const Primitive& other) const {
  const Squeeze& a_other = static_cast<const Squeeze&>(other);
  return (axes_ == a_other.axes_);
}

Shape Squeeze::output_shape(const array& input, const std::vector<int>& axes) {
  Shape shape;
  for (int i = 0, j = 0; i < input.ndim(); ++i) {
    if (j < axes.size() && i == axes[j]) {
      j++;
    } else {
      shape.push_back(input.shape(i));
    }
  }
  return shape;
}

std::vector<Shape> Squeeze::output_shapes(const std::vector<array>& inputs) {
  return {Squeeze::output_shape(inputs[0], axes_)};
}

std::vector<array> Tan::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Tan::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cos_sq = square(cos(primals[0], stream()), stream());
  return {divide(tangents[0], cos_sq, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Tan::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{tan(inputs[0], stream())}, axes};
}

std::vector<array> Tanh::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Tanh::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  array cosh_sq = square(cosh(primals[0], stream()), stream());
  return {divide(tangents[0], cosh_sq, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Tanh::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{tanh(inputs[0], stream())}, axes};
}

std::vector<array> BlockMaskedMM::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  /////////////////////////////////////////////////////////////////////////////
  // The operation that is done w/o intermediates by the primitive is
  //    - tm = (M + block_size - 1) // block_size; MP = tm * block_size;
  //    - tn = (N + block_size - 1) // block_size; NP = tn * block_size;
  //    - tm = (K + block_size - 1) // block_size; KP = tk * block_size;
  //    - mask_b <- mask broadcasted to block sizes
  //    - A_m = A [..., M, K] * mask_b_lhs [..., MP, KP]
  //    - B_m = B [..., K, N] * mask_b_rhs [..., KP, MP]
  //    - C = A_m [..., M, K]  @ B_m [..., K, N]
  //    - C_m = C [..., M, N] * mask_b_out [..., MP, NP]
  //
  // The grads are therefore
  //    - dC_m = cotan [..., M, N]
  //    - dmask_b_out = cotan [..., M, N] * C [..., M, N]
  //    - dC = cotan [..., M, N] * mask_b_out [..., MP, NP]
  //    - dA_m = dC [..., M, N] @ B_m.T [..., N, K]
  //    - dB_m = A_m.T [..., K, M] @ dC [..., M, N]
  //    - dA = dA_m * mask_b_lhs [..., MP, KP]
  //    - dB = dB_m * mask_b_rhs [..., KP, MP]
  //    - dmask_b_lhs = dA_m [..., M, K] * A [..., M, K] // need [..., MP, KP]
  //    - dmask_b_rhs = dB_m [..., K, N] * B [..., K, N] // need [..., KP, NP]
  //
  // Observations:
  //  * If dmask_b_lhs is not needed, then dA can be calulated in one go as a
  //    as a block_masked_mm with mask_b_lhs as the out_mask without needing to
  //    materialize the intermediate dA_m. Similar for dB.
  //  * If dmask_b_lhs is needed, we need to materialize dA_m directly and then
  //    point-wise multiply with A. But the output needs to be padded

  std::vector<array> vjps;
  auto& cotan = cotangents[0];
  std::vector<int> reorder(cotan.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  std::iter_swap(reorder.end() - 1, reorder.end() - 2);

  bool has_op_mask = primals.size() > 3;
  bool has_out_mask = primals.size() == 3 || primals.size() == 5;

  const int op_mask_idx = has_out_mask ? 3 : 2;
  bool needs_lhs_mask_vjp = has_op_mask;
  bool needs_rhs_mask_vjp = has_op_mask;
  bool needs_lhs_vjp = false;
  bool needs_rhs_vjp = false;

  for (auto arg : argnums) {
    needs_lhs_vjp = arg == 0;
    needs_rhs_vjp = arg == 1;
    needs_lhs_mask_vjp = arg == op_mask_idx;
    needs_rhs_mask_vjp = arg == op_mask_idx + 1;
  }

  if ((needs_lhs_mask_vjp && primals[op_mask_idx].dtype() == bool_) ||
      (needs_rhs_mask_vjp && primals[op_mask_idx + 1].dtype() == bool_)) {
    throw std::invalid_argument(
        "[BlockMaskedMM] Cannot calculate VJP with respect to boolean masks.");
  }

  auto expand_mask = [&](array mask, int Y, int X) {
    // Exapnd mask
    auto mask_reshape = mask.shape();
    mask = expand_dims(mask, {-3, -1}, stream());
    auto mask_shape = mask.shape();
    int mask_ndim = mask_shape.size();

    // Broadcast mask
    mask_shape[mask_ndim - 1] = block_size_;
    mask_shape[mask_ndim - 3] = block_size_;
    mask = broadcast_to(mask, mask_shape, stream());

    // Reshape mask to squeeze in braodcasted dims
    mask_ndim = mask_reshape.size();
    mask_reshape[mask_ndim - 2] *= block_size_;
    mask_reshape[mask_ndim - 1] *= block_size_;
    mask = reshape(mask, mask_reshape, stream());

    // Slice mask
    mask_reshape[mask_ndim - 2] = Y;
    mask_reshape[mask_ndim - 1] = X;
    mask = slice(mask, Shape(mask_ndim, 0), mask_reshape, stream());

    return mask;
  };

  array zero = array(0, cotan.dtype());

  auto multiply_pad_reduce = [&](array p, array q, int align_Y, int align_X) {
    // Multiply with cotan
    auto r = multiply(p, q, stream());

    // Pad if needed
    if ((align_Y != 0) || (align_X != 0)) {
      r = pad(
          r, {-2, -1}, {0, 0}, {align_Y, align_X}, zero, "constant", stream());
    }

    // Reshape
    Shape r_reshape(r.shape().begin(), r.shape().end() - 2);
    r_reshape.push_back(r.shape(-2) / block_size_);
    r_reshape.push_back(block_size_);
    r_reshape.push_back(r.shape(-1) / block_size_);
    r_reshape.push_back(block_size_);
    r = reshape(r, r_reshape, stream());

    // Reduce
    return sum(r, {-3, -1}, false, stream());
  };

  // Prepare for padding if needed
  const int M = cotan.shape(-2);
  const int N = cotan.shape(-1);
  const int K = primals[0].shape(-1);
  const int tm = (M + block_size_ - 1) / block_size_;
  const int tn = (N + block_size_ - 1) / block_size_;
  const int tk = (K + block_size_ - 1) / block_size_;
  const int align_M = tm * block_size_ - M;
  const int align_N = tn * block_size_ - N;
  const int align_K = tk * block_size_ - K;

  // Potential intermediates
  array unmasked_lhs_grad = primals[0];
  array unmasked_rhs_grad = primals[1];

  bool unmasked_lhs_grad_calculated = false;
  bool unmasked_rhs_grad_calculated = false;

  for (auto arg : argnums) {
    if (arg == 0) {
      // M X N * (K X N).T -> M X K
      auto b_t = transpose(primals[1], reorder, stream());
      auto out_mask =
          has_out_mask ? std::make_optional<array>(primals[2]) : std::nullopt;
      auto lhs_mask = has_op_mask && !needs_lhs_mask_vjp
          ? std::make_optional<array>(primals[op_mask_idx])
          : std::nullopt;
      auto rhs_mask_t = has_op_mask
          ? std::make_optional<array>(
                transpose(primals[op_mask_idx + 1], reorder, stream()))
          : std::nullopt;

      auto grad = block_masked_mm(
          cotan, b_t, block_size_, lhs_mask, out_mask, rhs_mask_t, stream());

      if (needs_lhs_mask_vjp) {
        unmasked_lhs_grad = grad;
        unmasked_lhs_grad_calculated = true;
        auto exp_mask = expand_mask(primals[op_mask_idx], M, K);
        grad = multiply(grad, exp_mask, stream());
      }

      vjps.push_back(grad);

    } else if (arg == 1) {
      // (M X K).T * M X N -> K X N
      auto a_t = transpose(primals[0], reorder, stream());
      auto out_mask =
          has_out_mask ? std::make_optional<array>(primals[2]) : std::nullopt;
      auto lhs_mask_t = has_op_mask
          ? std::make_optional<array>(
                transpose(primals[op_mask_idx], reorder, stream()))
          : std::nullopt;
      auto rhs_mask = has_op_mask && !needs_rhs_mask_vjp
          ? std::make_optional<array>(primals[op_mask_idx + 1])
          : std::nullopt;

      auto grad = block_masked_mm(
          a_t, cotan, block_size_, rhs_mask, lhs_mask_t, out_mask, stream());

      if (needs_rhs_mask_vjp) {
        unmasked_rhs_grad = grad;
        unmasked_rhs_grad_calculated = true;
        auto exp_mask = expand_mask(primals[op_mask_idx + 1], K, N);
        grad = multiply(grad, exp_mask, stream());
      }

      vjps.push_back(grad);

    } else if (arg == 2 && has_out_mask) {
      // Produce the forward result
      auto lhs_mask = has_op_mask
          ? std::make_optional<array>(primals[op_mask_idx])
          : std::nullopt;
      auto rhs_mask = has_op_mask
          ? std::make_optional<array>(primals[op_mask_idx + 1])
          : std::nullopt;

      auto C = block_masked_mm(
          primals[0],
          primals[1],
          block_size_,
          primals[2],
          lhs_mask,
          rhs_mask,
          stream());

      // Multiply, Pad and Reduce if needed
      auto grad = multiply_pad_reduce(cotan, C, align_M, align_N);
      vjps.push_back(grad);

    } else if (arg == op_mask_idx && has_op_mask) {
      if (!unmasked_lhs_grad_calculated) {
        // (M X K).T * M X N -> K X N
        auto b_t = transpose(primals[1], reorder, stream());
        auto out_mask =
            has_out_mask ? std::make_optional<array>(primals[2]) : std::nullopt;
        auto rhs_mask_t =
            transpose(primals[op_mask_idx + 1], reorder, stream());

        unmasked_lhs_grad = block_masked_mm(
            cotan,
            b_t,
            block_size_,
            std::nullopt,
            out_mask,
            rhs_mask_t,
            stream());

        unmasked_lhs_grad_calculated = true;
      }

      // Multiply, Pad and Reduce if needed
      auto grad =
          multiply_pad_reduce(primals[0], unmasked_lhs_grad, align_M, align_K);
      vjps.push_back(grad);

    } else if (arg == op_mask_idx + 1 && has_op_mask) {
      if (!unmasked_rhs_grad_calculated) {
        // (M X K).T * M X N -> K X N
        auto a_t = transpose(primals[0], reorder, stream());
        auto out_mask =
            has_out_mask ? std::make_optional<array>(primals[2]) : std::nullopt;
        auto lhs_mask_t = transpose(primals[op_mask_idx], reorder, stream());

        unmasked_rhs_grad = block_masked_mm(
            a_t,
            cotan,
            block_size_,
            std::nullopt,
            lhs_mask_t,
            out_mask,
            stream());

        unmasked_rhs_grad_calculated = true;
      }

      // Multiply, Pad and Reduce if needed
      auto grad =
          multiply_pad_reduce(primals[1], unmasked_rhs_grad, align_K, align_N);
      vjps.push_back(grad);

    } else {
      throw std::invalid_argument(
          "[BlockMaskedMM] Cannot calculate VJP with respect to masks.");
    }
  }
  return vjps;
}

std::vector<array> GatherMM::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  std::vector<array> vjps;
  auto& cotan = cotangents[0];

  auto& lhs_indices = primals[2];
  auto& rhs_indices = primals[3];

  int M = cotan.shape(-2);
  int N = cotan.shape(-1);
  int K = primals[0].shape(-1);

  for (auto arg : argnums) {
    if (arg == 0) {
      // M X N * (K X N).T -> M X K
      auto base = zeros_like(primals[0], stream());
      auto bt = swapaxes(primals[1], -1, -2, stream());

      auto base_shape = base.shape();
      base = reshape(base, {-1, M, K}, stream());

      // g : (out_batch_shape) + (M, K)
      auto g = gather_mm(cotan, bt, std::nullopt, rhs_indices, stream());
      g = expand_dims(g, -3, stream());
      auto gacc = scatter_add(base, lhs_indices, g, 0, stream());

      vjps.push_back(reshape(gacc, base_shape, stream()));

    } else if (arg == 1) {
      // (M X K).T * M X N -> K X N
      auto base = zeros_like(primals[1], stream());
      auto at = swapaxes(primals[0], -1, -2, stream());

      auto base_shape = base.shape();
      base = reshape(base, {-1, K, N}, stream());

      // g : (out_batch_shape) + (K, N)
      auto g = gather_mm(at, cotan, lhs_indices, std::nullopt, stream());
      g = expand_dims(g, -3, stream());
      auto gacc = scatter_add(base, rhs_indices, g, 0, stream());

      vjps.push_back(reshape(gacc, base_shape, stream()));
    } else {
      throw std::invalid_argument(
          "[GatherMM] Cannot calculate VJP with respect to indices.");
    }
  }
  return vjps;
}

bool BlockMaskedMM::is_equivalent(const Primitive& other) const {
  const BlockMaskedMM& a_other = static_cast<const BlockMaskedMM&>(other);
  return (block_size_ == a_other.block_size_);
}

std::vector<array> Transpose::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  std::vector<int> iaxes(axes_.size());
  for (int i = 0; i < axes_.size(); ++i) {
    iaxes[axes_[i]] = i;
  }
  return {transpose(cotangents[0], iaxes, stream())};
}

std::vector<array> Transpose::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(tangents.size() == 1);
  return {transpose(tangents[0], axes_, stream())};
}

std::pair<std::vector<array>, std::vector<int>> Transpose::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto vdim = axes[0];
  if (vdim >= 0) {
    for (auto& dim : axes_) {
      if (dim >= vdim) {
        dim++;
      }
    }
    axes_.insert(axes_.begin() + vdim, vdim);
  }
  return {{transpose(inputs[0], axes_, stream())}, {vdim}};
}

bool Transpose::is_equivalent(const Primitive& other) const {
  const Transpose& t_other = static_cast<const Transpose&>(other);
  return axes_ == t_other.axes_;
}

std::vector<Shape> Transpose::output_shapes(const std::vector<array>& inputs) {
  auto& in = inputs[0];
  Shape shape(in.ndim(), 0);
  for (int i = 0; i < axes_.size(); ++i) {
    shape[i] = in.shape()[axes_[i]];
  }
  return {std::move(shape)};
}

std::pair<std::vector<array>, std::vector<int>> NumberOfElements::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);

  std::vector<int> new_axes = axes_;
  auto vdim = axes[0];
  if (vdim >= 0) {
    for (auto& dim : new_axes) {
      if (dim >= vdim) {
        dim++;
      }
    }
  }

  array out = array(
      {},
      dtype_,
      std::make_shared<NumberOfElements>(stream(), new_axes, inverted_, dtype_),
      inputs);

  return {{out}, {-1}};
}

bool NumberOfElements::is_equivalent(const Primitive& other) const {
  const NumberOfElements& n_other = static_cast<const NumberOfElements&>(other);
  return axes_ == n_other.axes_ && inverted_ == n_other.inverted_ &&
      dtype_ == n_other.dtype_;
}

std::pair<std::vector<array>, std::vector<int>> SVD::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::svd(a, stream())}, {ax, ax, ax}};
}

std::pair<std::vector<array>, std::vector<int>> Inverse::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::inv(a, stream())}, {ax}};
}

std::pair<std::vector<array>, std::vector<int>> View::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  return {{view(inputs[0], dtype_, stream())}, axes};
}

void View::print(std::ostream& os) {
  os << "View " << dtype_;
}

bool View::is_equivalent(const Primitive& other) const {
  const View& a_other = static_cast<const View&>(other);
  return (dtype_ == a_other.dtype_);
}

std::pair<std::vector<array>, std::vector<int>> Hadamard::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  assert(inputs.size() == 1);
  assert(axes.size() == 1);
  auto& s = stream();
  if (axes[0] == inputs[0].ndim() - 1) {
    auto a = moveaxis(inputs[0], axes[0], 0, s);
    auto b = hadamard_transform(a, scale_, s);
    return {{b}, {0}};
  }
  return {{hadamard_transform(inputs[0], scale_, s)}, axes};
}

std::vector<array> Hadamard::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return jvp(primals, cotangents, argnums);
}

std::vector<array> Hadamard::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  assert(primals.size() == 1);
  assert(argnums.size() == 1);
  return {hadamard_transform(tangents[0], scale_, stream())};
}

bool Hadamard::is_equivalent(const Primitive& other) const {
  const Hadamard& h_other = static_cast<const Hadamard&>(other);
  return scale_ == h_other.scale_;
}

} // namespace mlx::core
