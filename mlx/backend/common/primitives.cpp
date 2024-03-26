// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/arange.h"
#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/ops.h"
#include "mlx/backend/common/threefry.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void Abs::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), unsignedinteger)) {
    // No-op for unsigned types
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, detail::Abs());
  }
}

void Arange::eval(const std::vector<array>& inputs, array& out) {
  arange(inputs, out, start_, step_);
}

void ArcCos::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcCos());
  } else {
    throw std::invalid_argument(
        "[arccos] Cannot compute inverse cosine of elements in array"
        " with non floating point type.");
  }
}

void ArcCosh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcCosh());
  } else {
    throw std::invalid_argument(
        "[arccosh] Cannot compute inverse hyperbolic cosine of elements in"
        " array with non floating point type.");
  }
}

void ArcSin::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcSin());
  } else {
    throw std::invalid_argument(
        "[arcsin] Cannot compute inverse sine of elements in array"
        " with non floating point type.");
  }
}

void ArcSinh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcSinh());
  } else {
    throw std::invalid_argument(
        "[arcsinh] Cannot compute inverse hyperbolic sine of elements in"
        " array with non floating point type.");
  }
}

void ArcTan::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcTan());
  } else {
    throw std::invalid_argument(
        "[arctan] Cannot compute inverse tangent of elements in array"
        " with non floating point type.");
  }
}

void ArcTanh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::ArcTanh());
  } else {
    throw std::invalid_argument(
        "[arctanh] Cannot compute inverse hyperbolic tangent of elements in"
        " array with non floating point type.");
  }
}

void AsType::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  CopyType ctype = in.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy(in, out, ctype);
}

void AsStrided::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& in = inputs[0];

  if (!in.flags().row_contiguous) {
    // Just ensuring that inputs[0] came from the ops which would ensure the
    // input is row contiguous.
    throw std::runtime_error(
        "AsStrided must be used with row contiguous arrays only.");
  }

  // Compute the flags given the shape and strides
  bool row_contiguous = true, col_contiguous = true;
  size_t r = 1, c = 1;
  for (int i = strides_.size() - 1, j = 0; i >= 0; i--, j++) {
    row_contiguous &= (r == strides_[i]) || (shape_[i] == 1);
    col_contiguous &= (c == strides_[j]) || (shape_[j] == 1);
    r *= shape_[i];
    c *= shape_[j];
  }
  auto flags = in.flags();
  // TODO: Compute the contiguous flag in a better way cause now we are
  //       unnecessarily strict.
  flags.contiguous = row_contiguous || col_contiguous;
  flags.row_contiguous = row_contiguous;
  flags.col_contiguous = col_contiguous;

  // There is no easy way to compute the actual data size so we use out.size().
  // The contiguous flag will almost certainly not be set so no code should
  // rely on data_size anyway.
  size_t data_size = out.size();

  return out.copy_shared_buffer(in, strides_, flags, data_size, offset_);
}

void Broadcast::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }
  std::vector<size_t> strides(out.ndim(), 0);
  int diff = out.ndim() - in.ndim();
  for (int i = in.ndim() - 1; i >= 0; --i) {
    strides[i + diff] = (in.shape()[i] == 1) ? 0 : in.strides()[i];
  }
  auto flags = in.flags();
  if (out.size() > in.size()) {
    flags.row_contiguous = flags.col_contiguous = false;
  }
  out.copy_shared_buffer(in, strides, flags, in.data_size());
}

void Ceil::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Ceil());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Concatenate::eval(const std::vector<array>& inputs, array& out) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis_));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis_] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_inplace(inputs[i], out_slice, CopyType::GeneralGeneral);
  }
}

void Copy::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.copy_shared_buffer(inputs[0]);
}

void Cos::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Cos());
  } else {
    throw std::invalid_argument(
        "[cos] Cannot compute cosine of elements in array"
        " with non floating point type.");
  }
}

void Cosh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Cosh());
  } else {
    throw std::invalid_argument(
        "[cosh] Cannot compute hyperbolic cosine of elements in array"
        " with non floating point type.");
  }
}

void CustomVJP::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() > outputs.size());
  for (int i = 0, j = inputs.size() - outputs.size(); i < outputs.size();
       i++, j++) {
    outputs[i].copy_shared_buffer(inputs[j]);
  }
}

void Depends::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() > outputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i].copy_shared_buffer(inputs[i]);
  }
}

void NumberOfElements::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  double numel = 1;
  for (auto ax : axes_) {
    numel *= inputs[0].shape(ax);
  }

  if (inverted_) {
    numel = 1.0 / numel;
  }

  switch (out.dtype()) {
    case bool_:
      *out.data<bool>() = static_cast<bool>(numel);
      break;
    case uint8:
      *out.data<uint8_t>() = static_cast<uint8_t>(numel);
      break;
    case uint16:
      *out.data<uint16_t>() = static_cast<uint16_t>(numel);
      break;
    case uint32:
      *out.data<uint32_t>() = static_cast<uint32_t>(numel);
      break;
    case uint64:
      *out.data<uint64_t>() = static_cast<uint64_t>(numel);
      break;
    case int8:
      *out.data<int8_t>() = static_cast<int8_t>(numel);
      break;
    case int16:
      *out.data<int16_t>() = static_cast<int16_t>(numel);
      break;
    case int32:
      *out.data<int32_t>() = static_cast<int32_t>(numel);
      break;
    case int64:
      *out.data<int64_t>() = static_cast<int64_t>(numel);
      break;
    case float16:
      *out.data<float16_t>() = static_cast<float16_t>(numel);
      break;
    case float32:
      *out.data<float>() = static_cast<float>(numel);
      break;
    case bfloat16:
      *out.data<bfloat16_t>() = static_cast<bfloat16_t>(numel);
      break;
    case complex64:
      *out.data<complex64_t>() = static_cast<complex64_t>(numel);
      break;
  }
}

void Erf::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (out.dtype()) {
    case float32:
      unary_op<float>(in, out, detail::Erf());
      break;
    case float16:
      unary_op<float16_t>(in, out, detail::Erf());
      break;
    case bfloat16:
      unary_op<bfloat16_t>(in, out, detail::Erf());
      break;
    default:
      throw std::invalid_argument(
          "[erf] Error function only defined for arrays"
          " with real floating point type.");
  }
}

void ErfInv::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (out.dtype()) {
    case float32:
      unary_op<float>(in, out, detail::ErfInv());
      break;
    case float16:
      unary_op<float16_t>(in, out, detail::ErfInv());
      break;
    case bfloat16:
      unary_op<bfloat16_t>(in, out, detail::ErfInv());
      break;
    default:
      throw std::invalid_argument(
          "[erf_inv] Inverse error function only defined for arrays"
          " with real floating point type.");
  }
}

void Exp::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Exp());
  } else {
    throw std::invalid_argument(
        "[exp] Cannot exponentiate elements in array"
        " with non floating point type.");
  }
}

void Floor::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Floor());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Full::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  assert(in.dtype() == out.dtype());
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy(in, out, ctype);
}

void Log::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    switch (base_) {
      case Base::e:
        unary_fp(in, out, detail::Log());
        break;
      case Base::two:
        unary_fp(in, out, detail::Log2());
        break;
      case Base::ten:
        unary_fp(in, out, detail::Log10());
        break;
    }
  } else {
    throw std::invalid_argument(
        "[log] Cannot compute log of elements in array with"
        " non floating point type.");
  }
}

void Log1p::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Log1p());
  } else {
    throw std::invalid_argument(
        "[log1p] Cannot compute log of elements in array with"
        " non floating point type.");
  }
}

void LogicalNot::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::LogicalNot());
}

void LogicalAnd::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalAnd requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalAnd());
}

void LogicalOr::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2); // LogicalOr requires two input arrays
  auto& in1 = inputs[0];
  auto& in2 = inputs[1];
  binary(in1, in2, out, detail::LogicalOr());
}

void Negative::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Negative());
}

void Pad::eval(const std::vector<array>& inputs, array& out) {
  // Inputs must be base input array and scalar val array
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  // Padding value must be a scalar
  assert(val.size() == 1);

  // Padding value, input and output must be of the same type
  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  // Fill output with val
  copy(val, out, CopyType::Scalar);

  // Find offset for start of input values
  size_t data_offset = 0;
  for (int i = 0; i < axes_.size(); i++) {
    auto ax = axes_[i] < 0 ? out.ndim() + axes_[i] : axes_[i];
    data_offset += out.strides()[ax] * low_pad_size_[i];
  }

  // Extract slice from output where input will be pasted
  array out_slice(in.shape(), out.dtype(), nullptr, {});
  out_slice.copy_shared_buffer(
      out, out.strides(), out.flags(), out_slice.size(), data_offset);

  // Copy input values into the slice
  copy_inplace(in, out_slice, CopyType::GeneralGeneral);
}

void RandomBits::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto kptr = inputs[0].data<uint32_t>();
  auto cptr = out.data<char>();
  size_t out_skip = (bytes_per_key + 4 - 1) / 4;
  auto half_size = out_skip / 2;
  bool even = out_skip % 2 == 0;
  for (int i = 0; i < num_keys; ++i, cptr += bytes_per_key) {
    auto ptr = reinterpret_cast<uint32_t*>(cptr);
    // Get ith key
    auto kidx = 2 * i;
    auto k1_elem = elem_to_loc(kidx, keys.shape(), keys.strides());
    auto k2_elem = elem_to_loc(kidx + 1, keys.shape(), keys.strides());
    auto key = std::make_pair(kptr[k1_elem], kptr[k2_elem]);

    std::pair<uintptr_t, uintptr_t> count{0, half_size + !even};
    for (; count.first + 1 < half_size; count.first++, count.second++) {
      std::tie(ptr[count.first], ptr[count.second]) =
          random::threefry2x32_hash(key, count);
    }
    if (count.first < half_size) {
      auto rb = random::threefry2x32_hash(key, count);
      ptr[count.first++] = rb.first;
      if (bytes_per_key % 4 > 0) {
        std::copy(
            reinterpret_cast<char*>(&rb.second),
            reinterpret_cast<char*>(&rb.second) + bytes_per_key % 4,
            cptr + 4 * count.second);
      } else {
        ptr[count.second] = rb.second;
      }
    }
    if (!even) {
      count.second = 0;
      ptr[half_size] = random::threefry2x32_hash(key, count).first;
    }
  }
}

std::pair<bool, std::vector<size_t>> Reshape::prepare_reshape(
    const array& in,
    const array& out) {
  // Special case for empty arrays or row contiguous arrays
  if (in.size() == 0 || in.flags().row_contiguous) {
    return {false, out.strides()};
  }

  // Special case for scalars
  if (in.ndim() == 0) {
    std::vector<size_t> out_strides(out.ndim(), 0);
    return {false, out_strides};
  }

  // Firstly let's collapse all the contiguous dimensions of the input
  auto [shape, _strides] = collapse_contiguous_dims(in);
  auto& strides = _strides[0];

  // If shapes fit exactly in the contiguous dims then no copy is necessary so
  // let's check.
  std::vector<size_t> out_strides;
  bool copy_necessary = false;
  int j = 0;
  for (int i = 0; i < out.ndim(); i++) {
    int N = out.shape(i);
    if (j < shape.size() && shape[j] % N == 0) {
      shape[j] /= N;
      out_strides.push_back(shape[j] * strides[j]);
      j += (shape[j] == 1);
    } else if (N == 1) {
      // i > 0 because otherwise j < shape.size() && shape[j] % 1 == 0
      out_strides.push_back(out_strides.back());
    } else {
      copy_necessary = true;
      break;
    }
  }

  return {copy_necessary, out_strides};
}

void Reshape::shared_buffer_reshape(
    const array& in,
    const std::vector<size_t>& out_strides,
    array& out) {
  auto flags = in.flags();
  if (flags.row_contiguous) {
    // For row contiguous reshapes:
    // - Shallow copy the buffer
    // - If reshaping into a vector (all singleton dimensions except one) it
    //    becomes col contiguous again.
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
  }
  out.copy_shared_buffer(in, out_strides, flags, in.data_size());
}

void Reshape::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];

  auto [copy_necessary, out_strides] = prepare_reshape(in, out);

  if (copy_necessary) {
    copy(in, out, in.data_size() == 1 ? CopyType::Scalar : CopyType::General);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

void Round::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (issubdtype(in.dtype(), inexact)) {
    unary_fp(in, out, detail::Round());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sigmoid::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Sigmoid());
  } else {
    throw std::invalid_argument(
        "[sigmoid] Cannot sigmoid of elements in array with"
        " non floating point type.");
  }
}

void Sign::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.dtype() == bool_) {
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, detail::Sign());
  }
}

void Sin::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Sin());
  } else {
    throw std::invalid_argument(
        "[sin] Cannot compute sine of elements in array"
        " with non floating point type.");
  }
}

void Sinh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Sinh());
  } else {
    throw std::invalid_argument(
        "[sinh] Cannot compute hyperbolic sine of elements in array"
        " with non floating point type.");
  }
}

std::tuple<bool, int64_t, std::vector<int64_t>> Slice::prepare_slice(
    const array& in) {
  int64_t data_offset = 0;
  bool copy_needed = false;
  std::vector<int64_t> inp_strides(in.ndim(), 0);
  for (int i = 0; i < in.ndim(); ++i) {
    data_offset += start_indices_[i] * in.strides()[i];
    inp_strides[i] = in.strides()[i] * strides_[i];

    copy_needed |= strides_[i] < 0;
  }

  return std::make_tuple(copy_needed, data_offset, inp_strides);
}

void Slice::shared_buffer_slice(
    const array& in,
    const std::vector<size_t>& out_strides,
    size_t data_offset,
    array& out) {
  // Compute row/col contiguity
  auto [data_size, is_row_contiguous, is_col_contiguous] =
      check_contiguity(out.shape(), out_strides);

  auto flags = in.flags();
  flags.row_contiguous = is_row_contiguous;
  flags.col_contiguous = is_col_contiguous;

  if (data_size == 1) {
    // Broadcasted scalar array is contiguous.
    flags.contiguous = true;
  } else if (data_size == in.data_size()) {
    // Means we sliced a broadcasted dimension so leave the "no holes" flag
    // alone.
  } else {
    // We sliced something. So either we are row or col contiguous or we
    // punched a hole.
    flags.contiguous &= flags.row_contiguous || flags.col_contiguous;
  }

  out.copy_shared_buffer(in, out_strides, flags, data_size, data_offset);
}

void Slice::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];

  // Calculate out strides, initial offset and if copy needs to be made
  auto [copy_needed, data_offset, inp_strides] = prepare_slice(in);

  // Do copy if needed
  if (copy_needed) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    std::vector<int64_t> ostrides{out.strides().begin(), out.strides().end()};
    copy_inplace<int64_t>(
        /* const array& src = */ in,
        /* array& dst = */ out,
        /* const std::vector<int>& data_shape = */ out.shape(),
        /* const std::vector<stride_t>& i_strides = */ inp_strides,
        /* const std::vector<stride_t>& o_strides = */ ostrides,
        /* int64_t i_offset = */ data_offset,
        /* int64_t o_offset = */ 0,
        /* CopyType ctype = */ CopyType::General);
  } else {
    std::vector<size_t> ostrides{inp_strides.begin(), inp_strides.end()};
    shared_buffer_slice(in, ostrides, data_offset, out);
  }
}

std::tuple<int64_t, std::vector<int64_t>> SliceUpdate::prepare_slice(
    const array& in) {
  int64_t data_offset = 0;
  std::vector<int64_t> inp_strides(in.ndim(), 0);
  for (int i = 0; i < in.ndim(); ++i) {
    data_offset += start_indices_[i] * in.strides()[i];
    inp_strides[i] = in.strides()[i] * strides_[i];
  }

  return std::make_tuple(data_offset, inp_strides);
}

void SliceUpdate::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  // Check if materialization is needed
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype);

  // Calculate out strides, initial offset and if copy needs to be made
  auto [data_offset, out_strides] = prepare_slice(out);

  // Do copy
  std::vector<int64_t> upd_strides{upd.strides().begin(), upd.strides().end()};
  copy_inplace<int64_t>(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd_strides,
      /* const std::vector<stride_t>& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral);
}

void Split::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);

  auto& in = inputs[0];

  auto compute_new_flags = [](const auto& shape,
                              const auto& strides,
                              size_t in_data_size,
                              auto flags) {
    size_t data_size = 1;
    size_t f_stride = 1;
    size_t b_stride = 1;
    flags.row_contiguous = true;
    flags.col_contiguous = true;
    for (int i = 0, ri = shape.size() - 1; ri >= 0; i++, ri--) {
      flags.col_contiguous &= strides[i] == f_stride || shape[i] == 1;
      flags.row_contiguous &= strides[ri] == b_stride || shape[ri] == 1;
      f_stride *= shape[i];
      b_stride *= shape[ri];
      if (strides[i] > 0) {
        data_size *= shape[i];
      }
    }

    if (data_size == 1) {
      // Broadcasted scalar array is contiguous.
      flags.contiguous = true;
    } else if (data_size == in_data_size) {
      // Means we sliced a broadcasted dimension so leave the "no holes" flag
      // alone.
    } else {
      // We sliced something. So either we are row or col contiguous or we
      // punched a hole.
      flags.contiguous &= flags.row_contiguous || flags.col_contiguous;
    }

    return std::pair<decltype(flags), size_t>{flags, data_size};
  };

  std::vector<int> indices(1, 0);
  indices.insert(indices.end(), indices_.begin(), indices_.end());
  for (int i = 0; i < indices.size(); i++) {
    size_t offset = indices[i] * in.strides()[axis_];
    auto [new_flags, data_size] = compute_new_flags(
        outputs[i].shape(), in.strides(), in.data_size(), in.flags());
    outputs[i].copy_shared_buffer(
        in, in.strides(), new_flags, data_size, offset);
  }
}

void Square::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, detail::Square());
}

void Sqrt::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (recip_) {
    unary_fp(in, out, detail::Rsqrt());
  } else {
    unary_fp(in, out, detail::Sqrt());
  }
}

void StopGradient::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.copy_shared_buffer(inputs[0]);
}

void Tan::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Tan());
  } else {
    throw std::invalid_argument(
        "[tan] Cannot compute tangent of elements in array"
        " with non floating point type.");
  }
}

void Tanh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Tanh());
  } else {
    throw std::invalid_argument(
        "[tanh] Cannot compute hyperbolic tangent of elements in array"
        " with non floating point type.");
  }
}

void Transpose::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  std::vector<size_t> out_strides(out.ndim());
  auto& in = inputs[0];
  for (int ax = 0; ax < axes_.size(); ++ax) {
    out_strides[ax] = in.strides()[axes_[ax]];
  }

  // Conditions for {row/col}_contiguous
  // - array must be contiguous (no gaps)
  // - underlying buffer size should have the same size as the array
  // - cumulative product of shapes is equal to the strides (we can ignore axes
  //   with size == 1)
  //   - in the forward direction (column contiguous)
  //   - in the reverse direction (row contiguous)
  // - vectors are both row and col contiguous (hence if both row/col are
  //   true, they stay true)
  auto flags = in.flags();
  if (flags.contiguous && in.data_size() == in.size()) {
    size_t f_stride = 1;
    size_t b_stride = 1;
    flags.col_contiguous = true;
    flags.row_contiguous = true;
    for (int i = 0, ri = out.ndim() - 1; i < out.ndim(); ++i, --ri) {
      flags.col_contiguous &= (out_strides[i] == f_stride || out.shape(i) == 1);
      f_stride *= out.shape(i);
      flags.row_contiguous &=
          (out_strides[ri] == b_stride || out.shape(ri) == 1);
      b_stride *= out.shape(ri);
    }
  }
  out.copy_shared_buffer(in, out_strides, flags, in.data_size());
}

} // namespace mlx::core
