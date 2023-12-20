// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/arange.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/erf.h"
#include "mlx/backend/common/threefry.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void Abs::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (is_unsigned(in.dtype())) {
    // No-op for unsigned types
    out.copy_shared_buffer(in);
  } else {
    unary(in, out, AbsOp());
  }
}

void Arange::eval(const std::vector<array>& inputs, array& out) {
  arange(inputs, out, start_, step_);
}

void ArcCos::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::acos(x); });
  } else {
    throw std::invalid_argument(
        "[arccos] Cannot compute inverse cosine of elements in array"
        " with non floating point type.");
  }
}

void ArcCosh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::acosh(x); });
  } else {
    throw std::invalid_argument(
        "[arccosh] Cannot compute inverse hyperbolic cosine of elements in"
        " array with non floating point type.");
  }
}

void ArcSin::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::asin(x); });
  } else {
    throw std::invalid_argument(
        "[arcsin] Cannot compute inverse sine of elements in array"
        " with non floating point type.");
  }
}

void ArcSinh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::asinh(x); });
  } else {
    throw std::invalid_argument(
        "[arcsinh] Cannot compute inverse hyperbolic sine of elements in"
        " array with non floating point type.");
  }
}

void ArcTan::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::atan(x); });
  } else {
    throw std::invalid_argument(
        "[arctan] Cannot compute inverse tangent of elements in array"
        " with non floating point type.");
  }
}

void ArcTanh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::atanh(x); });
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
  if (not is_integral(in.dtype())) {
    unary_fp(in, out, [](auto x) { return std::ceil(x); });
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
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::cos(x); });
  } else {
    throw std::invalid_argument(
        "[cos] Cannot compute cosine of elements in array"
        " with non floating point type.");
  }
}

void Cosh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::cosh(x); });
  } else {
    throw std::invalid_argument(
        "[cosh] Cannot compute hyperbolic cosine of elements in array"
        " with non floating point type.");
  }
}

void Erf::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  switch (out.dtype()) {
    case float32:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<float>(in, out, [](auto x) { return std::erf(x); });
      break;
    case float16:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<float16_t>(in, out, [](auto x) {
        return static_cast<float16_t>(std::erf(static_cast<float>(x)));
      });
      break;
    case bfloat16:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<bfloat16_t>(in, out, [](auto x) {
        return static_cast<bfloat16_t>(std::erf(static_cast<float>(x)));
      });
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
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<float>(in, out, [](auto x) { return erfinv(x); });
      break;
    case float16:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<float16_t>(in, out, [](auto x) {
        return static_cast<float16_t>(erfinv(static_cast<float>(x)));
      });
      break;
    case bfloat16:
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
      unary_op<bfloat16_t>(in, out, [](auto x) {
        return static_cast<bfloat16_t>(erfinv(static_cast<float>(x)));
      });
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

  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::exp(x); });
  } else {
    throw std::invalid_argument(
        "[exp] Cannot exponentiate elements in array"
        " with non floating point type.");
  }
}

void Floor::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (not is_integral(in.dtype())) {
    unary_fp(in, out, [](auto x) { return std::floor(x); });
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
  if (is_floating_point(out.dtype())) {
    switch (base_) {
      case Base::e:
        unary_fp(in, out, [](auto x) { return std::log(x); });
        break;
      case Base::two:
        unary_fp(in, out, [](auto x) { return std::log2(x); });
        break;
      case Base::ten:
        unary_fp(in, out, [](auto x) { return std::log10(x); });
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
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::log1p(x); });
  } else {
    throw std::invalid_argument(
        "[log1p] Cannot compute log of elements in array with"
        " non floating point type.");
  }
}

void LogicalNot::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, [](auto x) { return !x; });
}

void Negative::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, [](auto x) { return -x; });
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

void Reshape::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (in.flags().row_contiguous) {
    // For row contiguous reshapes:
    // - Shallow copy the buffer
    // - If reshaping into a vector (all singleton dimensions except one) it
    //    becomes col contiguous again.
    auto flags = in.flags();
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.copy_shared_buffer(in, out.strides(), flags, in.data_size());
  } else {
    copy(in, out, in.data_size() == 1 ? CopyType::Scalar : CopyType::General);
  }
}

void Round::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (not is_integral(in.dtype())) {
    unary_fp(in, out, RoundOp());
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}

void Sigmoid::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    auto sigmoid_op = [](auto x) {
      auto one = static_cast<decltype(x)>(1.0);
      return one / (one + std::exp(-x));
    };
    unary_fp(in, out, sigmoid_op);
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
    unary(in, out, SignOp());
  }
}

void Sin::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::sin(x); });
  } else {
    throw std::invalid_argument(
        "[sin] Cannot compute sine of elements in array"
        " with non floating point type.");
  }
}

void Sinh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::sinh(x); });
  } else {
    throw std::invalid_argument(
        "[sinh] Cannot compute hyperbolic sine of elements in array"
        " with non floating point type.");
  }
}

void Slice::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }
  auto& in = inputs[0];
  auto strides = in.strides();
  auto flags = in.flags();
  size_t data_offset = 0;
  for (int i = 0; i < in.ndim(); ++i) {
    data_offset += start_indices_[i] * in.strides()[i];
    strides[i] *= strides_[i];
  }

  // Compute row/col contiguity
  size_t data_size = 1;
  size_t f_stride = 1;
  size_t b_stride = 1;
  flags.row_contiguous = true;
  flags.col_contiguous = true;
  for (int i = 0, ri = out.ndim() - 1; ri >= 0; i++, ri--) {
    flags.col_contiguous &= strides[i] == f_stride || out.shape(i) == 1;
    flags.row_contiguous &= strides[ri] == b_stride || out.shape(ri) == 1;
    f_stride *= out.shape(i);
    b_stride *= out.shape(ri);
    if (strides[i] > 0) {
      data_size *= out.shape(i);
    }
  }

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

  out.copy_shared_buffer(in, strides, flags, data_size, data_offset);
}

void Square::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  unary(in, out, [](auto x) { return x * x; });
}

void Sqrt::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (recip_) {
    unary_fp(in, out, [](auto x) {
      return static_cast<decltype(x)>(1.0) / sqrt(x);
    });
  } else {
    unary_fp(in, out, [](auto x) { return sqrt(x); });
  }
}

void StopGradient::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.copy_shared_buffer(inputs[0]);
}

void Tan::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::tan(x); });
  } else {
    throw std::invalid_argument(
        "[tan] Cannot compute tangent of elements in array"
        " with non floating point type.");
  }
}

void Tanh::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (is_floating_point(out.dtype())) {
    unary_fp(in, out, [](auto x) { return std::tanh(x); });
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
