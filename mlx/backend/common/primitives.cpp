// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/arange.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/ops.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/threefry.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

void reshape(const array& in, array& out) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    copy_inplace(in, out, CopyType::General);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

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

void Conjugate::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (out.dtype() == complex64) {
    unary_fp(in, out, detail::Conjugate());
  } else {
    throw std::invalid_argument(
        "[conjugate] conjugate must be called on complex input.");
  }
}

void Contiguous::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  if (in.flags().row_contiguous ||
      (allow_col_major_ && in.flags().col_contiguous)) {
    out.copy_shared_buffer(in);
  } else {
    copy(in, out, CopyType::General);
  }
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

void Expm1::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  if (issubdtype(out.dtype(), inexact)) {
    unary_fp(in, out, detail::Expm1());
  } else {
    throw std::invalid_argument(
        "[expm1] Cannot exponentiate elements in array"
        " with non floating point type.");
  }
}

void Flatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void Unflatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
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

void Imag::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_op<complex64_t, float>(inputs[0], out, detail::Imag());
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

void Real::eval_cpu(const std::vector<array>& inputs, array& out) {
  unary_op<complex64_t, float>(inputs[0], out, detail::Real());
}

void Reshape::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
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

void Slice::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];

  // Calculate out strides, initial offset and if copy needs to be made
  auto [data_offset, inp_strides] = prepare_slice(in, start_indices_, strides_);
  size_t data_end = 1;
  for (int i = 0; i < end_indices_.size(); ++i) {
    if (in.shape()[i] > 1) {
      auto end_idx = start_indices_[i] + out.shape()[i] * strides_[i] - 1;
      data_end += end_idx * in.strides()[i];
    }
  }
  size_t data_size = data_end - data_offset;
  Strides ostrides{inp_strides.begin(), inp_strides.end()};
  shared_buffer_slice(in, ostrides, data_offset, data_size, out);
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
  auto [data_offset, out_strides] = prepare_slice(in, start_indices_, strides_);

  // Do copy
  Strides upd_strides{upd.strides().begin(), upd.strides().end()};
  copy_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd_strides,
      /* const std::vector<stride_t>& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral);
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

void View::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  auto ibytes = size_of(in.dtype());
  auto obytes = size_of(out.dtype());
  // Conditions for buffer copying (disjunction):
  // - type size is the same
  // - type size is smaller and the last axis is contiguous
  // - the entire array is row contiguous
  if (ibytes == obytes || (obytes < ibytes && in.strides().back() == 1) ||
      in.flags().row_contiguous) {
    auto strides = in.strides();
    for (int i = 0; i < static_cast<int>(strides.size()) - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    out.copy_shared_buffer(
        in, strides, in.flags(), in.data_size() * ibytes / obytes);
  } else {
    auto tmp = array(
        in.shape(), in.dtype() == bool_ ? uint8 : in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc_or_wait(tmp.nbytes()));
    if (in.dtype() == bool_) {
      auto in_tmp = array(in.shape(), uint8, nullptr, {});
      in_tmp.copy_shared_buffer(in);
      copy_inplace(in_tmp, tmp, CopyType::General);
    } else {
      copy_inplace(in, tmp, CopyType::General);
    }

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.move_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

} // namespace mlx::core
