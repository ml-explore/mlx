// Copyright © 2024 Apple Inc.
#include <cassert>

#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

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

void Copy::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.copy_shared_buffer(inputs[0]);
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

void StopGradient::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  out.copy_shared_buffer(inputs[0]);
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
