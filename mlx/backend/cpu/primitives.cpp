// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/backend/common/load.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/arange.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/threefry.h"
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

int64_t compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes) {
  auto compute_offset = [&strides, &axes](const auto* indices) {
    int64_t offset = 0;
    for (int i = 0; i < axes.size(); ++i) {
      offset += indices[i] * strides[axes[i]];
    }
    return offset;
  };
  switch (indices.dtype()) {
    case int8:
    case uint8:
      return compute_offset(indices.data<uint8_t>());
    case int16:
    case uint16:
      return compute_offset(indices.data<uint16_t>());
    case int32:
    case uint32:
      return compute_offset(indices.data<uint32_t>());
    case int64:
    case uint64:
      return compute_offset(indices.data<uint64_t>());
    default:
      throw std::runtime_error("Invalid indices type.");
  }
}

void AsStrided::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Broadcast::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void BroadcastAxes::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Copy::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void CustomTransforms::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void Depends::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void ExpandDims::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void NumberOfElements::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Slice::eval_cpu(const std::vector<array>& inputs, array& out) {
  slice(inputs[0], out, start_indices_, strides_);
}
void Split::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  eval(inputs, outputs);
}
void Squeeze::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void StopGradient::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}
void Transpose::eval_cpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Arange::eval_cpu(const std::vector<array>& inputs, array& out) {
  arange(inputs, out, start_, step_);
}

void AsType::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  CopyType ctype = in.flags().contiguous ? CopyType::Vector : CopyType::General;
  copy(in, out, ctype);
}

void Concatenate::eval_cpu(const std::vector<array>& inputs, array& out) {
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

void Flatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void Unflatten::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void Full::eval_cpu(const std::vector<array>& inputs, array& out) {
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

void Load::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  load(out, offset_, reader_, swap_endianness_);
}

void Pad::eval_cpu(const std::vector<array>& inputs, array& out) {
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

void RandomBits::eval_cpu(const std::vector<array>& inputs, array& out) {
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

void Reshape::eval_cpu(const std::vector<array>& inputs, array& out) {
  reshape(inputs[0], out);
}

void DynamicSlice::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }
  auto& in = inputs[0];
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto i_offset = compute_dynamic_offset(inputs[1], in.strides(), axes_);
  copy_inplace(
      /* const array& src = */ in,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ out.shape(),
      /* const Strides& i_strides = */ in.strides(),
      /* const Strides& o_strides = */ out.strides(),
      /* int64_t i_offset = */ i_offset,
      /* int64_t o_offset = */ 0,
      /* CopyType ctype = */ CopyType::GeneralGeneral);
}

void DynamicSliceUpdate::eval_cpu(
    const std::vector<array>& inputs,
    array& out) {
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];

  // Copy or move src to dst
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype);

  auto o_offset = compute_dynamic_offset(inputs[2], out.strides(), axes_);
  copy_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd.strides(),
      /* const std::vector<stride_t>& o_strides = */ out.strides(),
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ o_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral);
}

void SliceUpdate::eval_cpu(const std::vector<array>& inputs, array& out) {
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
  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  // Do copy
  copy_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const std::vector<int>& data_shape = */ upd.shape(),
      /* const std::vector<stride_t>& i_strides = */ upd.strides(),
      /* const std::vector<stride_t>& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral);
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
