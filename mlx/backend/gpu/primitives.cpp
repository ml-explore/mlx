// Copyright © 2025 Apple Inc.

#include "mlx/primitives.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"

#if defined(MLX_USE_CUDA)
#include <nvtx3/nvtx3.hpp>
#endif

#include <cassert>

#if defined(MLX_USE_CUDA)
#define MLX_PROFILER_RANGE(message) nvtx3::scoped_range r(message)
#else
#define MLX_PROFILER_RANGE(message)
#endif

namespace mlx::core {

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("AsStrided::eval_gpu");
  eval(inputs, out);
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("AsType::eval_gpu");
  CopyType ctype =
      inputs[0].flags().contiguous ? CopyType::Vector : CopyType::General;
  copy_gpu(inputs[0], out, ctype);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Broadcast::eval_gpu");
  eval(inputs, out);
}

void BroadcastAxes::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("BroadcastAxes::eval_gpu");
  eval(inputs, out);
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Concatenate::eval_gpu");
  concatenate_gpu(inputs, out, axis_, stream());
}

void Contiguous::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Contiguous::eval_gpu");
  assert(inputs.size() == 1);
  auto& in = inputs[0];
  constexpr size_t extra_bytes = 16384;
  if (in.buffer_size() <= out.nbytes() + extra_bytes &&
      (in.flags().row_contiguous ||
       (allow_col_major_ && in.flags().col_contiguous))) {
    out.copy_shared_buffer(in);
  } else {
    copy_gpu(in, out, CopyType::General);
  }
}

void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Copy::eval_gpu");
  eval(inputs, out);
}

void CustomTransforms::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  MLX_PROFILER_RANGE("CustomTransforms::eval_gpu");
  eval(inputs, outputs);
}

void Depends::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  MLX_PROFILER_RANGE("Depends::eval_gpu");
  eval(inputs, outputs);
}

void DynamicSlice::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("DynamicSlice::eval_gpu");
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& start = inputs[1];
  out.set_data(allocator::malloc(out.nbytes()));

  auto s = stream();
  auto in_offset = compute_dynamic_offset(start, in.strides(), axes_, s);
  copy_gpu_inplace(
      /* const array& src = */ in,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ out.shape(),
      /* const Strides& i_strides = */ in.strides(),
      /* const Strides& o_strides = */ out.strides(),
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ 0,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      /* const Stream& s = */ s,
      /* std::optional<array> dynamic_i_offset = */ std::move(in_offset),
      /* std::optional<array> dynamic_o_offset = */ std::nullopt);
}

void DynamicSliceUpdate::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  MLX_PROFILER_RANGE("DynamicSliceUpdate::eval_gpu");
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  auto& upd = inputs[1];
  auto& start_indices = inputs[2];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  // Copy or donate input to output
  auto s = stream();
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, s);

  auto out_offset =
      compute_dynamic_offset(start_indices, out.strides(), axes_, s);
  copy_gpu_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ upd.shape(),
      /* const Strides& i_strides = */ upd.strides(),
      /* const Strides& o_strides = */ out.strides(),
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ 0,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      /* const Stream& s = */ s,
      /* std::optional<array> dynamic_i_offset = */ std::nullopt,
      /* std::optional<array> dynamic_o_offset = */ std::move(out_offset));
}

void ExpandDims::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("ExpandDims::eval_gpu");
  eval(inputs, out);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Full::eval_gpu");
  auto in = inputs[0];
  CopyType ctype;
  if (in.data_size() == 1) {
    ctype = CopyType::Scalar;
  } else if (in.flags().contiguous) {
    ctype = CopyType::Vector;
  } else {
    ctype = CopyType::General;
  }
  copy_gpu(in, out, ctype);
}

void Flatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Flatten::eval_gpu");
  reshape_gpu(inputs[0], out, stream());
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("NumberOfElements::eval_gpu");
  eval(inputs, out);
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Pad::eval_gpu");
  // Inputs must be base input array and scalar val array
  assert(inputs.size() == 2);
  auto& in = inputs[0];
  auto& val = inputs[1];

  // Padding value must be a scalar
  assert(val.size() == 1);

  // Padding value, input and output must be of the same type
  assert(val.dtype() == in.dtype() && in.dtype() == out.dtype());

  pad_gpu(in, val, out, axes_, low_pad_size_, stream());
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Reshape::eval_gpu");
  reshape_gpu(inputs[0], out, stream());
}

void Split::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  MLX_PROFILER_RANGE("Split::eval_gpu");
  eval(inputs, outputs);
}

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Slice::eval_gpu");
  assert(inputs.size() == 1);
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  auto& in = inputs[0];
  slice_gpu(in, out, start_indices_, strides_, stream());
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
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

  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());
  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  // Do copy
  copy_gpu_inplace(
      /* const array& src = */ upd,
      /* array& dst = */ out,
      /* const Shape& data_shape = */ upd.shape(),
      /* const Strides& i_strides = */ upd.strides(),
      /* const Strides& o_strides = */ out_strides,
      /* int64_t i_offset = */ 0,
      /* int64_t o_offset = */ data_offset,
      /* CopyType ctype = */ CopyType::GeneralGeneral,
      /* const Stream& s = */ stream());
}

void Squeeze::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Squeeze::eval_gpu");
  eval(inputs, out);
}

void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("StopGradient::eval_gpu");
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Transpose::eval_gpu");
  eval(inputs, out);
}

void Unflatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("Unflatten::eval_gpu");
  reshape_gpu(inputs[0], out, stream());
}

void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  MLX_PROFILER_RANGE("View::eval_gpu");
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
    auto tmp = array(in.shape(), in.dtype(), nullptr, {});
    tmp.set_data(allocator::malloc(tmp.nbytes()));
    copy_gpu_inplace(in, tmp, CopyType::General, stream());

    auto flags = out.flags();
    flags.contiguous = true;
    flags.row_contiguous = true;
    auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
    flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
    out.copy_shared_buffer(tmp, out.strides(), flags, out.size());
  }
}

} // namespace mlx::core
