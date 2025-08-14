// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/reshape.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (copy_necessary) {
    out.set_data(allocator::malloc(out.nbytes()));
    copy_gpu_inplace(
        in,
        out,
        in.shape(),
        in.strides(),
        make_contiguous_strides(in.shape()),
        0,
        0,
        CopyType::General,
        s);
  } else {
    shared_buffer_reshape(in, out_strides, out);
  }
}

array flatten_in_eval(const array& x, int start_axis, int end_axis, Stream s) {
  int ndim = x.ndim();
  if (start_axis < 0) {
    start_axis += ndim;
  }
  if (end_axis < 0) {
    end_axis += ndim;
  }
  start_axis = std::max(0, start_axis);
  end_axis = std::min(ndim - 1, end_axis);

  return reshape_in_eval(x, Flatten::output_shape(x, start_axis, end_axis), s);
}

array reshape_in_eval(const array& x, Shape shape, Stream s) {
  array out(std::move(shape), x.dtype(), nullptr, {});
  reshape_gpu(x, out, s);
  return out;
}

array swapaxes_in_eval(const array& x, int axis1, int axis2) {
  int ndim = x.ndim();
  if (axis1 < 0) {
    axis1 += ndim;
  }
  if (axis2 < 0) {
    axis2 += ndim;
  }

  auto shape = x.shape();
  std::swap(shape[axis1], shape[axis2]);
  auto strides = x.strides();
  std::swap(strides[axis1], strides[axis2]);

  auto [data_size, row_contiguous, col_contiguous] =
      check_contiguity(shape, strides);
  bool contiguous = data_size == x.data_size();

  array out(std::move(shape), x.dtype(), nullptr, {});
  out.copy_shared_buffer(
      x,
      std::move(strides),
      {contiguous, row_contiguous, col_contiguous},
      x.data_size());
  return out;
}

} // namespace mlx::core
