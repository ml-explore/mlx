// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <cassert>

namespace mlx::core {

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(in, out, ctype);
  if (donated && in.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu(const array& in, array& out, CopyType ctype) {
  copy_gpu(in, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out.strides(), 0, 0, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), i_strides, out.strides(), i_offset, 0, ctype, s);
}

array contiguous_copy_gpu(const array& arr, const Stream& s) {
  array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
  copy_gpu(arr, arr_copy, CopyType::General, s);
  return arr_copy;
}

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
