// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/gpu/copy.h"

namespace mlx::core {

// Helper to ensure an array is fully row contiguous.
// Makes a contiguous copy if needed.
inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

// Helper to ensure an array is contiguous (row or column).
inline array
ensure_contiguous(const array& x, cu::CommandEncoder& enc, const Stream& s) {
  if (x.flags().row_contiguous || x.flags().col_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

// Helper to ensure the last two dimensions of a matrix are row-contiguous.
// This is sufficient for 2D weight matrices where we don't need to flatten
// leading dimensions.
inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

void affine_quantize(
    const array& w,
    array& wq,
    array& scales,
    array& biases,
    int group_size_,
    int bits_,
    cu::CommandEncoder& enc,
    const Stream& s);

void affine_dequantize(
    const array& wq,
    const array& scales,
    const array& biases,
    array& w,
    int group_size_,
    int bits_,
    cu::CommandEncoder& enc,
    const Stream& s);

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
