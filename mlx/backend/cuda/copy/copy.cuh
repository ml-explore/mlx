// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

namespace mlx::core {

#define MLX_SWITCH_COPY_TYPES(in, out, InType, OutType, ...) \
  MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, {               \
    MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {           \
      using InType = cuda_type_t<CTYPE_IN>;                  \
      using OutType = cuda_type_t<CTYPE_OUT>;                \
      __VA_ARGS__;                                           \
    });                                                      \
  })

void copy_contiguous(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out);

void copy_general(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out);

void copy_general_dynamic(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    const array& dynamic_offset_in,
    const array& dynamic_offset_out);

void copy_general_input(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in);

} // namespace mlx::core
