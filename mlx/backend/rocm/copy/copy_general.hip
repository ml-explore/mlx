// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// General copy kernel with by-value shape/strides (no hipMemcpyAsync needed)
template <typename In, typename Out, typename IdxT>
__global__ void copy_gg_byval(
    const In* in,
    Out* out,
    IdxT size,
    hip_array<int32_t, MAX_NDIM> shape,
    hip_array<int64_t, MAX_NDIM> strides_in,
    hip_array<int64_t, MAX_NDIM> strides_out,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;

  IdxT loc_in = 0, loc_out = 0;
  IdxT elem = index;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    IdxT dim_idx = elem % shape[i];
    loc_in += dim_idx * IdxT(strides_in[i]);
    loc_out += dim_idx * IdxT(strides_out[i]);
    elem /= shape[i];
  }
  out[loc_out] = cast_to<Out>(in[loc_in]);
}

} // namespace rocm

void copy_general(
    rocm::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t offset_in,
    int64_t offset_out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out) {

  int ndim = shape.size();
  size_t data_size = 1;
  for (auto& s : shape) {
    data_size *= s;
  }

  if (data_size == 0) {
    return;
  }

  // Pack shape/strides into by-value structs (no device allocation needed)
  rocm::hip_array<int32_t, MAX_NDIM> shape_arg = {};
  rocm::hip_array<int64_t, MAX_NDIM> strides_in_arg = {};
  rocm::hip_array<int64_t, MAX_NDIM> strides_out_arg = {};
  for (int i = 0; i < ndim; i++) {
    shape_arg.data_[i] = static_cast<int32_t>(shape[i]);
    strides_in_arg.data_[i] = strides_in[i];
    strides_out_arg.data_[i] = strides_out[i];
  }

  const void* in_ptr = gpu_ptr<void>(in);
  void* out_ptr = gpu_ptr<void>(out);

  int block_size = 256;
  int num_blocks = (data_size + block_size - 1) / block_size;
  int64_t size_arg = static_cast<int64_t>(data_size);

  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      using InType = hip_type_t<MLX_GET_TYPE(in_type_tag)>;
      using OutType = hip_type_t<MLX_GET_TYPE(out_type_tag)>;

      const InType* in_typed = static_cast<const InType*>(in_ptr) + offset_in;
      OutType* out_typed = static_cast<OutType*>(out_ptr) + offset_out;

      encoder.add_kernel_node(
          &rocm::copy_gg_byval<InType, OutType, int64_t>,
          dim3(num_blocks), dim3(block_size), 0,
          in_typed,
          out_typed,
          size_arg,
          shape_arg,
          strides_in_arg,
          strides_out_arg,
          ndim);
    });
  });
}

} // namespace mlx::core
