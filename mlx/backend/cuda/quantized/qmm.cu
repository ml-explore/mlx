// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace cu {} // namespace cu

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    bool transpose_,
    int group_size_,
    int bits_,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc,
    const Stream& s) {
  dispatch_float_types(x.dtype(), "qmm", [&](auto type_tag) {
    dispatch_groups(group_size_, [&](auto group_size) {
      dispatch_bits(bits_, [&](auto bits) {
        dispatch_bool(transpose_, [&](auto transpose) {
          using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
        });
      });
    });
  });
}

} // namespace mlx::core
