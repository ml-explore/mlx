// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qmm/qmm.h"

#include <cute/tensor.hpp>

namespace mlx::core {

#if defined(MLX_CUDA_SM90A_ENABLED)
// Defined in qmm_impl_sm90_xxx.cu files.
template <typename TileShape, typename ClusterShape>
void qmm_impl_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);
#endif // defined(MLX_CUDA_SM90A_ENABLED)

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
#if defined(MLX_CUDA_SM90A_ENABLED)
  auto dispatch = [&]<int tile_m, int tile_n, int cluster_m>() {
    using cute::Int;
    using TileShapeMN = cute::Shape<Int<tile_m>, Int<tile_n>>;
    using ClusterShape = cute::Shape<Int<cluster_m>, Int<1>, Int<1>>;
    qmm_impl_sm90<TileShapeMN, ClusterShape>(
        x, w, scales, biases, out, bits, group_size, encoder, s);
  };
  int m = out.shape(-2);
  if (m <= 16) {
    dispatch.template operator()<128, 16, 1>();
  } else if (m <= 32) {
    dispatch.template operator()<128, 32, 1>();
  } else if (m <= 64) {
    dispatch.template operator()<128, 64, 2>();
  } else if (m <= 128) {
    dispatch.template operator()<128, 128, 2>();
  } else {
    dispatch.template operator()<128, 256, 2>();
  }
#else
  throw std::runtime_error(
      "[quantized_matmul] Hopper-only kernel is not available.");
#endif // defined(MLX_CUDA_SM90A_ENABLED)
}

} // namespace mlx::core
