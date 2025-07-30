// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/steel/gemm.cuh"
#include "mlx/dtype_utils.h"

namespace mlx::core::cu {

void simple_gemm(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc) {
  enc.set_input_array(a);
  enc.set_input_array(b);
  enc.set_output_array(out);
  dispatch_float_types(a.dtype(), "simple_gemm", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;

    auto kernel = ab_t_aligned<DataType, BM, BN, BK>;
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    dim3 grid(N / BN, M / BM);
    enc.add_kernel_node(
        kernel,
        grid,
        4 * WARP_SIZE,
        2 * sizeof(DataType) * (BM * BK + BN * BK),
        a.data<DataType>(),
        b.data<DataType>(),
        out.data<DataType>(),
        N,
        K);
  });
}

} // namespace mlx::core::cu
