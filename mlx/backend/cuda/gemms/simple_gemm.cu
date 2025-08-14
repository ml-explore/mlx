// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/steel/gemm.cuh"
#include "mlx/dtype_utils.h"

#include <iostream>

namespace mlx::core::cu {

namespace {

template <typename Kernel>
static void configure_smem(Kernel kernel, int SM) {
  static bool done = false;
  if (done) {
    return;
  }
  std::cout << "configuring" << std::endl;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
  cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);
  done = true;
}

} // namespace

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
    constexpr int BK = 32;
    constexpr int PIPE = 3;
    constexpr int SM = PIPE * sizeof(DataType) * (BM * BK + BN * BK);
    constexpr int WM = 2;
    constexpr int WN = 4;

    auto kernel = ab_t_aligned<DataType, BM, BN, BK, WM, WN, PIPE>;
    configure_smem(kernel, SM);

    dim3 grid(N / BN, M / BM);
    enc.add_kernel_node(
        kernel,
        grid,
        WM * WN * WARP_SIZE,
        SM,
        a.data<DataType>(),
        b.data<DataType>(),
        out.data<DataType>(),
        N,
        K);
  });
}

} // namespace mlx::core::cu
