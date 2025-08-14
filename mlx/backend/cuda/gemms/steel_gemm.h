#pragma once

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <numeric>

namespace mlx::core {

void dispatch_steel_gemm(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& d,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldd,
    bool a_transposed,
    bool b_transposed);

} // namespace mlx::core