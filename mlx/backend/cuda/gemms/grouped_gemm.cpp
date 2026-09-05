// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/cudnn_utils.h"

#include <stdexcept>

namespace mlx::core {

void grouped_mm(
    [[maybe_unused]] bool a_transposed,
    [[maybe_unused]] int lda,
    [[maybe_unused]] bool b_transposed,
    [[maybe_unused]] int ldb,
    const array& a,
    const array& b,
    const array& offsets,
    array& out,
    cu::CommandEncoder& encoder) {
#if CUDNN_VERSION >= 91800
  cudnn_grouped_mm(a, b, offsets, out, encoder);
#else
  throw std::runtime_error(
      "[grouped_mm] Grouped matmul requires cuDNN 9.18 or newer.");
#endif
}

} // namespace mlx::core
