// Copyright © 2025 Apple Inc.

#include "mlx/distributed/primitives.h"
#include <cuda_runtime.h>
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no CUDA implementation.");    \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no CUDA implementation.");   \
  }

#if CUDART_VERSION < 12080
void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[QQMatmul::eval_gpu] QQMM is only supported with CUDA 12.8 or higher.");
}
#endif

NO_GPU(BlockMaskedMM)
NO_GPU(FFT)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU(SegmentedMM)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)
NO_GPU(MaskedScatter)

namespace distributed {
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
} // namespace distributed

} // namespace mlx::core
