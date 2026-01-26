// Copyright © 2025 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no ROCm implementation.");    \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no ROCm implementation.");   \
  }

// Convolution requires MIOpen integration (AMD's equivalent of cuDNN)
NO_GPU(Convolution)

NO_GPU(BlockMaskedMM)
NO_GPU(FFT)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU(QQMatmul)
NO_GPU(QuantizedMatmul)
NO_GPU(SegmentedMM)
NO_GPU_MULTI(SVD)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)
NO_GPU(MaskedScatter)

// Note: The following are now implemented in their respective files:
// - Load: load.cpp
// - CustomKernel: custom_kernel.cpp
// - ScaledDotProductAttention: scaled_dot_product_attention.cpp
// - ScaledDotProductAttentionVJP: scaled_dot_product_attention.cpp
// - Quantize: quantized/quantized.cpp
// - AffineQuantize: quantized/quantized.cpp
// - ConvertFP8: quantized/quantized.cpp
// - AllGather, AllReduce, ReduceScatter, Send, Recv: distributed.hip

} // namespace mlx::core
