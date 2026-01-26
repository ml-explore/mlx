// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/conv/conv.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

// MIOpen integration is optional
// To enable, define MLX_USE_MIOPEN and link against MIOpen library
#ifdef MLX_USE_MIOPEN
#include <miopen/miopen.h>
#endif

namespace mlx::core::rocm {

bool miopen_available() {
#ifdef MLX_USE_MIOPEN
  return true;
#else
  return false;
#endif
}

#ifdef MLX_USE_MIOPEN

namespace {

miopenDataType_t to_miopen_dtype(Dtype dtype) {
  switch (dtype) {
    case float32:
      return miopenFloat;
    case float16:
      return miopenHalf;
    case bfloat16:
      return miopenBFloat16;
    default:
      throw std::runtime_error("Unsupported dtype for MIOpen convolution");
  }
}

} // namespace

void conv_forward(
    CommandEncoder& encoder,
    const array& input,
    const array& weight,
    array& output,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  // MIOpen convolution implementation
  // This requires proper MIOpen handle management and descriptor setup
  throw std::runtime_error(
      "MIOpen convolution forward not yet fully implemented. "
      "Please use CPU fallback.");
}

void conv_backward_input(
    CommandEncoder& encoder,
    const array& grad_output,
    const array& weight,
    array& grad_input,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  throw std::runtime_error(
      "MIOpen convolution backward input not yet fully implemented. "
      "Please use CPU fallback.");
}

void conv_backward_weight(
    CommandEncoder& encoder,
    const array& input,
    const array& grad_output,
    array& grad_weight,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  throw std::runtime_error(
      "MIOpen convolution backward weight not yet fully implemented. "
      "Please use CPU fallback.");
}

#else // MLX_USE_MIOPEN not defined

void conv_forward(
    CommandEncoder& encoder,
    const array& input,
    const array& weight,
    array& output,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  throw std::runtime_error(
      "ROCm convolution requires MIOpen. "
      "Build with MLX_USE_MIOPEN=ON or use CPU fallback.");
}

void conv_backward_input(
    CommandEncoder& encoder,
    const array& grad_output,
    const array& weight,
    array& grad_input,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  throw std::runtime_error(
      "ROCm convolution requires MIOpen. "
      "Build with MLX_USE_MIOPEN=ON or use CPU fallback.");
}

void conv_backward_weight(
    CommandEncoder& encoder,
    const array& input,
    const array& grad_output,
    array& grad_weight,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups) {
  throw std::runtime_error(
      "ROCm convolution requires MIOpen. "
      "Build with MLX_USE_MIOPEN=ON or use CPU fallback.");
}

#endif // MLX_USE_MIOPEN

} // namespace mlx::core::rocm

namespace mlx::core {

// Convolution primitive implementation
// For now, always use fallback since MIOpen integration is not complete
void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "Convolution::eval_gpu requires MIOpen integration for ROCm. "
      "Please use the CPU fallback.");
}

} // namespace mlx::core
