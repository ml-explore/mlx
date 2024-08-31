// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <optional>

#include "mlx/utils.h"

namespace mlx::core::fast {

array rms_norm(
    const array& x,
    const array& weight,
    float eps,
    StreamOrDevice s = {});

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask = std::nullopt,
    const std::optional<int>& memory_efficient_threshold = std::nullopt,
    StreamOrDevice s = {});

std::tuple<array, array, array> affine_quantize(
    const array& w,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

array affine_quantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

array affine_dequantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

typedef std::variant<int, bool, Dtype> TemplateArg;
typedef int MetalGrid1D;
typedef std::tuple<int, int> MetalGrid2D;
typedef std::tuple<int, int, int> MetalGrid3D;

class MetalKernel {
 public:
  MetalKernel(
      const std::string& name,
      const std::string& source,
      const std::string& header = "",
      bool ensure_row_contiguous = true,
      bool atomic_outputs = false)
      : name_(name),
        source_(source),
        header_(header),
        ensure_row_contiguous_(ensure_row_contiguous),
        atomic_outputs_(atomic_outputs) {}

  template <
      typename T,
      std::enable_if_t<
          std::is_same_v<T, MetalGrid1D> || std::is_same_v<T, MetalGrid2D> ||
              std::is_same_v<T, MetalGrid3D>,
          bool> = true>
  std::map<std::string, array> operator()(
      std::map<std::string, array>& inputs,
      std::map<std::string, std::vector<int>> output_shapes,
      std::map<std::string, Dtype> output_dtypes,
      T grid,
      T threadgroup,
      std::optional<std::map<std::string, TemplateArg>> template_args =
          std::nullopt,
      std::optional<float> init_value = std::nullopt,
      bool verbose = false,
      StreamOrDevice s = {});

 private:
  std::string name_;
  std::string source_;
  std::string header_;
  bool ensure_row_contiguous_;
  bool atomic_outputs_;
};
} // namespace mlx::core::fast
