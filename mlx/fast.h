// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <any>
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

class MetalKernel {
 public:
  MetalKernel(
      const std::string& name,
      const std::string& source,
      std::map<std::string, std::vector<int>> output_shapes,
      std::map<std::string, Dtype> output_dtypes,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      bool ensure_row_contiguous,
      bool verbose)
      : name_(name),
        source_(source),
        output_shapes_(output_shapes),
        output_dtypes_(output_dtypes),
        grid_(grid),
        threadgroup_(threadgroup),
        ensure_row_contiguous_(ensure_row_contiguous),
        verbose_(verbose) {
    validate_output_shapes();
  }

  std::map<std::string, array> run(
      std::map<std::string, array>& inputs,
      StreamOrDevice s = {});

  std::map<std::string, TemplateArg> template_args;

 private:
  void validate_output_shapes();
  void write_signature(
      std::string& func_name,
      std::map<std::string, array>& inputs,
      std::ostringstream& kernel_source);
  std::string write_template(
      std::string& func_name,
      std::ostringstream& kernel_source);

  std::string name_;
  std::string source_;
  std::map<std::string, std::vector<int>> output_shapes_;
  std::map<std::string, Dtype> output_dtypes_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> threadgroup_;
  bool ensure_row_contiguous_ = true;
  bool verbose_ = false;
};

} // namespace mlx::core::fast
