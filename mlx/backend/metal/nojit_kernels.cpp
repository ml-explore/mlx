// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

MTL::ComputePipelineState* get_arange_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const std::string) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const std::string) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    Dtype,
    const std::string) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype,
    const std::string) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool,
    bool,
    const std::string&,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_mb_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&,
    const Dtype&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&,
    const std::string&,
    const Dtype&,
    const Dtype&,
    const std::string&,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_fused_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name, "mlx", hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_gemm_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool,
    bool,
    int,
    int,
    int,
    int,
    int,
    bool,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_gemv_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const std::optional<array>&,
    const std::optional<array>&,
    bool,
    int,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_conv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    int,
    int,
    int,
    int,
    int,
    int,
    bool) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    int,
    int,
    int,
    int,
    int) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string&) {
  return d.get_kernel(kernel_name, "mlx", hash_name, func_consts);
}

MTL::ComputePipelineState* get_quantized_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string&) {
  return d.get_kernel(kernel_name);
}

} // namespace mlx::core
