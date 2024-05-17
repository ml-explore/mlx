
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
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&,
    const array&) {
  return d.get_kernel(kernel_name);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array&) {
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

} // namespace mlx::core
