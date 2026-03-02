// Copyright © 2026 Apple Inc.

// MoE Expert Parallelism Metal kernel launch helpers.
//
// Provides get_moe_kernel() which JIT-compiles and caches the six MoE
// Metal kernels declared in kernels/moe.h:
//   - moe_dispatch_local
//   - moe_dispatch_scatter_remote
//   - moe_combine_gather_remote
//   - moe_combine_weighted_sum
//   - moe_packet_gather
//   - moe_packet_scatter
//
// The actual eval_gpu dispatch logic lives in distributed.cpp.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::distributed {

namespace {

std::string moe_type_string(Dtype dtype) {
  switch (dtype) {
    case float32:
      return "float";
    case float16:
      return "half";
    case bfloat16:
      return "bfloat16_t";
    default:
      throw std::runtime_error(
          "[moe] Unsupported dtype for Metal kernel. "
          "Expected float32, float16, or bfloat16.");
  }
}

std::string moe_type_suffix(Dtype dtype) {
  switch (dtype) {
    case float32:
      return "float32";
    case float16:
      return "float16";
    case bfloat16:
      return "bfloat16";
    default:
      throw std::runtime_error("[moe] Unsupported dtype.");
  }
}

} // namespace

MTL::ComputePipelineState*
get_moe_kernel(metal::Device& d, const std::string& base_name, Dtype dtype) {
  auto type_str = moe_type_string(dtype);
  auto suffix = moe_type_suffix(dtype);
  auto kernel_name = base_name + "_" + suffix;

  auto lib = d.get_library(kernel_name, [&]() {
    std::string source = metal::utils();
    source += metal::moe();
    source += "\ntemplate [[host_name(\"" + kernel_name + "\")]] ";
    source += "[[kernel]] void " + base_name + "<" + type_str + ">(";

    // Explicit template instantiation with named parameters so that
    // Metal [[buffer(N)]] / [[thread_position_in_grid]] attributes
    // bind to parameters (not types).
    if (base_name == "moe_dispatch_local") {
      source += "const device " + type_str + "* tokens [[buffer(0)]], ";
      source += "device " + type_str + "* dispatched [[buffer(1)]], ";
      source += "const device int* slot_map [[buffer(2)]], ";
      source += "const device int* nk_indices [[buffer(3)]], ";
      source += "constant int& D [[buffer(4)]], ";
      source += "constant int& top_k [[buffer(5)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else if (base_name == "moe_dispatch_scatter_remote") {
      source += "const device " + type_str + "* recv_payload [[buffer(0)]], ";
      source += "device " + type_str + "* dispatched [[buffer(1)]], ";
      source += "const device int* recv_flat_idx [[buffer(2)]], ";
      source += "constant int& D [[buffer(3)]], ";
      source += "constant int& cnt [[buffer(4)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else if (base_name == "moe_combine_gather_remote") {
      source += "const device " + type_str + "* expert_out [[buffer(0)]], ";
      source += "device " + type_str + "* send_results [[buffer(1)]], ";
      source += "const device int* eo_flat_idx [[buffer(2)]], ";
      source += "constant int& D [[buffer(3)]], ";
      source += "constant int& cnt [[buffer(4)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else if (base_name == "moe_combine_weighted_sum") {
      source += "const device " + type_str + "* data_src [[buffer(0)]], ";
      source += "device " + type_str + "* output [[buffer(1)]], ";
      source += "const device " + type_str + "* original [[buffer(2)]], ";
      source += "const device float* weights [[buffer(3)]], ";
      source += "const device int* src_idx [[buffer(4)]], ";
      source += "constant int& D [[buffer(5)]], ";
      source += "constant int& N [[buffer(6)]], ";
      source += "constant int& top_k [[buffer(7)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else if (base_name == "moe_packet_gather") {
      source += "const device " + type_str + "* source [[buffer(0)]], ";
      source += "device uint8_t* packet [[buffer(1)]], ";
      source += "const device int* src_idx [[buffer(2)]], ";
      source += "const device uint32_t* headers [[buffer(3)]], ";
      source += "constant int& D [[buffer(4)]], ";
      source += "constant int& cnt [[buffer(5)]], ";
      source += "constant int& row_stride [[buffer(6)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else if (base_name == "moe_packet_scatter") {
      source += "const device uint8_t* packet [[buffer(0)]], ";
      source += "device " + type_str + "* target [[buffer(1)]], ";
      source += "const device int* flat_idx [[buffer(2)]], ";
      source += "constant int& D [[buffer(3)]], ";
      source += "constant int& cnt [[buffer(4)]], ";
      source += "constant int& row_stride [[buffer(5)]], ";
      source += "uint2 gid [[thread_position_in_grid]]);\n";
    } else {
      throw std::runtime_error(
          "[get_moe_kernel] Unknown kernel base name: " + base_name);
    }

    return source;
  });

  return d.get_kernel(kernel_name, lib);
}

} // namespace mlx::core::distributed
