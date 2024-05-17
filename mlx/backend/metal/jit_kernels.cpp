// Copyright © 2024 Apple Inc.

#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/jit/arange.h"
#include "mlx/backend/metal/jit/binary.h"
#include "mlx/backend/metal/jit/binary_two.h"
#include "mlx/backend/metal/jit/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/jit/reduce.h"
#include "mlx/backend/metal/jit/scan.h"
#include "mlx/backend/metal/jit/softmax.h"
#include "mlx/backend/metal/jit/sort.h"
#include "mlx/backend/metal/jit/ternary.h"
#include "mlx/backend/metal/jit/unary.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::string op_name(const array& arr) {
  std::ostringstream op_t;
  arr.primitive().print(op_t);
  return op_t.str();
}

MTL::ComputePipelineState* get_arange_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source
        << metal::utils() << metal::arange()
        << fmt::format(arange_kernels, lib_name, get_type_string(out.dtype()));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  std::string lib_name = kernel_name.substr(1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::unary_ops() << metal::unary()
                  << fmt::format(
                         unary_kernels,
                         lib_name,
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(2);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::binary_ops() << metal::binary()
                  << fmt::format(
                         binary_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(2);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::binary_ops()
                  << metal::binary_two()
                  << fmt::format(
                         binary_two_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::ternary_ops() << metal::ternary()
                  << fmt::format(
                         ternary_kernels,
                         lib_name,
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::copy()
                  << fmt::format(
                         copy_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool precise,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::softmax()
                  << fmt::format(
                         softmax_kernels,
                         lib_name,
                         get_type_string(out.dtype()),
                         get_type_string(precise ? float32 : out.dtype()));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool reverse,
    bool inclusive,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::scan()
                  << fmt::format(
                         scan_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         op_name(out),
                         inclusive,
                         reverse);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    int bn,
    int tn) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::sort()
                  << fmt::format(
                         block_sort_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         bn,
                         tn);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_mb_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& idx,
    int bn,
    int tn) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::sort()
                  << fmt::format(
                         multiblock_sort_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(idx.dtype()),
                         bn,
                         tn);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  auto lib = d.get_library(kernel_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::reduction()
                  << fmt::format(
                         reduce_init_kernels,
                         kernel_name,
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(kernel_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    bool non_atomic = out.dtype() == int64 || out.dtype() == uint64;
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::reduction() << metal::reduce()
                  << fmt::format(
                         non_atomic ? reduce_non_atomic_kernels
                                    : reduce_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         op_name(out));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

} // namespace mlx::core
