// Copyright © 2024 Apple Inc.
#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/jit/arange.h"
#include "mlx/backend/metal/jit/binary.h"
#include "mlx/backend/metal/jit/binary_two.h"
#include "mlx/backend/metal/jit/copy.h"
#include "mlx/backend/metal/jit/fft.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/jit/reduce.h"
#include "mlx/backend/metal/jit/scan.h"
#include "mlx/backend/metal/jit/softmax.h"
#include "mlx/backend/metal/jit/sort.h"
#include "mlx/backend/metal/jit/steel_conv.h"
#include "mlx/backend/metal/jit/steel_gemm.h"
#include "mlx/backend/metal/jit/ternary.h"
#include "mlx/backend/metal/jit/unary.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"

using namespace fmt::literals;

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
    kernel_source << metal::utils() << metal::reduce_utils()
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
    const std::string& op_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::string op_type = op_name;
    op_type[0] = std::toupper(op_name[0]);
    bool non_atomic = out.dtype() == int64 || out.dtype() == uint64;
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::reduce_utils() << metal::reduce()
                  << fmt::format(
                         non_atomic ? reduce_non_atomic_kernels
                                    : reduce_kernels,
                         lib_name,
                         get_type_string(in.dtype()),
                         get_type_string(out.dtype()),
                         op_type);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_fused_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_fused()
                  << fmt::format(
                         steel_gemm_fused_kernels,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(out.dtype()),
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "bk"_a = bk,
                         "wm"_a = wm,
                         "wn"_a = wn,
                         "trans_a"_a = transpose_a,
                         "trans_b"_a = transpose_b);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool mn_aligned,
    bool k_aligned) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << fmt::format(
                         steel_gemm_splitk_kernels,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(in.dtype()),
                         "otype"_a = get_type_string(out.dtype()),
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "bk"_a = bk,
                         "wm"_a = wm,
                         "wn"_a = wn,
                         "trans_a"_a = transpose_a,
                         "trans_b"_a = transpose_b,
                         "mn_aligned"_a = mn_aligned,
                         "k_aligned"_a = k_aligned);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool axbpy) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << fmt::format(
                         axbpy ? steel_gemm_splitk_accum_axbpy_kernels
                               : steel_gemm_splitk_accum_kernels,
                         "name"_a = lib_name,
                         "atype"_a = get_type_string(in.dtype()),
                         "otype"_a = get_type_string(out.dtype()));
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool transpose_a,
    bool transpose_b,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool mn_aligned,
    bool k_aligned) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    auto out_mask_type = mask_out.has_value()
        ? get_type_string((*mask_out).dtype())
        : "nomask_t";
    auto op_mask_type =
        mask_op.has_value() ? get_type_string((*mask_op).dtype()) : "nomask_t";
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_masked()
                  << fmt::format(
                         steel_gemm_masked_kernels,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(out.dtype()),
                         "outmasktype"_a = out_mask_type,
                         "opmasktype"_a = op_mask_type,
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "bk"_a = bk,
                         "wm"_a = wm,
                         "wn"_a = wn,
                         "trans_a"_a = transpose_a,
                         "trans_b"_a = transpose_b,
                         "mn_aligned"_a = mn_aligned,
                         "k_aligned"_a = k_aligned);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_conv_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    int n_channel_specialization,
    bool small_filter) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::conv() << metal::steel_conv()
                  << fmt::format(
                         steel_conv_kernels,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(out.dtype()),
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "bk"_a = bk,
                         "wm"_a = wm,
                         "wn"_a = wn,
                         "n_channels"_a = n_channel_specialization,
                         "small_filter"_a = small_filter);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::conv()
                  << metal::steel_conv_general()
                  << fmt::format(
                         steel_conv_general_kernels,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(out.dtype()),
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "bk"_a = bk,
                         "wm"_a = wm,
                         "wn"_a = wn);
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const int tg_mem_size,
    const std::string& in_type,
    const std::string& out_type,
    int step,
    bool real,
    const metal::MTLFCList& func_consts) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    std::ostringstream kernel_source;
    std::string kernel_string;
    if (lib_name.find("bluestein") != std::string::npos) {
      kernel_string = bluestein_fft_kernel;
    } else if (lib_name.find("rader") != std::string::npos) {
      kernel_string = rader_fft_kernel;
    } else if (lib_name.find("four_step") != std::string::npos) {
      kernel_string = four_step_fft_kernel;
    } else {
      kernel_string = fft_kernel;
    }
    kernel_source << metal::fft();
    if (lib_name.find("four_step") != std::string::npos) {
      kernel_source << fmt::format(
          kernel_string,
          "name"_a = lib_name,
          "tg_mem_size"_a = tg_mem_size,
          "in_T"_a = in_type,
          "out_T"_a = out_type,
          "step"_a = step,
          "real"_a = real);
    } else {
      kernel_source << fmt::format(
          kernel_string,
          "name"_a = lib_name,
          "tg_mem_size"_a = tg_mem_size,
          "in_T"_a = in_type,
          "out_T"_a = out_type);
    }
    lib = d.get_library(lib_name, kernel_source.str());
  }
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

} // namespace mlx::core
