// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/jit/arange.h"
#include "mlx/backend/metal/jit/gemv_masked.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/jit/softmax.h"
#include "mlx/backend/metal/jit/steel_conv.h"
#include "mlx/backend/metal/jit/steel_gemm.h"
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
  auto lib = d.get_library(kernel_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::arange()
                  << fmt::format(
                         arange_kernels,
                         kernel_name,
                         get_type_string(out.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const std::string op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    auto in_t = get_type_string(in_type);
    auto out_t = get_type_string(out_type);
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::unary_ops() << metal::unary();
    kernel_source << get_template_definition(
        "v_" + lib_name, "unary_v", in_t, out_t, op);
    kernel_source << get_template_definition(
        "v2_" + lib_name, "unary_v2", in_t, out_t, op);
    kernel_source << get_template_definition(
        "gn4_" + lib_name, "unary_g", in_t, out_t, op, 4);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

void add_binary_kernels(
    const std::string lib_name,
    Dtype in_type,
    Dtype out_type,
    const std::string op,
    std::ostringstream& kernel_source) {
  const std::array<std::pair<std::string, std::string>, 10> kernel_types = {{
      {"ss", "binary_ss"},
      {"vs", "binary_vs"},
      {"sv", "binary_sv"},
      {"vv", "binary_vv"},
      {"vs2", "binary_vs2"},
      {"sv2", "binary_sv2"},
      {"vv2", "binary_vv2"},
      {"g1", "binary_g_nd1"},
      {"g2", "binary_g_nd2"},
      {"g3", "binary_g_nd3"},
  }};
  for (auto& [name, func] : kernel_types) {
    std::string template_def;
    template_def = get_template_definition(
        name + "_" + lib_name,
        func,
        get_type_string(in_type),
        get_type_string(out_type),
        op);
    kernel_source << template_def;
  }
  kernel_source << get_template_definition(
      "gn4_" + lib_name,
      "binary_g",
      get_type_string(in_type),
      get_type_string(out_type),
      op,
      4);
}

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const std::string op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::binary_ops() << metal::binary();
    add_binary_kernels(lib_name, in_type, out_type, op, kernel_source);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const std::string op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::binary_ops()
                  << metal::binary_two();
    add_binary_kernels(lib_name, in_type, out_type, op, kernel_source);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype type,
    const std::string op) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    const std::array<std::pair<std::string, std::string>, 5> kernel_types = {{
        {"v", "ternary_v"},
        {"v2", "ternary_v2"},
        {"g1", "ternary_g_nd1"},
        {"g2", "ternary_g_nd2"},
        {"g3", "ternary_g_nd3"},
    }};
    kernel_source << metal::utils() << metal::ternary_ops() << metal::ternary();
    for (auto& [name, func] : kernel_types) {
      std::string template_def;
      template_def = get_template_definition(
          name + "_" + lib_name, func, get_type_string(type), op);
      kernel_source << template_def;
    }
    kernel_source << get_template_definition(
        "gn4_" + lib_name, "ternary_g", get_type_string(type), op, 4);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    kernel_source << metal::utils() << metal::copy()
                  << get_template_definition(
                         "s_" + lib_name, "copy_s", in_type, out_type)
                  << get_template_definition(
                         "v_" + lib_name, "copy_v", in_type, out_type)
                  << get_template_definition(
                         "g1_" + lib_name, "copy_g_nd1", in_type, out_type)
                  << get_template_definition(
                         "g2_" + lib_name, "copy_g_nd2", in_type, out_type)
                  << get_template_definition(
                         "g3_" + lib_name, "copy_g_nd3", in_type, out_type)
                  << get_template_definition(
                         "gn4_" + lib_name, "copy_g", in_type, out_type, 4)
                  << get_template_definition(
                         "gg1_" + lib_name, "copy_gg_nd1", in_type, out_type)
                  << get_template_definition(
                         "gg2_" + lib_name, "copy_gg_nd2", in_type, out_type)
                  << get_template_definition(
                         "gg3_" + lib_name, "copy_gg_nd3", in_type, out_type)
                  << get_template_definition(
                         "ggn4_" + lib_name, "copy_gg", in_type, out_type, 4);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool precise,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&] {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::softmax()
                  << fmt::format(
                         softmax_kernels,
                         lib_name,
                         get_type_string(out.dtype()),
                         get_type_string(precise ? float32 : out.dtype()));
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool reverse,
    bool inclusive,
    const std::string& reduce_type,
    const array& in,
    const array& out) {
  std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
  auto lib = d.get_library(lib_name, [&]() {
    auto out_type = get_type_string(out.dtype());
    std::string op = "Cum" + reduce_type + "<" + out_type + ">";
    op[3] = toupper(op[3]);
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::scan();
    const std::array<std::pair<std::string, std::string>, 2> scan_kernels = {{
        {"contig_", "contiguous_scan"},
        {"strided_", "strided_scan"},
    }};
    for (auto& [prefix, kernel] : scan_kernels) {
      kernel_source << get_template_definition(
          prefix + lib_name,
          kernel,
          get_type_string(in.dtype()),
          get_type_string(out.dtype()),
          op,
          in.itemsize() <= 4 ? 4 : 2,
          inclusive,
          reverse);
    }
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    kernel_source << metal::utils() << metal::sort();
    for (bool is_argsort : {true, false}) {
      std::string bool_string = is_argsort ? "true" : "false";
      std::string func_string = is_argsort ? "carg_" : "c_";
      kernel_source << get_template_definition(
          func_string + lib_name,
          "block_sort",
          in_type,
          out_type,
          bool_string,
          bn,
          tn);
      kernel_source << get_template_definition(
          "n" + func_string + lib_name,
          "block_sort_nc",
          in_type,
          out_type,
          bool_string,
          bn,
          tn);
    }
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::sort();
    std::array<std::pair<std::string, std::string>, 3> kernel_types = {
        {{"sort_", "mb_block_sort"},
         {"partition_", "mb_block_partition"},
         {"merge_", "mb_block_merge"}}};
    for (auto& [name, func] : kernel_types) {
      kernel_source << get_template_definition(
          name + lib_name,
          func,
          get_type_string(in.dtype()),
          get_type_string(idx.dtype()),
          "true",
          bn,
          tn);
    }
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::ostringstream kernel_source;
    std::string op_type = op_name(out);
    op_type[0] = std::toupper(op_name(out)[0]);
    auto out_type = get_type_string(out.dtype());
    std::string op = op_type + "<" + out_type + ">";
    kernel_source << metal::utils() << metal::reduce_utils() << metal::reduce();
    kernel_source << get_template_definition(
        kernel_name, "init_reduce", out_type, op);
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& func_name,
    const std::string& op_name,
    const array& in,
    const array& out,
    int ndim /* = -1 */,
    int bm /* = -1 */,
    int bn /* = -1 */) {
  auto lib = d.get_library(kernel_name, [&]() {
    std::string op_type = op_name;
    op_type[0] = std::toupper(op_name[0]);
    std::ostringstream kernel_source;
    auto in_type = get_type_string(in.dtype());
    auto out_type = get_type_string(out.dtype());
    std::string op = op_type + "<" + out_type + ">";
    kernel_source << metal::utils() << metal::reduce_utils() << metal::reduce();
    if (bm >= 0) {
      kernel_source << get_template_definition(
          kernel_name, func_name, in_type, out_type, op, ndim, bm, bn);
    } else if (ndim >= 0) {
      kernel_source << get_template_definition(
          kernel_name, func_name, in_type, out_type, op, ndim);
    } else {
      kernel_source << get_template_definition(
          kernel_name, func_name, in_type, out_type, op);
    }
    return kernel_source.str();
  });
  auto st = d.get_kernel(kernel_name, lib);
  return st;
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
  auto lib = d.get_library(lib_name, [&]() {
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
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
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
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool axbpy) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm()
                  << metal::steel_gemm_splitk()
                  << fmt::format(
                         axbpy ? steel_gemm_splitk_accum_axbpy_kernels
                               : steel_gemm_splitk_accum_kernels,
                         "name"_a = lib_name,
                         "atype"_a = get_type_string(in.dtype()),
                         "otype"_a = get_type_string(out.dtype()));
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
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
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_gemv_masked_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    const std::optional<array>& mask_out,
    const std::optional<array>& mask_op,
    bool transpose_mat,
    int bm,
    int bn,
    int sm,
    int sn,
    int tm,
    int tn,
    bool contiguous) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    auto out_mask_type = mask_out.has_value()
        ? get_type_string((*mask_out).dtype())
        : "nomask_t";
    auto op_mask_type =
        mask_op.has_value() ? get_type_string((*mask_op).dtype()) : "nomask_t";
    kernel_source << metal::utils() << metal::gemv_masked()
                  << fmt::format(
                         gemv_masked_kernel,
                         "name"_a = lib_name,
                         "itype"_a = get_type_string(out.dtype()),
                         "outm_t"_a = out_mask_type,
                         "opm_t"_a = op_mask_type,
                         "bm"_a = bm,
                         "bn"_a = bn,
                         "sm"_a = sm,
                         "sn"_a = sn,
                         "tm"_a = tm,
                         "tn"_a = tn,
                         "trans"_a = transpose_mat ? "t_" : "",
                         "nc"_a = contiguous ? "0" : "1");
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
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
    return kernel_source.str();
  });
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
  auto lib = d.get_library(lib_name, [&]() {
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
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string& template_def) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    std::string kernel_string;
    kernel_source << metal::fft() << template_def;
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib, hash_name, func_consts);
}

MTL::ComputePipelineState* get_quantized_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& template_def) {
  const auto& lib_name = kernel_name;
  auto lib = d.get_library(lib_name, [&]() {
    std::ostringstream kernel_source;
    kernel_source << metal::utils() << metal::gemm() << metal::quantized()
                  << template_def;
    return kernel_source.str();
  });
  return d.get_kernel(kernel_name, lib);
}

} // namespace mlx::core
