// Copyright © 2024 Apple Inc.

#pragma once

#include <fmt/format.h>

#include "mlx/array.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core {

MTL::ComputePipelineState* get_arange_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out);

MTL::ComputePipelineState* get_unary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op);

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op);

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype in_type,
    Dtype out_type,
    const char* op);

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    Dtype type,
    const char* op);

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_dynamic_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool precise,
    const array& out);

MTL::ComputePipelineState* get_logsumexp_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out);

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool reverse,
    bool inclusive,
    const std::string& reduce_type,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    int bn,
    int tn);

MTL::ComputePipelineState* get_mb_sort_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& idx,
    int bn,
    int tn);

MTL::ComputePipelineState* get_reduce_init_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& func_name,
    const std::string& op_name,
    const Dtype& out_type);

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& func_name,
    const std::string& op_name,
    const Dtype& in_type,
    const Dtype& out_type,
    const std::string& idx_t,
    int ndim = -1,
    int bm = -1,
    int bn = -1);

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
    int wn);

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
    bool k_aligned);

MTL::ComputePipelineState* get_steel_gemm_splitk_accum_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out,
    bool axbpy);

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
    bool k_aligned);

MTL::ComputePipelineState* get_steel_gemm_gather_kernel(
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
    int wn,
    bool rhs);

MTL::ComputePipelineState* get_steel_gemm_segmented_kernel(
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
    int wn);

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
    bool small_filter);

MTL::ComputePipelineState* get_steel_conv_3d_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool small_filter);

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
    bool contiguous);

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& out,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn);

MTL::ComputePipelineState* get_fft_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const std::string& template_def);

MTL::ComputePipelineState* get_quantized_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& template_def,
    const std::string& mode);

MTL::ComputePipelineState* get_gather_qmm_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& x,
    int group_size,
    int bits,
    const std::string& mode,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool transpose);

MTL::ComputePipelineState* get_steel_gemm_fused_nax_kernel(
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
    int wn);

MTL::ComputePipelineState* get_steel_gemm_gather_nax_kernel(
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
    int wn,
    bool rhs);

MTL::ComputePipelineState* get_steel_gemm_splitk_nax_kernel(
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
    int wn);

MTL::ComputePipelineState* get_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& template_def,
    const std::string& mode);

MTL::ComputePipelineState* get_gather_qmm_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& x,
    int group_size,
    int bits,
    const std::string& mode,
    int bm,
    int bn,
    int bk,
    int wm,
    int wn,
    bool transpose);

MTL::ComputePipelineState* get_steel_attention_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& q,
    int bq,
    int bk,
    int bd,
    int wm,
    int wn,
    const array& m);

MTL::ComputePipelineState* get_steel_attention_nax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& hash_name,
    const metal::MTLFCList& func_consts,
    const array& q,
    int bq,
    int bk,
    int bd,
    int wm,
    int wn,
    const array& m);

// Create a GPU kernel template definition for JIT compilation
template <typename... Args>
std::string get_template_definition(
    std::string_view name,
    std::string_view func,
    Args... args) {
  std::ostringstream s;
  s << func << "<";
  bool first = true;
  auto add_arg = [&s, &first](const auto& arg) {
    if (!first) {
      s << ", ";
    }
    first = false;
    s << arg;
  };
  (add_arg(args), ...);
  s << ">";
  return fmt::format(
      "\ntemplate [[host_name(\"{0}\")]] [[kernel]] decltype({1}) {1};\n",
      name,
      s.str());
}

} // namespace mlx::core
