// Copyright © 2024 Apple Inc.

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
    const array& out);

MTL::ComputePipelineState* get_binary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_binary_two_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_ternary_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& out);

MTL::ComputePipelineState* get_copy_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const array& in,
    const array& out);

MTL::ComputePipelineState* get_softmax_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool precise,
    const array& out);

MTL::ComputePipelineState* get_scan_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    bool reverse,
    bool inclusive,
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
    const array& out);

MTL::ComputePipelineState* get_reduce_kernel(
    metal::Device& d,
    const std::string& kernel_name,
    const std::string& op_name,
    const array& in,
    const array& out);

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

MTL::ComputePipelineState* get_steel_conv_general_kernel(
    metal::Device& d,
    const std::string& kernel_name,
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
    const int tg_mem_size,
    const std::string& in_type,
    const std::string& out_type,
    int step,
    bool real,
    const metal::MTLFCList& func_consts);

} // namespace mlx::core
