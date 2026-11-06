// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/qmm_sm80.cuh"

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"

#include "cuda_jit_sources.h"

namespace mlx::core {

namespace {

inline auto make_cta_tiler(int m, int group_size) {
  int tile_m = std::max(16, std::min(64, next_power_of_2(m)));
  int tile_n = 128;
  int tile_k = std::max(64, group_size);
  return cute::make_shape(tile_m, tile_n, tile_k);
}

inline int get_quantized_batch_count(const array& w) {
  return w.ndim() > 2 ? w.size() / (w.shape(-1) * w.shape(-2)) : 1;
}

} // namespace

void qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);
  auto cta_tiler = make_cta_tiler(m, group_size);

  std::string module_name = fmt::format(
      "qmm_sm80_tn_{}_m{}_b{}_g{}_{}",
      dtype_to_string(x.dtype()),
      cute::size<0>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits, mode);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::qmm_sm80_kernel<{}, {}, {}, {}, {}>",
      group_size,
      ctype_x,
      ctype_q,
      ctype_s,
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_qmm_sm80, std::vector{kernel_name});
  });

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  if (biases) {
    encoder.set_input_array(*biases);
  }
  if (lhs_indices) {
    encoder.set_input_array(*lhs_indices);
  }
  if (rhs_indices) {
    encoder.set_input_array(*rhs_indices);
  }
  encoder.set_output_array(out);

  dim3 num_blocks{
      uint32_t(cute::ceil_div(m, cute::size<0>(cta_tiler))),
      uint32_t(cute::ceil_div(n, cute::size<1>(cta_tiler))),
      uint32_t(l)};
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma()))};

  auto [sA_layout, sB_layout, sC_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes = std::max(
      cute::cosize(sA_layout) * x.itemsize() +
          cute::cosize(sB_layout) * bits / 8,
      cute::cosize(sC_layout) * x.itemsize());

  auto kernel = mod.get_kernel(kernel_name, [&](CUfunction kernel) {
    if (smem_bytes > 48000) {
      cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    }
  });

  encoder.add_kernel_node_ex(
      kernel,
      num_blocks,
      block_dims,
      {},
      smem_bytes,
      gpu_ptr<void>(x),
      gpu_ptr<void>(w),
      gpu_ptr<void>(scales),
      biases ? gpu_ptr<void>(*biases) : nullptr,
      lhs_indices ? gpu_ptr<void>(*lhs_indices) : nullptr,
      rhs_indices ? gpu_ptr<void>(*rhs_indices) : nullptr,
      gpu_ptr<void>(out),
      m,
      n,
      k,
      l,
      broadcast_b);
}

bool supports_gather_qmm_rhs_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int out_m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (out_m != 1) {
    return false;
  }
  if (!supports_qmm_sm80(
          x,
          w,
          scales,
          biases,
          out,
          transpose,
          bits,
          group_size,
          mode,
          device)) {
    return false;
  }

  int rows = out.size() / out.shape(-1);
  int input_rows = x.size() / x.shape(-1);
  int group_count = get_quantized_batch_count(w);
  return input_rows == rows && rows >= 16 && group_count > 0 &&
      rows / group_count >= 4;
}

void gather_qmm_rhs_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  int m = x.size() / x.shape(-1);
  int n = out.shape(-1);
  int k = x.shape(-1);
  int group_count = get_quantized_batch_count(w);
  auto cta_tiler = make_cta_tiler(m, group_size);

  std::string module_name = fmt::format(
      "gather_qmm_rhs_sm80_tn_{}_m{}_b{}_g{}_{}",
      dtype_to_string(x.dtype()),
      cute::size<0>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits, mode);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::qmm_sm80_rhs_kernel<{}, {}, {}, {}, {}>",
      group_size,
      ctype_x,
      ctype_q,
      ctype_s,
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_qmm_sm80, std::vector{kernel_name});
  });

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  if (biases) {
    encoder.set_input_array(*biases);
  }
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  dim3 num_blocks{
      uint32_t(cute::ceil_div(m, cute::size<0>(cta_tiler))),
      uint32_t(cute::ceil_div(n, cute::size<1>(cta_tiler))),
      1};
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma()))};

  auto [sA_layout, sB_layout, sC_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes = std::max(
      cute::cosize(sA_layout) * x.itemsize() +
          cute::cosize(sB_layout) * bits / 8,
      cute::cosize(sC_layout) * x.itemsize());

  auto kernel = mod.get_kernel(kernel_name, [&](CUfunction kernel) {
    if (smem_bytes > 48000) {
      cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes);
    }
  });

  encoder.add_kernel_node_ex(
      kernel,
      num_blocks,
      block_dims,
      {},
      smem_bytes,
      gpu_ptr<void>(x),
      gpu_ptr<void>(w),
      gpu_ptr<void>(scales),
      biases ? gpu_ptr<void>(*biases) : nullptr,
      gpu_ptr<void>(rhs_indices),
      gpu_ptr<void>(out),
      m,
      n,
      k,
      group_count);
}

} // namespace mlx::core
