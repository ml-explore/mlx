// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/qmm_naive.cuh"

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"

#include "cuda_jit_sources.h"

namespace mlx::core {

namespace {

inline auto make_cta_tiler(int itemsize, int m, int group_size, bool sm80) {
  bool enough_smem = sm80 && itemsize <= 2 && group_size <= 64;
  int tile_m = std::max(16, std::min(64, next_power_of_2(m)));
  int tile_n = enough_smem ? 128 : 64;
  int tile_k = std::max(64, group_size);
  return cute::make_shape(tile_m, tile_n, tile_k);
}

} // namespace

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);
  bool sm80 = encoder.device().compute_capability_major() >= 8;
  auto cta_tiler = make_cta_tiler(x.itemsize(), m, group_size, sm80);
  bool has_k_residue = (k % cute::size<2>(cta_tiler)) != 0;

  std::string module_name = fmt::format(
      "qmm_naive_t{}_{}_{}_m{}_b{}_g{}_{}",
      transpose ? "n" : "t",
      has_k_residue ? "residue" : "aligned",
      dtype_to_string(x.dtype()),
      cute::size<0>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits, mode);
  std::string kernel_name = fmt::format(
      "mlx::core::cu::qmm_naive_kernel<{}, {}, {}, {}, {}, {}, {}, {}>",
      group_size,
      transpose,
      has_k_residue,
      sm80,
      ctype_x,
      ctype_q,
      ctype_s,
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_qmm_naive, std::vector{kernel_name});
  });

  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(scales);
  if (biases) {
    encoder.set_input_array(*biases);
  }
  if (global_scale) {
    encoder.set_input_array(*global_scale);
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
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma(cta_tiler)))};

  auto [sA_layout, sB_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes =
      x.itemsize() * (cute::cosize(sA_layout) + cute::cosize(sB_layout));

  encoder.add_kernel_node_ex(
      mod.get_kernel(kernel_name),
      num_blocks,
      block_dims,
      {},
      smem_bytes,
      gpu_ptr<void>(x),
      gpu_ptr<void>(w),
      gpu_ptr<void>(scales),
      biases ? gpu_ptr<void>(*biases) : nullptr,
      global_scale ? gpu_ptr<void>(*global_scale) : nullptr,
      lhs_indices ? gpu_ptr<void>(*lhs_indices) : nullptr,
      rhs_indices ? gpu_ptr<void>(*rhs_indices) : nullptr,
      gpu_ptr<void>(out),
      m,
      n,
      k,
      l,
      broadcast_b);
}

} // namespace mlx::core
