// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/qmm_naive.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

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

inline auto cta_tiler_to_string(auto cta_tiler) {
  return fmt::format(
      "cute::Shape<cute::Int<{}>, cute::Int<{}>, cute::Int<{}>>",
      cute::size<0>(cta_tiler),
      cute::size<1>(cta_tiler),
      cute::size<2>(cta_tiler));
}

const char* get_weight_cutlass_type(const Dtype& dtype) {
  switch (dtype) {
    case float16:
      return "cutlass::half_t";
    case bfloat16:
      return "cutlass::bfloat16_t";
    case float32:
      return "float";
    default:
      throw std::invalid_argument(
          fmt::format(
              "[quantized_matmul] Unsupported dtype: {}.",
              dtype_to_string(dtype)));
  }
}

inline std::tuple<const char*, const char*>
get_quant_cutlass_types(const char* ctype_x, int bits, QuantizationMode mode) {
  if (mode == QuantizationMode::Mxfp4) {
    return {"cutlass::float_e2m1_t", "cutlass::float_ue8m0_t"};
  } else if (mode == QuantizationMode::Mxfp8) {
    return {"cutlass::float_e4m3_t", "cutlass::float_ue8m0_t"};
  } else if (mode == QuantizationMode::Nvfp4) {
    return {"cutlass::float_e2m1_t", "cutlass::float_e4m3_t"};
  } else {
    if (bits == 2) {
      return {"cutlass::uint2b_t", ctype_x};
    } else if (bits == 3) {
      return {"cutlass::uint3b_t", ctype_x};
    } else if (bits == 4) {
      return {"cutlass::uint4b_t", ctype_x};
    } else if (bits == 5) {
      return {"cutlass::uint5b_t", ctype_x};
    } else if (bits == 6) {
      return {"cutlass::uint6b_t", ctype_x};
    } else if (bits == 8) {
      return {"uint8_t", ctype_x};
    } else {
      throw std::invalid_argument(
          fmt::format(
              "[quantized_matmul] {}-bit quantization is not supported.",
              bits));
    }
  }
}

} // namespace

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int n = out.shape(-1);
  int k = x.shape(-1);
  int l = out.size() / (m * n);
  bool broadcast_b = (w.ndim() <= 2) || (w.size() != w.data_size());

  bool sm80 = encoder.device().compute_capability_major() >= 8;
  auto cta_tiler = make_cta_tiler(x.itemsize(), m, group_size, sm80);
  bool has_k_residue = (k % cute::size<2>(cta_tiler)) != 0;

  std::string module_name = fmt::format(
      "qmm_naive_{}_{}_{}_m{}_b{}_g{}_{}",
      dtype_to_string(x.dtype()),
      transpose ? "k" : "n",
      has_k_residue ? "residue" : "aligned",
      cute::size<0>(cta_tiler),
      bits,
      group_size,
      quantization_mode_to_string(mode));

  auto ctype_x = get_weight_cutlass_type(x.dtype());
  auto [ctype_q, ctype_s] = get_quant_cutlass_types(ctype_x, bits, mode);

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
