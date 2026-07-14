// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/gemm_sm70.cuh"

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include "cuda_jit_sources.h"

namespace mlx::core {

namespace {

inline auto make_cta_tiler(int m) {
  int tile_m = std::max(16, std::min(256, next_power_of_2(m)));
  int tile_n = 128;
  int tile_k = 64;
  return cute::make_shape(tile_m, tile_n, tile_k);
}

} // namespace

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

void gather_mm(
    bool a_transposed,
    bool b_transposed,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    cu::CommandEncoder& encoder) {
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = a.shape(-1);
  int l = out.size() / (m * n);

  bool aligned = (k % 8 == 0);
  bool sm80 = encoder.device().compute_capability_major() >= 8;
  auto cta_tiler = make_cta_tiler(m);

  std::string module_name = fmt::format(
      "gemm_sm70_{}{}_{}_{}_m{}_n{}_k{}",
      a_transposed ? "n" : "t",
      b_transposed ? "n" : "t",
      dtype_to_string(out.dtype()),
      aligned ? "aligned" : "unaligned",
      cute::size<0>(cta_tiler),
      cute::size<1>(cta_tiler),
      cute::size<2>(cta_tiler));

  std::string kernel_name = fmt::format(
      "mlx::core::cu::gemm_sm70_kernel<{}, {}, {}, {}, {}, {}>",
      !a_transposed,
      b_transposed,
      aligned,
      sm80,
      dtype_to_cutlass_type(out.dtype()),
      cta_tiler_to_string(cta_tiler));

  cu::JitModule& mod = cu::get_jit_module(encoder.device(), module_name, [&]() {
    return std::make_tuple(
        false, jit_source_gemm_sm70, std::vector{kernel_name});
  });

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  dim3 num_blocks{
      uint32_t(cute::ceil_div(m, cute::size<0>(cta_tiler))),
      uint32_t(cute::ceil_div(n, cute::size<1>(cta_tiler))),
      uint32_t(l)};
  dim3 block_dims{uint32_t(cute::size(cu::make_tiled_mma(cta_tiler)))};

  auto [sA_layout, sB_layout] = cu::make_smem_layouts(cta_tiler);
  size_t smem_bytes =
      out.itemsize() * (cute::cosize(sA_layout) + cute::cosize(sB_layout));

  encoder.add_kernel_node_ex(
      mod.get_kernel(kernel_name),
      num_blocks,
      block_dims,
      {},
      smem_bytes,
      gpu_ptr<void>(a),
      gpu_ptr<void>(b),
      gpu_ptr<void>(lhs_indices),
      gpu_ptr<void>(rhs_indices),
      gpu_ptr<void>(out),
      m,
      n,
      k,
      l);
}

} // namespace mlx::core
