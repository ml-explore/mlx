// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device/qmm_sm90.cuh"

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/jit_module.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"
#include "mlx/backend/gpu/copy.h"

#include "cuda_jit_sources.h"

namespace mlx::core {

using namespace cute;

namespace {

inline array transpose_last_2_dims(
    const array& x,
    cu::CommandEncoder& encoder,
    const Stream& s) {
  array transposed = swapaxes_in_eval(x, -1, -2);
  array transposed_copy = contiguous_copy_gpu(transposed, s);
  encoder.add_temporary(transposed_copy);
  return transposed_copy;
}

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

template <typename F>
inline void dispatch_quant_types(int bits, const char* tag, F&& f) {
  if (bits == 2) {
    f.template operator()<cutlass::uint2b_t>();
  } else if (bits == 4) {
    f.template operator()<cutlass::uint4b_t>();
  } else if (bits == 8) {
    f.template operator()<uint8_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} {}-bit quantization is not supported.", tag, bits));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

template <typename F>
inline void dispatch_tile(int m, F&& f) {
  if (m <= 16) {
    f.template operator()<16>();
  } else if (m <= 32) {
    f.template operator()<32>();
  } else if (m <= 64) {
    f.template operator()<64>();
  } else if (m <= 128) {
    f.template operator()<128>();
  } else {
    f.template operator()<256>();
  }
}

template <typename F>
inline void dispatch_gemm(
    const array& x,
    int n,
    int bits,
    int group_size,
    const char* tag,
    F&& f) {
  dispatch_element_types(x.dtype(), tag, [&]<typename Element>() {
    dispatch_tile(n, [&]<int TileN>() {
      dispatch_quant_types(bits, tag, [&]<typename Quant>() {
        dispatch_groups(group_size, tag, [&]<int GroupSize>() {
          auto cta_tiler = make_shape(
              Int<128>{},
              Int<TileN>{},
              Int<std::max(64, 128 * 8 / sizeof_bits_v<Element>)>{});
          auto gemm = cu::make_qmm_sm90_kernel<
              GroupSize,
              Element,
              Quant,
              decltype(cta_tiler)>();
          f(cta_tiler, gemm);
        });
      });
    });
  });
}

} // namespace

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales_,
    const array& biases_,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  const char* tag = "[quantized_matmul]";
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);

  auto dA = make_stride(int64_t(k), Int<1>{}, int64_t(m * k));
  auto dB = make_stride(int64_t(k), Int<1>{}, int64_t(n * k));
  auto dS = make_stride(Int<1>{}, int64_t(n), int64_t(n * k / group_size));
  auto dD = make_stride(Int<1>{}, int64_t(n), int64_t(m * n));
  if (broadcast_b) {
    get<2>(dB) = 0;
    get<2>(dS) = 0;
  }

  // FIXME: Copy happens for every call.
  array scales = transpose_last_2_dims(scales_, encoder, s);
  array biases = transpose_last_2_dims(biases_, encoder, s);

  dispatch_gemm(x, n, bits, group_size, tag, [&](auto cta_tiler, auto gemm) {
    // JIT compilation.
    std::string module_name = fmt::format(
        "qmm_sm90_tn_{}_n{}_b{}_g{}_affine",
        dtype_to_string(x.dtype()),
        int(size<1>(cta_tiler)),
        bits,
        group_size);

    auto [ctype_x, ctype_q, ctype_s] = get_qmm_cutlass_types(x, bits);
    std::string kernel_name = fmt::format(
        "cutlass::device_kernel<mlx::core::cu::qmm_sm90_kernel_t<{}, {}, {}, {}>>",
        group_size,
        ctype_x,
        ctype_q,
        cta_tiler_to_string(cta_tiler));

    cu::JitModule& mod =
        cu::get_jit_module(encoder.device(), module_name, [&]() {
          return std::make_tuple(
              false, jit_source_qmm_sm90, std::vector{kernel_name});
        });

    // Prepare kernel args.
    using Gemm = decltype(gemm);

    const auto* A = gpu_ptr<typename Gemm::ElementB>(x);
    const auto* B = gpu_ptr<typename Gemm::ElementA>(w);
    const auto* S =
        gpu_ptr<typename Gemm::CollectiveMainloop::ElementScale>(scales);
    const auto* Z =
        gpu_ptr<typename Gemm::CollectiveMainloop::ElementZero>(biases);
    auto* D = gpu_ptr<typename Gemm::ElementD>(out);

    auto params = Gemm::to_underlying_arguments(
        {cutlass::gemm::GemmUniversalMode::kGemm,
         {n, m, k, l},
         {B, dB, A, dA, S, dS, group_size, Z},
         {{1.f, 0.f}, D, dD, D, dD}},
        nullptr);

    size_t smem_bytes = Gemm::SharedStorageSize;
    auto kernel = mod.get_kernel(kernel_name, [&](CUfunction kernel) {
      if (smem_bytes > 48000) {
        cuFuncSetAttribute(
            kernel,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes);
      }
    });

    // Append to CUDA graph.
    encoder.set_input_array(x);
    encoder.set_input_array(w);
    encoder.set_input_array(scales);
    encoder.set_input_array(biases);
    encoder.set_output_array(out);
    encoder.add_kernel_node_ex(
        kernel,
        Gemm::get_grid_shape(params),
        Gemm::get_block_shape(),
        {},
        smem_bytes,
        params);
  });
}

} // namespace mlx::core
