// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

template <typename GroupSize, typename Element, typename Quant, typename F>
void qmm_sm90(
    const Element* A,
    const Quant* B,
    const Element* S,
    const Element* Z,
    Element* D,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t l,
    GroupSize group_size,
    F&& launch_kernel) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  using namespace cute;

  constexpr int kAlignmentA = 128 / sizeof_bits<Element>::value;
  constexpr int kAlignmentB = 128 / sizeof_bits<Quant>::value;
  constexpr int kTileShapeK =
      std::max(64, 128 * 8 / sizeof_bits<Element>::value);
  static_assert(group_size % kTileShapeK == 0);

  using Arch = cutlass::arch::Sm90;
  using Accumulator = float;
  using TileShape = Shape<_128, _16, Int<kTileShapeK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator,
      Accumulator,
      // ElementC:
      void,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      // ElementD:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  // Note that A/B are swapped and transposed to use TMA epilogue.
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      // ElementA:
      cute::tuple<Quant, Element, Element>,
      cutlass::layout::RowMajor,
      kAlignmentB,
      // ElementB:
      Element,
      cutlass::layout::ColumnMajor,
      kAlignmentA,
      Accumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  auto dA = make_stride(k, Int<1>{}, m * k);
  auto dB = make_stride(k, Int<1>{}, n * k);
  auto dS = make_stride(Int<1>{}, n, n * k / group_size);
  auto dD = make_stride(Int<1>{}, n, m * n);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {int(n), int(m), int(k), int(l)},
      {B, dB, A, dA, S, dS, group_size, Z},
      {{1.f, 0.f}, D, dD, D, dD}};

  CHECK_CUTLASS_ERROR(gemm.can_implement(args));
  CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));

  auto* kernel = &cutlass::device_kernel<GemmKernel>;
  void* kernel_params[] = {const_cast<Gemm::Params*>(&gemm.params())};
  launch_kernel(
      reinterpret_cast<void*>(kernel),
      gemm.get_grid_shape(gemm.params()),
      GemmKernel::get_block_shape(),
      GemmKernel::SharedStorageSize,
      kernel_params);
#else
  throw std::runtime_error(
      "[quantized_matmul] Hopper-only kernel is not available.");
#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
}

} // namespace cutlass_gemm

namespace mlx::core {

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
    f(cute::Int<64>{});
  } else if (group_size == 128) {
    f(cute::Int<128>{});
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

void qmm_sm90(
    const array& x_,
    const array& w,
    const array& scales_,
    const std::optional<array>& biases_,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder,
    Stream s) {
  if ((mode != QuantizationMode::Affine) || !biases_) {
    throw std::runtime_error("qmm_sm90 NYI");
  }

  const char* tag = "[quantized_matmul]";
  int m = out.shape(-2);
  int n = out.shape(-1);
  int k = x_.shape(-1);
  int l = out.size() / (m * n);
  if (k % 64 != 0) {
    throw std::runtime_error(fmt::format("{} K must be multiples of 64.", tag));
  }
  if (!w.flags().row_contiguous) {
    throw std::runtime_error(
        fmt::format("{} Weights must be row contiguous.", tag));
  }
  if (!scales_.flags().row_contiguous) {
    throw std::runtime_error(
        fmt::format("{} Scales must be row contiguous.", tag));
  }
  if (!biases_->flags().row_contiguous) {
    throw std::runtime_error(
        fmt::format("{} Biases must be row contiguous.", tag));
  }

  // TODO: Support column-major x.
  array x = ensure_row_contiguous(x_, encoder, s);
  // FIXME: Copy happens for every call.
  array scales = transpose_last_2_dims(scales_, encoder, s);
  array biases = transpose_last_2_dims(*biases_, encoder, s);

  dispatch_element_types(out.dtype(), tag, [&]<typename Element>() {
    dispatch_quant_types(bits, tag, [&]<typename Quant>() {
      dispatch_groups(group_size, tag, [&](auto group_size) {
        encoder.set_input_array(x);
        encoder.set_input_array(w);
        encoder.set_input_array(scales);
        encoder.set_input_array(biases);
        encoder.set_output_array(out);
        cutlass_gemm::qmm_sm90(
            gpu_ptr<Element>(x),
            gpu_ptr<Quant>(w),
            gpu_ptr<Element>(scales),
            gpu_ptr<Element>(biases),
            gpu_ptr<Element>(out),
            m,
            n,
            k,
            l,
            group_size,
            [&](auto* kernel,
                dim3 num_blocks,
                dim3 block_dims,
                uint32_t smem_bytes,
                void** args) {
              encoder.add_kernel_node(
                  kernel, num_blocks, block_dims, smem_bytes, args);
            });
      });
    });
  });
}

} // namespace mlx::core
