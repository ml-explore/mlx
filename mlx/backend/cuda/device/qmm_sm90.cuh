// Copyright © 2026 Apple Inc.

#if defined(__CUDACC_RTC__)

#include <cuda/cmath>
#include <cuda/std/type_traits>

// Some CUTLASS headers use following std functions but we can't use STL when
// compiling with NVRTC.
namespace std {
using cuda::std::is_pointer_v;
using cuda::std::max;
using cuda::std::void_t;
} // namespace std

// The cutlass/floating_point_nvrtc.h file assumes following constants not
// being defined but they are in the CUDA std headers.
#undef FP_NAN
#undef FP_INFINITE
#undef FP_ZERO
#undef FP_SUBNORMAL
#undef FP_NORMAL

#endif // defined(__CUDACC_RTC__)

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

namespace mlx::core::cu {

using namespace cute;

template <int GroupSize, typename Element, typename Quant, typename CtaTiler>
CUTLASS_HOST_DEVICE auto make_qmm_sm90_kernel() {
  constexpr int AlignmentA = 128 / sizeof_bits<Element>::value;
  constexpr int AlignmentB = 128 / sizeof_bits<Quant>::value;

  using Arch = cutlass::arch::Sm90;
  using Accumulator = float;
  using ClusterShape = Shape<_1, _1, _1>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      CtaTiler,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator,
      Accumulator,
      // ElementC:
      void,
      cutlass::layout::ColumnMajor,
      AlignmentA,
      // ElementD:
      Element,
      cutlass::layout::ColumnMajor,
      AlignmentA,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

  // Note that A/B are swapped and transposed to use TMA epilogue.
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch,
      cutlass::arch::OpClassTensorOp,
      // ElementA:
      tuple<Quant, Element, Element>,
      cutlass::layout::RowMajor,
      AlignmentB,
      // ElementB:
      Element,
      cutlass::layout::ColumnMajor,
      AlignmentA,
      Accumulator,
      CtaTiler,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  return cutlass::gemm::kernel::
      GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>{};
}

template <int GroupSize, typename Element, typename Quant, typename CtaTiler>
using qmm_sm90_kernel_t =
    decltype(make_qmm_sm90_kernel<GroupSize, Element, Quant, CtaTiler>());

} // namespace mlx::core::cu
