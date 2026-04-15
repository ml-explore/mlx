// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cutlass/epilogue/collective/collective_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

// We can't put kernel code in mlx::core due to name conflicts of "Shape".
namespace cutlass_gemm {

using namespace cute;

// Modified from cutlass/include/cutlass/gemm/kernel/sm70_gemm.hpp to fuse
// gather into GEMM.
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_>
class GatherGemm {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementA = typename CollectiveMainloop::ElementA;
  using StrideA = typename CollectiveMainloop::StrideA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideB = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD = typename CollectiveEpilogue::StrideD;

  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage)));
  static constexpr uint32_t MaxThreadsPerBlock =
      CUTE_STATIC_V(size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Arguments {
    ProblemShape problem_shape;
    const uint32_t* lhs_indices;
    const uint32_t* rhs_indices;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
  };

  struct Params {
    ProblemShape problem_shape;
    const uint32_t* lhs_indices;
    const uint32_t* rhs_indices;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
  };

  static Params to_underlying_arguments(
      const Arguments& args,
      void* workspace) {
    return {
        args.problem_shape,
        args.lhs_indices,
        args.rhs_indices,
        CollectiveMainloop::to_underlying_arguments(
            args.problem_shape, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(
            args.problem_shape, args.epilogue, workspace)};
  }

  static cutlass::Status
  initialize_workspace(const Arguments&, void*, cudaStream_t, void*) {
    return cutlass::Status::kSuccess;
  }

  static dim3 get_grid_shape(const Params& params) {
    auto [m, n, k, l] = params.problem_shape;
    return dim3{
        uint32_t(ceil_div(m, shape<0>(TileShape{}))),
        uint32_t(ceil_div(n, shape<1>(TileShape{}))),
        uint32_t(l)};
  }

  static dim3 get_block_shape() {
    return dim3{MaxThreadsPerBlock, 1, 1};
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem_buf) {
    int thread_idx = int(threadIdx.x);
    auto [m_coord, n_coord, l_coord] = uint3(blockIdx);

    auto shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto cta_tile = TileShape{};
    auto cta_coord = make_coord(m_coord, n_coord, _, l_coord);

    // Represent the full tensors.
    Tensor mA_mkl = make_tensor(
        make_gmem_ptr(params.mainloop.ptr_A),
        select<0, 2, 3>(shape_MNKL),
        params.mainloop.dA);
    Tensor mB_nkl = make_tensor(
        make_gmem_ptr(params.mainloop.ptr_B),
        select<1, 2, 3>(shape_MNKL),
        params.mainloop.dB);

    // Get batch slice.
    Tensor mA_mk = mA_mkl(_, _, params.lhs_indices[l_coord]);
    Tensor mB_nk = mB_nkl(_, _, params.rhs_indices[l_coord]);

    // Slice to get the tiles this thread block is responsible for.
    Tensor gA =
        local_tile(mA_mk, cta_tile, take<0, 3>(cta_coord), Step<_1, X, _1>{});
    Tensor gB =
        local_tile(mB_nk, cta_tile, take<0, 3>(cta_coord), Step<X, _1, _1>{});

    // Compute tile residues for predication.
    auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * get<0>(cta_coord);
    auto n_max_coord = size<1>(shape_MNKL) - size<0>(gB) * get<1>(cta_coord);
    auto k_residue = size<2>(shape_MNKL) - size<1>(gA) * size<2>(gA);
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) cta_tile.
    TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(cta_tile));
    clear(accum);

    auto k_tile_iter = make_coord_iterator(shape<2>(gA));
    int k_tile_count = size<2>(gA);

    // Perform the collective scoped MMA.
    CollectiveMainloop collective_mma;
    collective_mma(
        accum,
        gA,
        gB,
        accum,
        k_tile_iter,
        k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf);

    // Epilogue and write to out.
    CollectiveEpilogue epilogue(params.epilogue);
    epilogue(
        shape_MNKL,
        cta_tile,
        cta_coord,
        accum,
        tiled_mma,
        residue_mnk,
        thread_idx,
        smem_buf);
  }
};

template <typename Element, bool KMajor>
struct SimtCopyTraits {};

template <typename Element>
struct SimtCopyTraits<Element, true> {
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<Element>, Element>{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _1>>{}));
  using SmemLayout = Layout<Shape<_128, _8>, Stride<_1, Int<128 + 1>>>;
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
};

template <typename Element>
struct SimtCopyTraits<Element, false> {
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<Element>, Element>{},
      Layout<Shape<_32, _8>, Stride<_1, _32>>{},
      Layout<Shape<_1, _1>>{}));
  using SmemLayout = Layout<Shape<_128, _8>, Stride<_1, _128>>;
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
};

template <typename F>
void dispatch_stride(bool k_major, int m, int k, F&& f) {
  if (k_major) {
    f(make_stride(k, Int<1>{}, m * k), std::true_type{});
  } else {
    f(make_stride(Int<1>{}, m, m * k), std::false_type{});
  }
}

template <typename Element, typename F>
void gather_mm(
    int m,
    int n,
    int k,
    int l,
    bool a_transposed,
    bool b_transposed,
    const Element* A,
    const Element* B,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    Element* C,
    F&& launch_kernel) {
  auto problem_shape = make_shape(m, n, k, l);
  auto dC = make_stride(m, Int<1>{}, m * n);
  dispatch_stride(!a_transposed, m, k, [&](auto dA, auto k_major_a) {
    dispatch_stride(b_transposed, n, k, [&](auto dB, auto k_major_b) {
      using Accumulator =
          std::conditional_t<(sizeof(Element) < 4), float, Element>;
      using TileShape = Shape<_128, _128, _8>;
      using DispatchPolicy = cutlass::gemm::MainloopSm70TwoStage;
      using TiledMma = TiledMMA<
          MMA_Atom<UniversalFMA<Accumulator, Element, Element, Element>>,
          Layout<Shape<_16, _16, _1>>>;

      using CopyTraitsA = SimtCopyTraits<Element, k_major_a.value>;
      using CopyTraitsB = SimtCopyTraits<Element, k_major_b.value>;

      using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy,
          TileShape,
          Element,
          decltype(dA),
          Element,
          decltype(dB),
          TiledMma,
          typename CopyTraitsA::GmemTiledCopy,
          typename CopyTraitsA::SmemLayout,
          typename CopyTraitsA::SmemCopyAtom,
          identity,
          typename CopyTraitsB::GmemTiledCopy,
          typename CopyTraitsB::SmemLayout,
          typename CopyTraitsB::SmemCopyAtom,
          identity>;

      using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
          Element,
          decltype(dC),
          decltype(dC),
          cutlass::epilogue::thread::
              LinearCombination<Element, 1, Accumulator, Accumulator>,
          cutlass::gemm::EpilogueDefault>;

      using GemmKernel = GatherGemm<
          decltype(problem_shape),
          CollectiveMainloop,
          CollectiveEpilogue>;
      using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

      Gemm gemm;
      typename Gemm::Arguments args{
          problem_shape,
          lhs_indices,
          rhs_indices,
          {A, dA, B, dB},
          {{1.f, 0.f}, C, dC, C, dC}};

      CHECK_CUTLASS_ERROR(gemm.initialize(args, nullptr));

      auto* kernel = &cutlass::device_kernel<GemmKernel>;
      void* kernel_params[] = {const_cast<Gemm::Params*>(&gemm.params())};
      launch_kernel(
          reinterpret_cast<void*>(kernel),
          gemm.get_grid_shape(gemm.params()),
          GemmKernel::get_block_shape(),
          GemmKernel::SharedStorageSize,
          kernel_params);
    });
  });
}

} // namespace cutlass_gemm

namespace mlx::core {

void cutlass_gather_mm(
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
  if (m < 16 || n < 16) {
    throw std::invalid_argument("[gather_mm] M/N is too small.");
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);

  dispatch_float_types(out.dtype(), "gather_mm", [&](auto type_tag) {
    using Element = cutlass_type_t<MLX_GET_TYPE(type_tag)>;
    cutlass_gemm::gather_mm(
        m,
        n,
        k,
        l,
        a_transposed,
        b_transposed,
        gpu_ptr<Element>(a),
        gpu_ptr<Element>(b),
        gpu_ptr<uint32_t>(lhs_indices),
        gpu_ptr<uint32_t>(rhs_indices),
        gpu_ptr<Element>(out),
        [&](auto* kernel,
            dim3 num_blocks,
            dim3 block_dims,
            uint32_t smem_bytes,
            void** args) {
          encoder.add_kernel_node_raw(
              kernel, num_blocks, block_dims, {}, smem_bytes, args);
        });
  });
}

} // namespace mlx::core
