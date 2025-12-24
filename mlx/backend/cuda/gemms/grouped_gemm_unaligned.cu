// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

using ProblemSize = cutlass::gemm::GemmCoord;

namespace cu {

namespace cg = cooperative_groups;

template <int N_READS>
__global__ void prepare_grouped_mm_data(
    const uint32_t* indices,
    size_t size,
    int group_count,
    int K,
    int N,
    int lda,
    int ldb,
    int item_size,
    int8_t* a_start,
    int8_t* b_start,
    int8_t* out_start,
    int a_batch_stride,
    int b_batch_stride,
    int out_batch_stride,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void** a_ptrs,
    void** b_ptrs,
    void** out_ptrs) {
  auto block = cg::this_thread_block();

  // cumsum(histogram(indices)) - offset for each group.
  extern __shared__ uint32_t cum_histo[];

  int group = block.thread_rank();
  if (group < group_count) {
    cum_histo[group] = 0;
  }

  block.sync();

  // Since |indices| is sorted, the position where element changes would be its
  // cumulative histogram.
  size_t elems_per_block = block.num_threads() * N_READS;
  for (int r = 0; r < cuda::ceil_div(size, elems_per_block); ++r) {
    // TODO: Use vectorized read.
    for (int i = 0; i < N_READS; ++i) {
      size_t pos = r * elems_per_block + group * N_READS + i;
      if (pos >= size) {
        break;
      }
      auto elem = indices[pos];
      auto next = pos < size - 1 ? indices[pos + 1] : group_count;
      while (elem < next) {
        cum_histo[elem] = pos + 1;
        elem++;
      }
    }
  }

  block.sync();

  if (group < group_count) {
    // Fill shapes.
    int delta =
        group == 0 ? cum_histo[0] : cum_histo[group] - cum_histo[group - 1];
    problem_sizes[group] = {delta, N, K};
    a_lds[group] = lda;
    b_lds[group] = ldb;
    out_lds[group] = N;
    // Fill pointers.
    auto offset = group == 0 ? 0 : cum_histo[group - 1];
    a_ptrs[group] = a_start + offset * item_size * a_batch_stride;
    b_ptrs[group] = b_start + group * item_size * b_batch_stride;
    out_ptrs[group] = out_start + offset * item_size * out_batch_stride;
  }
}

} // namespace cu

namespace {

template <typename T, int kAlignment, typename Arch, typename OpClass>
void grouped_gemm_v2(
    bool a_transposed,
    bool b_transposed,
    int group_count,
    ProblemSize* problem_sizes,
    int64_t* a_lds,
    int64_t* b_lds,
    int64_t* out_lds,
    void* a_ptrs,
    void* b_ptrs,
    void* out_ptrs,
    cu::CommandEncoder& encoder) {
  using ElementAccumulator = float;
  using GemmConfiguration = typename cutlass::gemm::device::
      DefaultGemmConfiguration<OpClass, Arch, T, T, T, ElementAccumulator>;
  using EpilogueOutputOp = typename GemmConfiguration::EpilogueOutputOp;

  dispatch_bool(a_transposed, [&](auto a_transposed_tag) {
    dispatch_bool(b_transposed, [&](auto b_transposed_tag) {
      using LayoutA = std::conditional_t<
          a_transposed_tag,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using LayoutB = std::conditional_t<
          b_transposed_tag,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          T,
          LayoutA,
          cutlass::ComplexTransform::kNone,
          kAlignment,
          T,
          LayoutB,
          cutlass::ComplexTransform::kNone,
          kAlignment,
          T,
          cutlass::layout::RowMajor,
          ElementAccumulator,
          OpClass,
          Arch,
          typename GemmConfiguration::ThreadblockShape,
          typename GemmConfiguration::WarpShape,
          typename GemmConfiguration::InstructionShape,
          EpilogueOutputOp,
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
          GemmConfiguration::kStages>::GemmKernel;
      using GemmGrouped =
          typename cutlass::gemm::device::GemmGrouped<GemmKernel>;

      typename EpilogueOutputOp::Params epilogue_op(
          /* alpha */ 1, /* beta */ 0);
      typename GemmGrouped::Arguments args(
          problem_sizes,
          group_count,
          GemmGrouped::sufficient(),
          epilogue_op,
          reinterpret_cast<T**>(a_ptrs),
          reinterpret_cast<T**>(b_ptrs),
          reinterpret_cast<T**>(out_ptrs),
          reinterpret_cast<T**>(out_ptrs),
          a_lds,
          b_lds,
          out_lds,
          out_lds);

      GemmGrouped gemm;
      cutlass::Status status = gemm.initialize(args, nullptr, encoder.stream());
      if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(fmt::format(
            "Failed to initialize GemmGrouped: {}",
            cutlass::cutlassGetStatusString(status)));
      }

      auto capture = encoder.capture_context();
      status = gemm.run(encoder.stream());
      if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(fmt::format(
            "Failed to run GemmGrouped: {}",
            cutlass::cutlassGetStatusString(status)));
      }
    });
  });
}

} // namespace

void cutlass_grouped_gemm_unaligned(
    bool a_transposed,
    int lda,
    bool b_transposed,
    int ldb,
    int group_count,
    const array& a,
    const array& b,
    const array& indices,
    array& out,
    cu::CommandEncoder& encoder) {
  // Prepare device pointers for matmul.
  int problem_sizes_nbytes =
      group_count * cuda::ceil_div(sizeof(ProblemSize), 8) * 8;
  int nbytes = problem_sizes_nbytes +
      group_count * (3 * sizeof(void*) + 3 * sizeof(int64_t));
  nbytes = cuda::ceil_div(nbytes, 256) * 256;
  array gemm_args(cu::malloc_async(nbytes, encoder), {nbytes}, int8);
  encoder.add_temporary(gemm_args);

  ProblemSize* problem_sizes = gpu_ptr<ProblemSize>(gemm_args);
  int64_t* a_lds = gpu_ptr<int64_t>(gemm_args) + problem_sizes_nbytes / 8;
  int64_t* b_lds = a_lds + group_count;
  int64_t* out_lds = b_lds + group_count;
  void** a_ptrs = reinterpret_cast<void**>(out_lds + group_count);
  void** b_ptrs = a_ptrs + group_count;
  void** out_ptrs = b_ptrs + group_count;

  // Fill the pointers by computing offsets from indices.
  constexpr int N_READS = 4;
  size_t n_threads = cuda::ceil_div(indices.size(), N_READS);
  n_threads = group_count < n_threads ? n_threads : group_count;
  dim3 block_dims(std::min(n_threads, 1024ul));
  dim3 num_blocks(1);

  encoder.set_input_array(indices);
  encoder.set_output_array(gemm_args);
  encoder.add_kernel_node(
      cu::prepare_grouped_mm_data<N_READS>,
      num_blocks,
      block_dims,
      group_count * sizeof(uint32_t), // sizeof(cum_histo)
      gpu_ptr<uint32_t>(indices),
      indices.size(),
      group_count,
      a.shape(-1), // K
      b.shape(-1), // N,
      lda,
      ldb,
      out.itemsize(),
      gpu_ptr<int8_t>(a),
      gpu_ptr<int8_t>(b),
      gpu_ptr<int8_t>(out),
      a.shape(-2) * a.shape(-1), // a_batch_stride
      b.shape(-2) * b.shape(-1), // b_batch_stride
      out.shape(-2) * out.shape(-1), // out_batch_stride
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs);

  // Invoke grouped GEMM.
  constexpr int kAlignment = 1;
  using Arch = cutlass::arch::Sm75;
  using OpClass = cutlass::arch::OpClassSimt;
  auto* fun = grouped_gemm_v2<float, kAlignment, Arch, OpClass>;
  switch (a.dtype()) {
    case float32:
      break;
    case float16:
      fun = grouped_gemm_v2<cutlass::half_t, kAlignment, Arch, OpClass>;
      break;
    case bfloat16:
      fun = grouped_gemm_v2<cutlass::bfloat16_t, kAlignment, Arch, OpClass>;
      break;
    default:
      throw std::runtime_error(fmt::format(
          "Unsupported dtype in cutlass_grouped_gemm_sm75: {}.",
          dtype_to_string(a.dtype())));
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(gemm_args);
  encoder.set_output_array(out);
  fun(a_transposed,
      b_transposed,
      group_count,
      problem_sizes,
      a_lds,
      b_lds,
      out_lds,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      encoder);
}

} // namespace mlx::core
