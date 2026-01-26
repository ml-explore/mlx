// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cutlass_utils.cuh"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
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

// Shared GEMM configuration for every type and arch.
template <typename T, typename ArchTag, int kAlignmentC>
struct CommonGemmConfiguration {
  using Element = T;
  using Arch = ArchTag;
  using Accumulator = std::conditional_t<(sizeof(T) < 4), float, T>;
  using EpilogueOutputOp = cutlass::epilogue::thread::
      LinearCombination<T, kAlignmentC, Accumulator, Accumulator>;
};

// Slow GEMM configuration as fallback.
template <
    typename T,
    typename Arch,
    int kAlignmentC = 1,
    bool kEnableTF32 = false,
    typename Enable = void>
struct GemmConfiguration : public CommonGemmConfiguration<T, Arch, 1> {
  using OpClass = cutlass::arch::OpClassSimt;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static const int kAlignmentAB = 1;
  static const int kStages = 2;
};

// Specialized GEMM configuration for sm80 and later.
template <typename T, typename Arch, int kAlignmentC, bool kEnableTF32>
struct GemmConfiguration<
    T,
    Arch,
    kAlignmentC,
    kEnableTF32,
    std::enable_if_t<Arch::kMinComputeCapability >= 80 && sizeof(T) <= 4>>
    : public CommonGemmConfiguration<T, cutlass::arch::Sm80, kAlignmentC> {
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32 / sizeof(T)>;
  static const int kAlignmentAB = 1;
  static const int kStages = 2;
};

// Specialized GEMM configuration for tf32 on sm80.
template <int kAlignmentC>
struct GemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC, true>
    : public CommonGemmConfiguration<float, cutlass::arch::Sm80, kAlignmentC> {
  using OpClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  static const int kAlignmentAB = 1;
  static const int kStages = 3; // use SM80_CP_ASYNC
};

// Get direct access to kernel.
template <typename GemmKernel>
class GemmGroupedEncoder
    : public cutlass::gemm::device::GemmGrouped<GemmKernel> {
 public:
  void encode(cu::CommandEncoder& encoder) {
    encoder.add_kernel_node(
        cutlass::Kernel<GemmKernel>,
        {static_cast<uint32_t>(this->params_.threadblock_count), 1, 1},
        {GemmKernel::kThreadCount, 1, 1},
        sizeof(typename GemmKernel::SharedStorage),
        this->params_);
  }
};

// Invoke the grouped GEMM of CUTLASS 2.x API, which supports small alignments.
template <typename GemmConfiguration>
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
  dispatch_bool(a_transposed, [&](auto a_transposed_tag) {
    dispatch_bool(b_transposed, [&](auto b_transposed_tag) {
      using LayoutA = std::conditional_t<
          decltype(a_transposed_tag)::value,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using LayoutB = std::conditional_t<
          decltype(b_transposed_tag)::value,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>;
      using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
          typename GemmConfiguration::Element,
          LayoutA,
          cutlass::ComplexTransform::kNone,
          GemmConfiguration::kAlignmentAB,
          typename GemmConfiguration::Element,
          LayoutB,
          cutlass::ComplexTransform::kNone,
          GemmConfiguration::kAlignmentAB,
          typename GemmConfiguration::Element,
          cutlass::layout::RowMajor,
          typename GemmConfiguration::Accumulator,
          typename GemmConfiguration::OpClass,
          typename GemmConfiguration::Arch,
          typename GemmConfiguration::ThreadblockShape,
          typename GemmConfiguration::WarpShape,
          typename GemmConfiguration::InstructionShape,
          typename GemmConfiguration::EpilogueOutputOp,
          cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
          GemmConfiguration::kStages>::GemmKernel;
      using GemmGrouped = GemmGroupedEncoder<GemmKernel>;

      static int threadblock_count = GemmGrouped::sufficient();
      typename GemmGrouped::Arguments args(
          problem_sizes,
          group_count,
          threadblock_count,
          {/* alpha */ 1, /* beta */ 0},
          reinterpret_cast<typename GemmGrouped::ElementA**>(a_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementB**>(b_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementC**>(out_ptrs),
          reinterpret_cast<typename GemmGrouped::ElementC**>(out_ptrs),
          a_lds,
          b_lds,
          out_lds,
          out_lds);

      GemmGrouped gemm;
      CHECK_CUTLASS_ERROR(gemm.initialize(
          args,
          allocate_workspace(encoder, gemm.get_workspace_size(args)),
          encoder.stream()));
      gemm.encode(encoder);
    });
  });
}

template <typename F>
void dispatch_cutlass_arch(cu::Device& device, F&& f) {
  if (device.compute_capability_major() < 8) {
    f(type_identity<cutlass::arch::Sm75>{});
  } else if (device.compute_capability_major() == 8) {
    f(type_identity<cutlass::arch::Sm80>{});
  } else {
    f(type_identity<cutlass::arch::Sm90>{});
  }
}

auto* get_grouped_mm_funcion(Dtype dtype, int N, cu::Device& device) {
  auto* fun = grouped_gemm_v2<GemmConfiguration<float, cutlass::arch::Sm75>>;
  dispatch_float_types(dtype, "grouped_gemm_v2", [&](auto type_tag) {
    using DataType = cutlass_type_t<MLX_GET_TYPE(type_tag)>;
    dispatch_cutlass_arch(device, [&](auto arch_tag) {
      using Arch = MLX_GET_TYPE(arch_tag);
      dispatch_bool(N % 8 == 0, [&](auto is_out_aligned) {
        dispatch_bool(env::enable_tf32(), [&](auto enable_tf32) {
          fun = grouped_gemm_v2<GemmConfiguration<
              DataType,
              Arch,
              is_out_aligned.value ? 8 : 1,
              enable_tf32.value>>;
        });
      });
    });
  });
  return fun;
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
  int K = a.shape(-1);
  int N = b.shape(-1);

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
  int n_threads = cuda::ceil_div(indices.size(), N_READS);
  n_threads = group_count < n_threads ? n_threads : group_count;
  dim3 block_dims(std::min(n_threads, 1024));
  dim3 num_blocks(1);

  encoder.set_input_array(indices);
  encoder.set_output_array(gemm_args);
  auto kernel = cu::prepare_grouped_mm_data<N_READS>;
  // Store params in variables to ensure they remain valid
  const uint32_t* indices_ptr = gpu_ptr<uint32_t>(indices);
  size_t size_val = indices.size();
  int group_count_val = group_count;
  int K_val = a.shape(-1);
  int N_val = b.shape(-1);
  int lda_val = lda;
  int ldb_val = ldb;
  int item_size_val = out.itemsize();
  int8_t* a_ptr = const_cast<int8_t*>(gpu_ptr<int8_t>(a));
  int8_t* b_ptr = const_cast<int8_t*>(gpu_ptr<int8_t>(b));
  int8_t* out_ptr_val = gpu_ptr<int8_t>(out);
  int a_batch_stride_val = a.shape(-2) * a.shape(-1);
  int b_batch_stride_val = b.shape(-2) * b.shape(-1);
  int out_batch_stride_val = out.shape(-2) * out.shape(-1);
  ProblemSize* problem_sizes_ptr = problem_sizes;
  int64_t* a_lds_ptr = a_lds;
  int64_t* b_lds_ptr = b_lds;
  int64_t* out_lds_ptr = out_lds;
  void** a_ptrs_ptr = a_ptrs;
  void** b_ptrs_ptr = b_ptrs;
  void** out_ptrs_ptr = out_ptrs;
  void* params[] = {
      &indices_ptr,
      &size_val,
      &group_count_val,
      &K_val,
      &N_val,
      &lda_val,
      &ldb_val,
      &item_size_val,
      &a_ptr,
      &b_ptr,
      &out_ptr_val,
      &a_batch_stride_val,
      &b_batch_stride_val,
      &out_batch_stride_val,
      &problem_sizes_ptr,
      &a_lds_ptr,
      &b_lds_ptr,
      &out_lds_ptr,
      &a_ptrs_ptr,
      &b_ptrs_ptr,
      &out_ptrs_ptr};
  encoder.add_kernel_node(
      reinterpret_cast<void*>(kernel),
      num_blocks,
      block_dims,
      group_count * sizeof(uint32_t), // sizeof(cum_histo)
      static_cast<void**>(params));

  // Invoke grouped GEMM.
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(gemm_args);
  encoder.set_output_array(out);
  auto* fun = get_grouped_mm_funcion(a.dtype(), N, encoder.device());
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
