// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/mps/gemm.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// MPS Matmul fallback
///////////////////////////////////////////////////////////////////////////////

namespace {

bool use_mps() {
  auto get_val = []() {
    if (const char* buff_str = std::getenv("MLX_USE_MPS")) {
      return std::string(buff_str) != "OFF";
    } else {
      return false;
    }
  };
  static bool use_mps_ = get_val();
  return use_mps_;
}

#define MAX_OPS_PER_BUFFER max_ops_per_buffer()

inline void mps_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    float alpha = 1.0f,
    float beta = 0.0f) {
  MPS::DataType mps_dtype = MPS::DataTypeFloat32;

  if (out.dtype() == float16) {
    mps_dtype = MPS::DataTypeFloat16;
  } else if (out.dtype() == bfloat16) {
    mps_dtype = MPS::DataTypeBFloat16;
  }

  // Used batched MPSMatrixMultiplication if batch_size_out > 1
  // We only accept the following cases:
  //  1. Both a, b have batch_size_out matrices worth of data
  //  2. Only one of a or b has batch_size_out matrices worth of data and
  //     the other has matrix worth of data

  // The matrix dimensions of a and b are sure to be regularly strided
  if (batch_size_out > 1) {
    // No broadcasting defaults
    auto batch_size_a = a.data_size() / (M * K);
    auto batch_size_b = b.data_size() / (K * N);

    auto matrix_stride_a = M * K;
    auto matrix_stride_b = K * N;
    auto matrix_stride_out = M * N;

    // At this point, batch_size_a, batch_size_b show the number of matrices
    //    in data, no broadcasted strides considered
    if (batch_size_out == std::max(batch_size_a, batch_size_b)) {
      // Handle simple broadcasting
      if (std::min(batch_size_a, batch_size_b) == 1) {
        matrix_stride_a = (batch_size_a == 1) ? 0 : matrix_stride_a;
        matrix_stride_b = (batch_size_b == 1) ? 0 : matrix_stride_b;

        batch_size_a = batch_size_out;
        batch_size_b = batch_size_out;
      }

      // Only proceed if broadcasting between a and b is simple
      // At this point, batch_size_a, batch_size_b show the number of matrices
      //    after broadcasting
      if (batch_size_a == batch_size_b) {
        auto a_desc = MPS::MatrixDescriptor::matrixDescriptor(
            (M * K) / lda,
            lda,
            batch_size_a,
            lda * a.itemsize(),
            (matrix_stride_a * a.itemsize()),
            mps_dtype);

        auto b_desc = MPS::MatrixDescriptor::matrixDescriptor(
            (K * N) / ldb,
            ldb,
            batch_size_b,
            ldb * b.itemsize(),
            (matrix_stride_b * b.itemsize()),
            mps_dtype);

        auto out_desc = MPS::MatrixDescriptor::matrixDescriptor(
            M,
            N,
            batch_size_out,
            N * out.itemsize(),
            matrix_stride_out * out.itemsize(),
            mps_dtype);

        auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
        auto a_mat = MPS::Matrix::alloc()->init(a_buf, a_desc);

        auto b_buf = static_cast<const MTL::Buffer*>(b.buffer().ptr());
        auto b_mat = MPS::Matrix::alloc()->init(b_buf, b_desc);

        auto out_buf = static_cast<MTL::Buffer*>(out.buffer().ptr());
        auto out_mat = MPS::Matrix::alloc()->init(out_buf, out_desc);

        auto kernel = MPS::MatrixMultiplication::alloc()->init(
            d.mtl_device(), transpose_a, transpose_b, M, N, K, alpha, beta);

        auto command_buffer = d.get_command_buffer(s.index);
        kernel->setBatchSize(batch_size_out);
        kernel->setBatchStart(0);
        kernel->encodeToCommandBuffer(command_buffer, a_mat, b_mat, out_mat);
        command_buffer->addCompletedHandler(
            [a_mat, b_mat, out_mat, kernel, copies](
                MTL::CommandBuffer*) mutable {
              a_mat->release();
              b_mat->release();
              out_mat->release();
              kernel->release();
              copies.clear();
            });

        return;
      }
    }
  }

  // Schedule as many calls to MPSMatrixMultiplication as needed otherwise
  auto a_desc = MPS::MatrixDescriptor::matrixDescriptor(
      a.data_size() / lda, lda, lda * a.itemsize(), mps_dtype);

  auto b_desc = MPS::MatrixDescriptor::matrixDescriptor(
      b.data_size() / ldb, ldb, ldb * b.itemsize(), mps_dtype);

  auto out_desc = MPS::MatrixDescriptor::matrixDescriptor(
      batch_size_out * M, N, N * out.itemsize(), mps_dtype);

  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  auto a_mat = MPS::Matrix::alloc()->init(a_buf, a_desc);

  auto b_buf = static_cast<const MTL::Buffer*>(b.buffer().ptr());
  auto b_mat = MPS::Matrix::alloc()->init(b_buf, b_desc);

  auto out_buf = static_cast<MTL::Buffer*>(out.buffer().ptr());
  auto out_mat = MPS::Matrix::alloc()->init(out_buf, out_desc);

  auto kernel = MPS::MatrixMultiplication::alloc()->init(
      d.mtl_device(), transpose_a, transpose_b, M, N, K, alpha, beta);

  auto command_buffer = d.get_command_buffer(s.index);
  for (int i = 0; i < batch_size_out; ++i) {
    auto a_row = elem_to_loc(M * K * i, a.shape(), a.strides()) / lda;
    auto b_row = elem_to_loc(K * N * i, b.shape(), b.strides()) / ldb;
    kernel->setLeftMatrixOrigin({a_row, 0, 0});
    kernel->setRightMatrixOrigin({b_row, 0, 0});
    kernel->setResultMatrixOrigin({i * static_cast<size_t>(M), 0, 0});
    kernel->encodeToCommandBuffer(command_buffer, a_mat, b_mat, out_mat);
  }

  command_buffer->addCompletedHandler(
      [a_mat, b_mat, out_mat, kernel, copies](MTL::CommandBuffer*) mutable {
        a_mat->release();
        b_mat->release();
        out_mat->release();
        kernel->release();
        copies.clear();
      });
}

inline auto collapse_batches(const array& a, const array& b) {
  // Get and check the shape for the batched dims
  std::vector<int> A_bshape{a.shape().begin(), a.shape().end() - 2};
  std::vector<int> B_bshape{b.shape().begin(), b.shape().end() - 2};
  if (A_bshape != B_bshape) {
    std::ostringstream msg;
    msg << "[matmul] Got matrices with incorrectly broadcasted shapes: " << "A "
        << a.shape() << ", B " << b.shape() << ".";
    throw std::runtime_error(msg.str());
  }

  std::vector<size_t> A_bstride{a.strides().begin(), a.strides().end() - 2};
  std::vector<size_t> B_bstride{b.strides().begin(), b.strides().end() - 2};

  auto [batch_shape, batch_strides] =
      collapse_contiguous_dims(A_bshape, std::vector{A_bstride, B_bstride});

  auto A_batch_stride = batch_strides[0];
  auto B_batch_stride = batch_strides[1];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    A_batch_stride.push_back(0);
    B_batch_stride.push_back(0);
  }

  return std::make_tuple(batch_shape, A_batch_stride, B_batch_stride);
}

inline auto collapse_batches(const array& a, const array& b, const array& c) {
  // Get and check the shape for the batched dims
  std::vector<int> A_bshape{a.shape().begin(), a.shape().end() - 2};
  std::vector<int> B_bshape{b.shape().begin(), b.shape().end() - 2};
  std::vector<int> C_bshape{c.shape().begin(), c.shape().end() - 2};
  if (A_bshape != B_bshape || A_bshape != C_bshape) {
    std::ostringstream msg;
    msg << "[addmm] Got matrices with incorrectly broadcasted shapes: " << "A "
        << a.shape() << ", B " << b.shape() << ", B " << c.shape() << ".";
    throw std::runtime_error(msg.str());
  }

  std::vector<size_t> A_bstride{a.strides().begin(), a.strides().end() - 2};
  std::vector<size_t> B_bstride{b.strides().begin(), b.strides().end() - 2};
  std::vector<size_t> C_bstride{c.strides().begin(), c.strides().end() - 2};

  auto [batch_shape, batch_strides] = collapse_contiguous_dims(
      A_bshape, std::vector{A_bstride, B_bstride, C_bstride});

  auto A_batch_stride = batch_strides[0];
  auto B_batch_stride = batch_strides[1];
  auto C_batch_stride = batch_strides[2];

  if (batch_shape.empty()) {
    batch_shape.push_back(1);
    A_batch_stride.push_back(0);
    B_batch_stride.push_back(0);
    C_batch_stride.push_back(0);
  }

  return std::make_tuple(
      batch_shape, A_batch_stride, B_batch_stride, C_batch_stride);
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Steel matmul fallback
///////////////////////////////////////////////////////////////////////////////

void steel_matmul_conv_groups(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    int groups,
    std::vector<array>& copies) {
  using namespace mlx::steel;

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch

  // Determine dispatch kernel
  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  if ((size_t)M * N >= 1ul << 20) {
    if (!transpose_a && transpose_b) {
      bm = 64;
      bn = (out.dtype() == float32) ? 64 : 32;
      bk = (out.dtype() == float32) ? 16 : 32;
    } else {
      bm = 64;
      bn = 64;
    }
  }

  // Prepare kernel name
  std::ostringstream kname;
  kname << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn;

  std::string base_name = kname.str();

  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool do_gather = false;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
      {&do_gather, MTL::DataType::DataTypeBool, 300},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_do_gather_" << (do_gather ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn);

  compute_encoder->setComputePipelineState(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ ldd,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ size_t(K),
      /* const size_t batch_stride_b = */ size_t(N) * K,
      /* const size_t batch_stride_d = */ size_t(N),
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ 1};

  // Prepare launch grid params
  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, groups);

  std::vector<int> batch_shape = {1};
  std::vector<size_t> batch_strides = {0};

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);
  compute_encoder->setBytes(&params, sizeof(GEMMParams), 4);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  // Clear copies
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

void steel_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    std::vector<int> batch_shape /* = {} */,
    std::vector<size_t> A_batch_stride /* = {} */,
    std::vector<size_t> B_batch_stride /* = {} */) {
  using namespace mlx::steel;

  if (batch_shape.empty()) {
    /////////////////////////////////////////////////////////////////////////////
    // Check and collapse batch dimensions
    auto [batch_shape_, A_bstride_, B_bstride_] = collapse_batches(a, b);

    batch_shape = batch_shape_;
    A_batch_stride = A_bstride_;
    B_batch_stride = B_bstride_;
    // Collapse batches into M if needed
    if (batch_size_out > 1 && !transpose_a && batch_shape.size() == 1 &&
        a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
        B_batch_stride.back() == 0) {
      M *= batch_shape.back();
      batch_size_out = 1;

      A_batch_stride = {0};
      B_batch_stride = {0};
      batch_shape = {1};
    }
  }

  size_t matrix_stride_out = size_t(M) * N;

  /////////////////////////////////////////////////////////////////////////////
  // Split K specialization

  int _tm = M / 16;
  int _tn = N / 16;
  int _tk = K / 16;

  if (batch_size_out == 1 && (_tm * _tn) <= 32 && _tk >= 8) {
    int bm = M < 40 ? 16 : 32;
    int bn = N < 40 ? 16 : 32;
    int bk = 16;
    int wm = 2, wn = 2;

    int split_k_partitions =
        _tk < 16 ? 2 : (_tk < 32 ? 4 : (_tk < 64 ? 8 : 16));
    int split_k_partition_stride = M * N;
    int gemm_k_iterations = (K / bk) / split_k_partitions;
    int split_k_partition_size = gemm_k_iterations * bk;

    array C_split({split_k_partitions, M, N}, float32, nullptr, {});
    C_split.set_data(allocator::malloc_or_wait(C_split.nbytes()));
    copies.push_back(C_split);

    bool mn_aligned = M % bm == 0 && N % bn == 0;
    bool k_aligned = K % bk == 0;
    std::ostringstream kname;
    kname << "steel_gemm_splitk_" << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
          << type_to_name(C_split) << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn << "_MN_" << (mn_aligned ? "t" : "n")
          << "aligned" << "_K_" << (k_aligned ? "t" : "n") << "aligned";

    // Encode and dispatch gemm kernel
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = get_steel_gemm_splitk_kernel(
        d,
        kname.str(),
        a,
        C_split,
        transpose_a,
        transpose_b,
        bm,
        bn,
        bk,
        wm,
        wn,
        mn_aligned,
        k_aligned);
    compute_encoder->setComputePipelineState(kernel);

    int tn = (N + bn - 1) / bn;
    int tm = (M + bm - 1) / bm;

    GEMMSpiltKParams params{
        /* const int M = */ M,
        /* const int N = */ N,
        /* const int K = */ K,
        /* const int lda = */ lda,
        /* const int ldb = */ ldb,
        /* const int ldc = */ N,
        /* const int tiles_n = */ tn,
        /* const int tiles_m = */ tm,
        /* const int split_k_partitions = */ split_k_partitions,
        /* const int split_k_partition_stride = */ split_k_partition_stride,
        /* const int split_k_partition_size = */ split_k_partition_size,
        /* const int gemm_k_iterations_aligned = */ gemm_k_iterations};

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(tn, tm, split_k_partitions);

    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(b, 1);
    compute_encoder.set_output_array(C_split, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMSpiltKParams), 3);
    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    // Do accum kernel
    {
      auto c_split_buf =
          static_cast<const MTL::Resource*>(C_split.buffer().ptr());
      const class MTL::Resource* const resources[1] = {c_split_buf};
      compute_encoder->memoryBarrier(resources, 1);
      auto kernel_name = "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
          type_to_name(C_split);

      auto kernel = get_steel_gemm_splitk_accum_kernel(
          d, kernel_name, C_split, out, false);
      compute_encoder->setComputePipelineState(kernel);

      // Set the arguments for the kernel
      compute_encoder.set_input_array(C_split, 0);
      compute_encoder.set_output_array(out, 1);
      compute_encoder->setBytes(&split_k_partitions, sizeof(int), 2);
      compute_encoder->setBytes(&split_k_partition_stride, sizeof(int), 3);
      compute_encoder->setBytes(&N, sizeof(int), 4);

      // Launch enough thread groups for each output
      MTL::Size grid_dims = MTL::Size(N, M, 1);
      MTL::Size group_dims = MTL::Size(std::min(1024, N * M), 1, 1);

      compute_encoder.dispatchThreads(grid_dims, group_dims);
    }

    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch

  // Determine dispatch kernel
  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  if ((size_t)batch_size_out * M * N >= 1ul << 20) {
    if (!transpose_a && transpose_b) {
      bm = 64;
      bn = (out.dtype() == float32) ? 64 : 32;
      bk = (out.dtype() == float32) ? 16 : 32;
    } else {
      bm = 64;
      bn = 64;
    }
  }

  // Prepare kernel name
  std::ostringstream kname;
  kname << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn;

  std::string base_name = kname.str();

  const bool has_batch = (batch_shape.size() > 1);
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool do_gather = false;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
      {&do_gather, MTL::DataType::DataTypeBool, 300},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_do_gather_" << (do_gather ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn);

  compute_encoder->setComputePipelineState(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ A_batch_stride.back(),
      /* const size_t batch_stride_b = */ B_batch_stride.back(),
      /* const size_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

  // Prepare launch grid params
  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  std::vector<size_t> batch_strides = A_batch_stride;
  batch_strides.insert(
      batch_strides.end(), B_batch_stride.begin(), B_batch_stride.end());

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&params, sizeof(GEMMParams), 4);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  // Clear copies
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    copy_gpu(zero, out, CopyType::Scalar, s);
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler([zero](MTL::CommandBuffer*) {});
    return;
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr, bool is_vector) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, a_cols, a] = check_transpose(a_pre, M == 1);
  auto [b_transposed, b_cols, b] = check_transpose(b_pre, N == 1);

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, A_batch_stride, B_batch_stride] = collapse_batches(a, b);

  auto batch_size_out = out.size() / (size_t(M) * size_t(N));

  // Collapse batches into M if needed
  if (batch_size_out > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
      B_batch_stride.back() == 0) {
    M *= batch_shape.back();
    batch_size_out = 1;

    A_batch_stride = {0};
    B_batch_stride = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Gemv specialization

  // Route to gemv if needed
  if (std::min(M, N) == 1) {
    // Collect problem info
    bool is_b_matrix = N != 1;

    auto& mat = is_b_matrix ? b : a;
    auto& vec = is_b_matrix ? a : b;
    bool transpose_mat = is_b_matrix ? !b_transposed : a_transposed;
    int in_vector_len = K;
    int out_vector_len = is_b_matrix ? N : M;

    int mat_cols = transpose_mat ? out_vector_len : in_vector_len;
    int mat_rows = transpose_mat ? in_vector_len : out_vector_len;
    int mat_ld = is_b_matrix ? b_cols : a_cols;

    auto batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
    auto batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;

    int stride_mat = batch_strides_mat.back();
    int stride_vec = batch_strides_vec.back();

    // Determine if inputs have simple batching / broadcasting
    bool contiguous_kernel = (batch_shape.size() == 1);

    int batch_ndim = batch_shape.size();

    // Determine dispatch kernel
    int tm = 4, tn = 4;
    int sm = 1, sn = 32;
    int bm = 1, bn = 1;
    int n_out_per_tgp;
    std::ostringstream kname;

    if (transpose_mat) {
      if (in_vector_len >= 8192 && out_vector_len >= 2048) {
        sm = 4;
        sn = 8;
      } else {
        sm = 8;
        sn = 4;
      }

      if (out_vector_len >= 2048) {
        bn = 16;
      } else if (out_vector_len >= 512) {
        bn = 4;
      } else {
        bn = 2;
      }

      // Specialized kernel for very small outputs
      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * sn * tn;
      kname << "gemv_t_" << type_to_name(out);

    } else {
      bm = out_vector_len >= 4096 ? 8 : 4;
      sn = 32;

      // Specialized kernel for very small outputs
      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * sm * tm;
      kname << "gemv_" << type_to_name(out);
    }

    kname << "_bm" << bm << "_bn" << bn << "_sm" << sm << "_sn" << sn << "_tm"
          << tm << "_tn" << tn;
    kname << "_nc" << !contiguous_kernel << "_axpby0";

    // Encode and dispatch kernel
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(32, bn, bm);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    compute_encoder.set_input_array(mat, 0);
    compute_encoder.set_input_array(vec, 1);
    compute_encoder.set_output_array(out, 3);

    compute_encoder->setBytes(&in_vector_len, sizeof(int), 4);
    compute_encoder->setBytes(&out_vector_len, sizeof(int), 5);
    compute_encoder->setBytes(&mat_ld, sizeof(int), 6);

    compute_encoder->setBytes(&batch_ndim, sizeof(int), 9);
    compute_encoder->setBytes(batch_shape.data(), batch_ndim * sizeof(int), 10);
    compute_encoder->setBytes(
        batch_strides_vec.data(), batch_ndim * sizeof(size_t), 11);
    compute_encoder->setBytes(
        batch_strides_mat.data(), batch_ndim * sizeof(size_t), 12);

    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Gemm specialization

  if (use_mps()) {
    d.end_encoding(s.index);

    return mps_matmul(
        s,
        d,
        a,
        b,
        out,
        M,
        N,
        K,
        batch_size_out,
        a_cols,
        b_cols,
        a_transposed,
        b_transposed,
        copies);
  }

  return steel_matmul(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ a_cols,
      /* int ldb = */ b_cols,
      /* bool transpose_a = */ a_transposed,
      /* bool transpose_b = */ b_transposed,
      /* std::vector<array>& = */ copies,
      /* std::vector<int> batch_shape = */ batch_shape,
      /* std::vector<size_t> A_batch_stride = */ A_batch_stride,
      /* std::vector<size_t> B_batch_stride = */ B_batch_stride);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto& c_pre = inputs[2];

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr, bool is_vector) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [transpose_a, a_cols, a] = check_transpose(a_pre, M == 1);
  auto [transpose_b, b_cols, b] = check_transpose(b_pre, N == 1);

  array c = c_pre;
  int ldc = c.strides()[c.ndim() - 2];
  int fdc = c.strides()[c.ndim() - 1];

  int lda = a_cols;
  int ldb = b_cols;
  int ldd = N;

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions
  auto [batch_shape, A_batch_stride, B_batch_stride, C_batch_stride] =
      collapse_batches(a, b, c);

  size_t matrix_stride_out = size_t(M) * size_t(N);
  auto batch_size_out = out.size() / (matrix_stride_out);

  // Collapse batches into M if needed
  if (batch_size_out > 1 && !transpose_a && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
      C_batch_stride.back() == M * c.strides()[c.ndim() - 2] &&
      B_batch_stride.back() == 0) {
    M *= batch_shape.back();
    batch_size_out = 1;

    A_batch_stride = {0};
    B_batch_stride = {0};
    C_batch_stride = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Gemv specialization

  // Route to gemv if needed
  if (std::min(M, N) == 1) {
    // Collect problem info
    bool is_b_matrix = N != 1;

    auto& mat = is_b_matrix ? b : a;
    auto& vec = is_b_matrix ? a : b;
    bool transpose_mat = is_b_matrix ? !transpose_b : transpose_a;
    int in_vector_len = K;
    int out_vector_len = is_b_matrix ? N : M;

    int mat_cols = transpose_mat ? out_vector_len : in_vector_len;
    int mat_rows = transpose_mat ? in_vector_len : out_vector_len;
    int mat_ld = is_b_matrix ? b_cols : a_cols;

    auto batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
    auto batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;

    int stride_mat = batch_strides_mat.back();
    int stride_vec = batch_strides_vec.back();

    // Determine if inputs have simple batching / broadcasting
    bool contiguous_kernel = (batch_shape.size() == 1);

    int batch_ndim = batch_shape.size();

    // Determine dispatch kernel
    int tm = 4, tn = 4;
    int sm = 1, sn = 32;
    int bm = 1, bn = 1;
    int n_out_per_tgp;
    std::ostringstream kname;

    if (transpose_mat) {
      if (in_vector_len >= 8192 && out_vector_len >= 2048) {
        sm = 4;
        sn = 8;
      } else {
        sm = 8;
        sn = 4;
      }

      if (out_vector_len >= 2048) {
        bn = 16;
      } else if (out_vector_len >= 512) {
        bn = 4;
      } else {
        bn = 2;
      }

      // Specialized kernel for very small outputs
      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * sn * tn;
      kname << "gemv_t_" << type_to_name(out);

    } else {
      bm = out_vector_len >= 4096 ? 8 : 4;
      sn = 32;

      // Specialized kernel for very small outputs
      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * sm * tm;
      kname << "gemv_" << type_to_name(out);
    }

    kname << "_bm" << bm << "_bn" << bn << "_sm" << sm << "_sn" << sn << "_tm"
          << tm << "_tn" << tn;
    kname << "_nc" << !contiguous_kernel << "_axpby1";

    // Encode and dispatch kernel
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(32, bn, bm);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    compute_encoder.set_input_array(mat, 0);
    compute_encoder.set_input_array(vec, 1);
    compute_encoder.set_input_array(c, 2);
    compute_encoder.set_output_array(out, 3);

    compute_encoder->setBytes(&in_vector_len, sizeof(int), 4);
    compute_encoder->setBytes(&out_vector_len, sizeof(int), 5);
    compute_encoder->setBytes(&mat_ld, sizeof(int), 6);

    compute_encoder->setBytes(&alpha_, sizeof(float), 7);
    compute_encoder->setBytes(&beta_, sizeof(float), 8);

    compute_encoder->setBytes(&batch_ndim, sizeof(int), 9);
    set_vector_bytes(compute_encoder, batch_shape, 10);
    set_vector_bytes(compute_encoder, batch_strides_vec, 11);
    set_vector_bytes(compute_encoder, batch_strides_mat, 12);
    set_vector_bytes(compute_encoder, C_batch_stride, 13);

    int bias_stride = c.strides()[c.ndim() - 1];
    compute_encoder->setBytes(&bias_stride, sizeof(int), 14);

    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }

  using namespace mlx::steel;

  /////////////////////////////////////////////////////////////////////////////
  // Split K specialization

  int _tm = M / 16;
  int _tn = N / 16;
  int _tk = K / 16;

  if (batch_size_out == 1 && (_tm * _tn) <= 32 && _tk >= 8) {
    int bm = M < 40 ? 16 : 32;
    int bn = N < 40 ? 16 : 32;
    int bk = 16;
    int wm = 2, wn = 2;

    int split_k_partitions =
        _tk < 16 ? 2 : (_tk < 32 ? 4 : (_tk < 64 ? 8 : 16));
    int split_k_partition_stride = M * N;
    int gemm_k_iterations = (K / bk) / split_k_partitions;
    int split_k_partition_size = gemm_k_iterations * bk;

    array C_split({split_k_partitions, M, N}, float32, nullptr, {});
    C_split.set_data(allocator::malloc_or_wait(C_split.nbytes()));
    copies.push_back(C_split);

    bool mn_aligned = M % bm == 0 && N % bn == 0;
    bool k_aligned = K % bk == 0;

    std::ostringstream kname;
    kname << "steel_gemm_splitk_" << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
          << type_to_name(C_split) << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn << "_MN_" << (mn_aligned ? "t" : "n")
          << "aligned" << "_K_" << (k_aligned ? "t" : "n") << "aligned";

    // Encode and dispatch gemm kernel
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = get_steel_gemm_splitk_kernel(
        d,
        kname.str(),
        a,
        C_split,
        transpose_a,
        transpose_b,
        bm,
        bn,
        bk,
        wm,
        wn,
        mn_aligned,
        k_aligned);

    compute_encoder->setComputePipelineState(kernel);

    int tn = (N + bn - 1) / bn;
    int tm = (M + bm - 1) / bm;

    GEMMSpiltKParams params{
        M,
        N,
        K,
        lda,
        ldb,
        N,
        tn,
        tm,
        split_k_partitions,
        split_k_partition_stride,
        split_k_partition_size,
        gemm_k_iterations};

    MTL::Size group_dims = MTL::Size(32, wn, wm);
    MTL::Size grid_dims = MTL::Size(tn, tm, split_k_partitions);

    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(b, 1);
    compute_encoder.set_output_array(C_split, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMSpiltKParams), 3);
    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    // Do accum kernel
    {
      auto kernel_name = "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
          type_to_name(C_split) + "_axbpy";
      auto kernel = get_steel_gemm_splitk_accum_kernel(
          d, kernel_name, C_split, out, true);

      compute_encoder->setComputePipelineState(kernel);

      // Set the arguments for the kernel
      compute_encoder.set_input_array(C_split, 0);
      compute_encoder.set_output_array(out, 1);
      compute_encoder->setBytes(&split_k_partitions, sizeof(int), 2);
      compute_encoder->setBytes(&split_k_partition_stride, sizeof(int), 3);
      compute_encoder->setBytes(&N, sizeof(int), 4);
      compute_encoder.set_input_array(c, 5);
      compute_encoder->setBytes(&ldc, sizeof(int), 6);
      compute_encoder->setBytes(&fdc, sizeof(int), 7);
      compute_encoder->setBytes(&alpha_, sizeof(float), 8);
      compute_encoder->setBytes(&beta_, sizeof(float), 9);

      // Launch enough thread groups for each output
      MTL::Size grid_dims = MTL::Size(N, M, 1);
      MTL::Size group_dims = MTL::Size(std::min(1024, N * M), 1, 1);

      compute_encoder.dispatchThreads(grid_dims, group_dims);
    }

    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular addmm dispatch

  // Determine dispatch kernel
  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  if ((size_t)batch_size_out * M * N >= 1ul << 20) {
    if (!transpose_a && transpose_b) {
      bm = 64;
      bn = (out.dtype() == float32) ? 64 : 32;
      bk = (out.dtype() == float32) ? 16 : 32;
    } else {
      bm = 64;
      bn = 64;
    }
  }

  // Prepare kernel name
  std::ostringstream kname;
  kname << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn;

  std::string base_name = kname.str();

  const bool has_batch = (batch_shape.size() > 1);
  const bool use_out_source = true;
  const bool do_axpby = !(alpha_ == 1. && beta_ == 1.);
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool do_gather = false;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
      {&do_gather, MTL::DataType::DataTypeBool, 300},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_do_gather_" << (do_gather ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn);

  compute_encoder->setComputePipelineState(kernel);

  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams gemm_params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ A_batch_stride.back(),
      /* const size_t batch_stride_b = */ B_batch_stride.back(),
      /* const size_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

  GEMMAddMMParams params{
      /* const int ldc = */ ldc,
      /* const int fdc = */ fdc,
      /* const size_t batch_stride_c = */ C_batch_stride.back(),
      /* const float alpha = */ alpha_,
      /* const float beta = */ beta_};

  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  std::vector<size_t> batch_strides = A_batch_stride;
  batch_strides.insert(
      batch_strides.end(), B_batch_stride.begin(), B_batch_stride.end());
  batch_strides.insert(
      batch_strides.end(), C_batch_stride.begin(), C_batch_stride.end());

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(c, 2);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&gemm_params, sizeof(GEMMParams), 4);
  compute_encoder->setBytes(&params, sizeof(GEMMAddMMParams), 5);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  using namespace mlx::steel;
  // assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    copy_gpu(zero, out, CopyType::Scalar, s);
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler([zero](MTL::CommandBuffer*) {});
    return;
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr, bool is_vector) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [transpose_a, a_cols, a] = check_transpose(a_pre, M == 1);
  auto [transpose_b, b_cols, b] = check_transpose(b_pre, N == 1);

  int lda = a_cols;
  int ldb = b_cols;

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  bool has_op_mask = inputs.size() > 3;
  bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;

  std::vector<int> batch_shape{1};
  size_t A_batch_str = 0;
  size_t B_batch_str = 0;

  std::vector<size_t> batch_strides;

  if (out.ndim() > 2) {
    auto get_batch_dims = [](const auto& v) {
      return decltype(v){v.begin(), v.end() - 2};
    };

    std::vector<int> bshape{out.shape().begin(), out.shape().end() - 2};
    std::vector<std::vector<size_t>> bstrides;

    for (auto& arr : inputs) {
      bstrides.emplace_back(arr.strides().begin(), arr.strides().end() - 2);
    }

    auto [bshape_c, bstrides_c] = collapse_contiguous_dims(bshape, bstrides);
    batch_shape = bshape_c;
    A_batch_str = bstrides_c[0].back();
    B_batch_str = bstrides_c[1].back();

    for (auto& bstr : bstrides_c) {
      batch_strides.insert(batch_strides.end(), bstr.begin(), bstr.end());
    }
  } else {
    batch_strides = std::vector<size_t>(inputs.size(), 0);
  }

  size_t matrix_stride_out = size_t(M) * N;
  size_t batch_size_out = out.size() / (matrix_stride_out);

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch

  // Determine dispatch kernel
  int bm = block_size_, bn = block_size_, bk = 16;
  int wm = 2, wn = 2;
  bool mn_aligned = M % bm == 0 && N % bn == 0;
  bool k_aligned = K % bk == 0;

  // Prepare kernel name
  std::string out_mask_nm = has_out_mask ? type_to_name(inputs[2]) : "nomask";
  std::string op_mask_nm = has_op_mask ? type_to_name(inputs.back()) : "nomask";

  std::ostringstream kname;
  kname << "steel_gemm_block_outmask_" << out_mask_nm << "_opmask_"
        << op_mask_nm << "_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn << "_MN_" << (mn_aligned ? "t" : "n")
        << "aligned" << "_K_" << (k_aligned ? "t" : "n") << "aligned";

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_masked_kernel(
      d,
      kname.str(),
      out,
      has_out_mask ? std::optional<array>{inputs[2]} : std::nullopt,
      has_op_mask ? std::optional<array>{inputs.back()} : std::nullopt,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn,
      mn_aligned,
      k_aligned);
  compute_encoder->setComputePipelineState(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ A_batch_str,
      /* const size_t batch_stride_b = */ B_batch_str,
      /* const size_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

  // Prepare launch grid params
  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  std::vector<int> mask_strides;

  if (has_out_mask) {
    auto& out_mask = inputs[2];
    mask_strides.push_back(*(out_mask.strides().end() - 1));
    mask_strides.push_back(*(out_mask.strides().end() - 2));

    compute_encoder.set_input_array(out_mask, 10);
  }

  if (has_op_mask) {
    auto& lhs_mask = inputs[2 + has_out_mask];
    mask_strides.push_back(*(lhs_mask.strides().end() - 1));
    mask_strides.push_back(*(lhs_mask.strides().end() - 2));

    compute_encoder.set_input_array(lhs_mask, 11);

    auto& rhs_mask = inputs[3 + has_out_mask];
    mask_strides.push_back(*(rhs_mask.strides().end() - 1));
    mask_strides.push_back(*(rhs_mask.strides().end() - 2));

    compute_encoder.set_input_array(rhs_mask, 12);
  }

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&params, sizeof(GEMMParams), 4);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  set_vector_bytes(compute_encoder, mask_strides, 13);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  // Clear copies
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  using namespace mlx::steel;
  // assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[GatherMM] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    copy_gpu(zero, out, CopyType::Scalar, s);
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler([zero](MTL::CommandBuffer*) {});
    return;
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr, bool is_vector) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [transpose_a, a_cols, a] = check_transpose(a_pre, M == 1);
  auto [transpose_b, b_cols, b] = check_transpose(b_pre, N == 1);

  int lda = a_cols;
  int ldb = b_cols;

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto get_batch_dims = [](const auto& v) {
    return decltype(v){v.begin(), v.end() - 2};
  };

  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  std::vector<int> batch_shape = get_batch_dims(out.shape());
  std::vector<size_t> batch_strides;

  batch_strides.insert(
      batch_strides.end(),
      lhs_indices.strides().begin(),
      lhs_indices.strides().end());
  size_t lhs_indices_str = batch_strides.empty() ? 0 : batch_strides.back();

  batch_strides.insert(
      batch_strides.end(),
      rhs_indices.strides().begin(),
      rhs_indices.strides().end());
  size_t rhs_indices_str = batch_strides.empty() ? 0 : batch_strides.back();

  int batch_ndim = batch_shape.size();

  if (batch_ndim == 0) {
    batch_shape = {1};
    batch_strides = {0};
  }

  int batch_ndim_A = a.ndim() - 2;
  int batch_ndim_B = b.ndim() - 2;
  std::vector<int> operand_batch_ndim = {batch_ndim_A, batch_ndim_B};

  std::vector<int> batch_shape_A = get_batch_dims(a.shape());
  std::vector<size_t> batch_strides_A = get_batch_dims(a.strides());
  std::vector<int> batch_shape_B = get_batch_dims(b.shape());
  std::vector<size_t> batch_strides_B = get_batch_dims(b.strides());

  if (batch_ndim_A == 0) {
    batch_shape_A = {1};
    batch_strides_A = {0};
  }

  if (batch_ndim_B == 0) {
    batch_shape_B = {1};
    batch_strides_B = {0};
  }

  size_t matrix_stride_out = size_t(M) * N;
  auto batch_size_out = out.size() / matrix_stride_out;

  /////////////////////////////////////////////////////////////////////////////
  // Gemv specialization

  // Route to gemv if needed
  if (std::min(M, N) == 1) {
    // Collect problem info
    bool is_b_matrix = N != 1;

    auto& mat = is_b_matrix ? b : a;
    auto& vec = is_b_matrix ? a : b;
    bool transpose_mat = is_b_matrix ? !transpose_b : transpose_a;
    int in_vector_len = K;
    int out_vector_len = is_b_matrix ? N : M;

    int mat_cols = transpose_mat ? out_vector_len : in_vector_len;
    int mat_rows = transpose_mat ? in_vector_len : out_vector_len;
    int mat_ld = is_b_matrix ? b_cols : a_cols;

    auto batch_strides_mat = is_b_matrix ? batch_strides_B : batch_strides_A;
    auto batch_strides_vec = is_b_matrix ? batch_strides_A : batch_strides_B;

    auto batch_shape_mat = is_b_matrix ? batch_shape_B : batch_shape_A;
    auto batch_shape_vec = is_b_matrix ? batch_shape_A : batch_shape_B;

    if (!is_b_matrix) {
      batch_strides = rhs_indices.strides();
      batch_strides.insert(
          batch_strides.end(),
          lhs_indices.strides().begin(),
          lhs_indices.strides().end());
    }

    int batch_ndim = batch_shape.size();

    // Determine dispatch kernel
    int tm = 4, tn = 4;
    int sm = 1, sn = 32;
    int bm = 1, bn = 1;
    int n_out_per_tgp;
    std::ostringstream kname;

    if (transpose_mat) {
      if (in_vector_len >= 8192 && out_vector_len >= 2048) {
        sm = 4;
        sn = 8;
      } else {
        sm = 8;
        sn = 4;
      }

      if (out_vector_len >= 2048) {
        bn = 16;
      } else if (out_vector_len >= 512) {
        bn = 4;
      } else {
        bn = 2;
      }

      // Specialized kernel for very small outputs
      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * sn * tn;
      kname << "gemv_t_gather_" << type_to_name(out);

    } else {
      bm = out_vector_len >= 4096 ? 8 : 4;
      sn = 32;

      // Specialized kernel for very small outputs
      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * sm * tm;
      kname << "gemv_gather_" << type_to_name(out);
    }

    kname << "_bm" << bm << "_bn" << bn << "_sm" << sm << "_sn" << sn << "_tm"
          << tm << "_tn" << tn;

    // Encode and dispatch kernel
    auto& compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(32, bn, bm);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    compute_encoder.set_input_array(mat, 0);
    compute_encoder.set_input_array(vec, 1);
    compute_encoder.set_output_array(out, 3);

    compute_encoder->setBytes(&in_vector_len, sizeof(int), 4);
    compute_encoder->setBytes(&out_vector_len, sizeof(int), 5);
    compute_encoder->setBytes(&mat_ld, sizeof(int), 6);

    compute_encoder->setBytes(&batch_ndim, sizeof(int), 9);
    set_vector_bytes(compute_encoder, batch_shape, 10);
    set_vector_bytes(compute_encoder, batch_strides, 11);

    int batch_ndim_vec = batch_shape_vec.size();
    compute_encoder->setBytes(&batch_ndim_vec, sizeof(int), 12);
    set_vector_bytes(compute_encoder, batch_shape_vec, 13);
    set_vector_bytes(compute_encoder, batch_strides_vec, 14);

    int batch_ndim_mat = batch_shape_mat.size();
    compute_encoder->setBytes(&batch_ndim_mat, sizeof(int), 15);
    set_vector_bytes(compute_encoder, batch_shape_mat, 16);
    set_vector_bytes(compute_encoder, batch_strides_mat, 17);

    compute_encoder.set_input_array(lhs_indices, 18 + int(!is_b_matrix));
    compute_encoder.set_input_array(rhs_indices, 18 + int(is_b_matrix));

    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch

  // Determine dispatch kernel
  int bm = 32, bn = 32, bk = 16;
  int wm = 2, wn = 2;

  if ((size_t)batch_size_out * M * N >= 1ul << 20) {
    if (!transpose_a && transpose_b) {
      bm = 64;
      bn = (out.dtype() == float32) ? 64 : 32;
      bk = (out.dtype() == float32) ? 16 : 32;
    } else {
      bm = 64;
      bn = 64;
    }
  }

  // Prepare kernel name
  std::ostringstream kname;
  kname << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn;

  std::string base_name = kname.str();

  const bool has_batch = batch_ndim > 1;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool do_gather = true;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
      {&do_gather, MTL::DataType::DataTypeBool, 300},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_do_gather_" << (do_gather ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      transpose_a,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn);

  compute_encoder->setComputePipelineState(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ ldb,
      /* const int ldd = */ N,
      /* const int tiles_n = */ tn,
      /* const int tiles_m = */ tm,
      /* const size_t batch_stride_a = */ lhs_indices_str,
      /* const size_t batch_stride_b = */ rhs_indices_str,
      /* const size_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ batch_ndim};

  // Prepare launch grid params
  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder->setBytes(&params, sizeof(GEMMParams), 4);

  set_vector_bytes(compute_encoder, batch_shape, 6);
  set_vector_bytes(compute_encoder, batch_strides, 7);

  compute_encoder.set_input_array(lhs_indices, 10);
  compute_encoder.set_input_array(rhs_indices, 11);

  std::vector operand_shape = batch_shape_A;
  operand_shape.insert(
      operand_shape.end(), batch_shape_B.begin(), batch_shape_B.end());

  std::vector operand_strides = batch_strides_A;
  operand_strides.insert(
      operand_strides.end(), batch_strides_B.begin(), batch_strides_B.end());

  operand_batch_ndim.push_back(0);

  set_vector_bytes(compute_encoder, operand_shape, 13);
  set_vector_bytes(compute_encoder, operand_strides, 14);
  set_vector_bytes(compute_encoder, operand_batch_ndim, 15);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  // Clear copies
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

} // namespace mlx::core
