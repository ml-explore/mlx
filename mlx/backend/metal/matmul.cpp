// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/host.h"
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

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Steel matmul fallback
///////////////////////////////////////////////////////////////////////////////

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
    std::vector<array>& copies) {
  using namespace mlx::steel;

  // Coalesce (B, M, K) X (K, N) to (B*M, K) X (K, N)
  if (batch_size_out > 1 && !transpose_a &&
      a.data_size() == batch_size_out * M * K && b.size() == K * N) {
    M = M * batch_size_out;
    batch_size_out = 1;
  }

  // Account for batch sizes and basic broadcasting
  int batch_size_a = a.data_size() / (M * K);
  int batch_size_b = b.data_size() / (K * N);

  int matrix_stride_a = (batch_size_a == 1) ? 0 : M * K;
  int matrix_stride_b = (batch_size_b == 1) ? 0 : K * N;
  int matrix_stride_out = M * N;

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

    std::ostringstream kname;
    kname << "steel_gemm_splitk_" << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
          << type_to_name(C_split) << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn << "_MN_"
          << ((M % bm == 0 && N % bn == 0) ? "t" : "n") << "aligned"
          << "_K_" << ((K % bk == 0) ? "t" : "n") << "aligned";

    // Encode and dispatch gemm kernel
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
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

    set_array_buffer(compute_encoder, a, 0);
    set_array_buffer(compute_encoder, b, 1);
    set_array_buffer(compute_encoder, C_split, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMSpiltKParams), 3);
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

    // Do accum kernel
    {
      auto c_split_buf =
          static_cast<const MTL::Resource*>(C_split.buffer().ptr());
      const class MTL::Resource* const resources[1] = {c_split_buf};
      compute_encoder->memoryBarrier(resources, 1);

      auto kernel = d.get_kernel(
          "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
          type_to_name(C_split));
      compute_encoder->setComputePipelineState(kernel);

      // Set the arguments for the kernel
      set_array_buffer(compute_encoder, C_split, 0);
      set_array_buffer(compute_encoder, out, 1);
      compute_encoder->setBytes(&split_k_partitions, sizeof(int), 2);
      compute_encoder->setBytes(&split_k_partition_stride, sizeof(int), 3);
      compute_encoder->setBytes(&N, sizeof(int), 4);

      // Launch enough thread groups for each output
      MTL::Size grid_dims = MTL::Size(N, M, 1);
      MTL::Size group_dims = MTL::Size(std::min(1024, N * M), 1, 1);

      compute_encoder->dispatchThreads(grid_dims, group_dims);
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
  kname << "steel_gemm_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn << "_MN_"
        << ((M % bm == 0 && N % bn == 0) ? "t" : "n") << "aligned"
        << "_K_" << ((K % bk == 0) ? "t" : "n") << "aligned";

  // Encode and dispatch kernel
  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  // Prepare steel matmul params
  GEMMParams params{
      M,
      N,
      K,
      lda,
      ldb,
      N,
      tn,
      tm,
      matrix_stride_a,
      matrix_stride_b,
      matrix_stride_out,
      swizzle_log,
      (K / bk)};

  // Prepare launch grid params
  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  // Launch only 1 kernel in the case of simple batching / broadcasting
  if (batch_size_out == std::max(batch_size_a, batch_size_b) &&
      (batch_size_a == batch_size_b ||
       std::min(batch_size_a, batch_size_b) == 1)) {
    set_array_buffer(compute_encoder, a, 0);
    set_array_buffer(compute_encoder, b, 1);
    set_array_buffer(compute_encoder, out, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMParams), 3);
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  } else { // Otherwise launch kernels with set offsets

    MTL::Size grid_dims_single = MTL::Size(tn, tm, 1);

    for (int i = 0; i < batch_size_out; ++i) {
      auto a_off = elem_to_loc(M * K * i, a.shape(), a.strides());
      auto b_off = elem_to_loc(K * N * i, b.shape(), b.strides());

      auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
      auto b_buf = static_cast<const MTL::Buffer*>(b.buffer().ptr());
      auto out_buf = static_cast<const MTL::Buffer*>(out.buffer().ptr());

      compute_encoder->setBuffer(a_buf, a_off * a.itemsize(), 0);
      compute_encoder->setBuffer(b_buf, b_off * b.itemsize(), 1);
      compute_encoder->setBuffer(out_buf, i * M * N * out.itemsize(), 2);

      compute_encoder->setBytes(&params, sizeof(GEMMParams), 3);
      compute_encoder->dispatchThreadgroups(grid_dims_single, group_dims);
    }
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (!is_floating_point(out.dtype())) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, a_cols, a] = check_transpose(a_pre);
  auto [b_transposed, b_cols, b] = check_transpose(b_pre);

  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto batch_size_out = out.size() / (M * N);

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

    int batch_size_mat = mat.data_size() / (mat_cols * mat_rows);
    int stride_mat = batch_size_mat == 1 ? 0 : mat_cols * mat_rows;

    int batch_size_vec = vec.data_size() / in_vector_len;
    int stride_vec = batch_size_vec == 1 ? 0 : in_vector_len;

    // Determine if inputs have simple batching / broadcasting
    bool contiguous_kernel =
        (batch_size_out == std::max(batch_size_mat, batch_size_vec) &&
         (batch_size_mat == batch_size_vec ||
          std::min(batch_size_mat, batch_size_vec) == 1));

    int nc_dim = out.ndim() - 2;

    // Determine dispatch kernel
    int tm = 4, tn = 4;
    int bm, bn, n_out_per_tgp;
    std::ostringstream kname;

    if (transpose_mat) {
      bm = 8;
      bn = 8;
      if (out_vector_len >= 24576) {
        bn = 128;
      } else if (out_vector_len >= 16384) {
        bn = 64;
      } else if (out_vector_len >= 8192) {
        bn = 16;
      }

      // Specialized kernel for very small outputs
      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * tn;
      kname << "gemv_t_" << type_to_name(out);

    } else {
      bm = out_vector_len >= 4096 ? 8 : 4;
      bn = 32;

      // Specialized kernel for very small outputs
      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * tm;
      kname << "gemv_" << type_to_name(out);
    }

    kname << "_bm" << bm << "_bn" << bn << "_tm" << tm << "_tn" << tn;

    if (!contiguous_kernel) {
      kname << "_nc";
    }

    // Encode and dispatch kernel
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(bn, bm, 1);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    set_array_buffer(compute_encoder, mat, 0);
    set_array_buffer(compute_encoder, vec, 1);
    set_array_buffer(compute_encoder, out, 2);

    compute_encoder->setBytes(&in_vector_len, sizeof(int), 3);
    compute_encoder->setBytes(&out_vector_len, sizeof(int), 4);

    if (contiguous_kernel) {
      compute_encoder->setBytes(&stride_vec, sizeof(int), 5);
      compute_encoder->setBytes(&stride_mat, sizeof(int), 6);
    } else {
      // In case of complex broadcasting, we consider the shape[:-2] and
      // strides [:-2] to determine the location of a batch
      // nc_dim = out.ndim() - 2
      compute_encoder->setBytes(&nc_dim, sizeof(int), 5);
      compute_encoder->setBytes(out.shape().data(), nc_dim * sizeof(int), 6);
      compute_encoder->setBytes(
          vec.strides().data(), nc_dim * sizeof(size_t), 7);
      compute_encoder->setBytes(
          mat.strides().data(), nc_dim * sizeof(size_t), 8);
    }

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

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

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  if (!is_floating_point(out.dtype())) {
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

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto check_transpose = [&copies, &s](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(arr_copy);
      size_t stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [transpose_a, a_cols, a] = check_transpose(a_pre);
  auto [transpose_b, b_cols, b] = check_transpose(b_pre);

  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto batch_size_out = out.size() / (M * N);

  array c = c_pre;
  int ldc = c.strides()[c.ndim() - 2];
  int fdc = c.strides()[c.ndim() - 1];
  int matrix_stride_c = c.ndim() <= 2 ? 0 : c.strides()[c.ndim() - 3];

  int lda = a_cols;
  int ldb = b_cols;

  using namespace mlx::steel;

  // Account for batch sizes and basic broadcasting
  int batch_size_a = a.data_size() / (M * K);
  int batch_size_b = b.data_size() / (K * N);

  int matrix_stride_a = (batch_size_a == 1) ? 0 : M * K;
  int matrix_stride_b = (batch_size_b == 1) ? 0 : K * N;
  int matrix_stride_out = M * N;

  int _tm = M / 16;
  int _tn = N / 16;
  int _tk = K / 16;

  /////////////////////////////////////////////////////////////////////////////
  // Split K specialization

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

    std::ostringstream kname;
    kname << "steel_gemm_splitk_" << (transpose_a ? 't' : 'n')
          << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
          << type_to_name(C_split) << "_bm" << bm << "_bn" << bn << "_bk" << bk
          << "_wm" << wm << "_wn" << wn << "_MN_"
          << ((M % bm == 0 && N % bn == 0) ? "t" : "n") << "aligned"
          << "_K_" << ((K % bk == 0) ? "t" : "n") << "aligned";

    // Encode and dispatch gemm kernel
    auto compute_encoder = d.get_command_encoder(s.index);
    auto kernel = d.get_kernel(kname.str());
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

    set_array_buffer(compute_encoder, a, 0);
    set_array_buffer(compute_encoder, b, 1);
    set_array_buffer(compute_encoder, C_split, 2);

    compute_encoder->setBytes(&params, sizeof(GEMMSpiltKParams), 3);
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

    // Do accum kernel
    {
      auto kernel = d.get_kernel(
          "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
          type_to_name(C_split) + "_axpby");
      compute_encoder->setComputePipelineState(kernel);

      // Set the arguments for the kernel
      set_array_buffer(compute_encoder, C_split, 0);
      set_array_buffer(compute_encoder, out, 1);
      compute_encoder->setBytes(&split_k_partitions, sizeof(int), 2);
      compute_encoder->setBytes(&split_k_partition_stride, sizeof(int), 3);
      compute_encoder->setBytes(&N, sizeof(int), 4);
      set_array_buffer(compute_encoder, c, 5);
      compute_encoder->setBytes(&ldc, sizeof(int), 6);
      compute_encoder->setBytes(&fdc, sizeof(int), 7);
      compute_encoder->setBytes(&alpha_, sizeof(float), 8);
      compute_encoder->setBytes(&beta_, sizeof(float), 9);

      // Launch enough thread groups for each output
      MTL::Size grid_dims = MTL::Size(N, M, 1);
      MTL::Size group_dims = MTL::Size(std::min(1024, N * M), 1, 1);

      compute_encoder->dispatchThreads(grid_dims, group_dims);
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

  std::ostringstream kname;
  kname << "steel_addmm_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn << "_MN_"
        << ((M % bm == 0 && N % bn == 0) ? "t" : "n") << "aligned"
        << "_K_" << ((K % bk == 0) ? "t" : "n") << "aligned"
        << ((alpha_ == 1. && beta_ == 1.) ? "_add" : "_axpby");

  // Encode and dispatch kernel
  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = 0; // tm >= 6 ? 3 : (tm <= 3 ? 0 : 2);

  GEMMAddMMParams params{
      M,
      N,
      K,
      lda,
      ldb,
      ldc,
      N,
      tn,
      tm,
      matrix_stride_a,
      matrix_stride_b,
      matrix_stride_c,
      matrix_stride_out,
      swizzle_log,
      (K / bk),
      alpha_,
      beta_,
      fdc};

  int tile = 1 << swizzle_log;
  tm = (tm + tile - 1) / tile;
  tn = tn * tile;

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tn, tm, batch_size_out);

  // Launch only 1 kernel in the case of simple batching / broadcasting
  if (batch_size_out == std::max(batch_size_a, batch_size_b) &&
      (batch_size_a == batch_size_b ||
       std::min(batch_size_a, batch_size_b) == 1)) {
    set_array_buffer(compute_encoder, a, 0);
    set_array_buffer(compute_encoder, b, 1);
    set_array_buffer(compute_encoder, c, 2);
    set_array_buffer(compute_encoder, out, 3);

    compute_encoder->setBytes(&params, sizeof(GEMMAddMMParams), 4);
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  } else { // Otherwise launch kernels with set offsets

    MTL::Size grid_dims_single = MTL::Size(tn, tm, 1);

    for (int i = 0; i < batch_size_out; ++i) {
      auto a_off = elem_to_loc(M * K * i, a.shape(), a.strides());
      auto b_off = elem_to_loc(K * N * i, b.shape(), b.strides());
      auto c_off = elem_to_loc(M * N * i, c.shape(), c.strides());

      auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
      auto b_buf = static_cast<const MTL::Buffer*>(b.buffer().ptr());
      auto c_buf = static_cast<const MTL::Buffer*>(c.buffer().ptr());
      auto out_buf = static_cast<const MTL::Buffer*>(out.buffer().ptr());

      compute_encoder->setBuffer(a_buf, a_off * a.itemsize(), 0);
      compute_encoder->setBuffer(b_buf, b_off * b.itemsize(), 1);
      compute_encoder->setBuffer(c_buf, c_off * c.itemsize(), 2);
      compute_encoder->setBuffer(out_buf, i * M * N * out.itemsize(), 3);

      compute_encoder->setBytes(&params, sizeof(GEMMAddMMParams), 4);
      compute_encoder->dispatchThreadgroups(grid_dims_single, group_dims);
    }
  }

  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  return;
}

} // namespace mlx::core
