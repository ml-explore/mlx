// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <numeric>
#include <sstream>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/binary.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/matmul.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array> check_transpose(
    std::vector<array>& copies,
    const Stream& s,
    const array& arr,
    bool is_vector) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && (!is_vector || stx == arr.shape(-1))) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && (!is_vector || sty == arr.shape(-2))) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    copies.push_back(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
};

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    d.add_temporary(x_copy, s.index);
    return x_copy;
  } else {
    return x;
  }
}

inline std::tuple<bool, int64_t, array>
ensure_batch_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides()[x.ndim() - 2], x);
  }

  bool rc = true;
  for (int i = 0; i < x.ndim() - 3; i++) {
    rc &= x.strides()[i + 1] * x.shape(i) == x.strides()[i];
  }
  if (rc) {
    auto stx = x.strides()[x.ndim() - 2];
    auto sty = x.strides()[x.ndim() - 1];
    auto K = x.shape(-2);
    auto N = x.shape(-1);
    if (sty == 1 && (N != 1 || stx == N)) {
      return std::make_tuple(false, stx, x);
    }
    if (stx == 1 && (N != 1 || sty == K)) {
      return std::make_tuple(true, sty, x);
    }
  }

  array x_copy = contiguous_copy_gpu(x, s);
  d.add_temporary(x_copy, s.index);
  return std::make_tuple(false, x_copy.strides()[x_copy.ndim() - 2], x_copy);
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
// Steel matmul fallback
///////////////////////////////////////////////////////////////////////////////

#define GEMM_TPARAM_MACRO(devc)                                           \
  if (devc == 'g' || devc == 'p') { /* Small device */                    \
    if (out.dtype() == complex64) {                                       \
      bm = 64;                                                            \
      bn = 32;                                                            \
      bk = 8;                                                             \
      wm = 4;                                                             \
      wn = 1;                                                             \
    } else if (!transpose_a && transpose_b) { /* nt */                    \
      bm = 64;                                                            \
      bn = 32;                                                            \
      bk = 32;                                                            \
      wm = 2;                                                             \
      wn = 2;                                                             \
    } else if (out.dtype() != float32) { /* half and bfloat */            \
      bm = 64;                                                            \
      bn = 64;                                                            \
      bk = 16;                                                            \
      wm = 1;                                                             \
      wn = 2;                                                             \
    }                                                                     \
  } else if (devc == 'd') { /* Large device */                            \
    if ((size_t)batch_size_out * M * N >= 1ul << 20) { /* large matmul */ \
      if (out.dtype() != float32) { /* half and bfloat */                 \
        if (2 * std::max(M, N) > K) { /* Reasonable K */                  \
          bm = 64;                                                        \
          bn = 64;                                                        \
          bk = 16;                                                        \
          wm = 1;                                                         \
          wn = 2;                                                         \
        } else if (!transpose_a && transpose_b) { /* nt with large k */   \
          bm = 64;                                                        \
          bn = 32;                                                        \
          bk = 32;                                                        \
          wm = 2;                                                         \
          wn = 2;                                                         \
        } else { /* nn with large K */                                    \
          bm = 32;                                                        \
          bn = 64;                                                        \
          bk = 16;                                                        \
          wm = 1;                                                         \
          wn = 2;                                                         \
        }                                                                 \
      } /* float takes default */                                         \
    } else { /* smaller matmul */                                         \
      if (out.dtype() != float32) { /* half and bfloat */                 \
        if (!transpose_a && transpose_b) { /* nt */                       \
          bm = 64;                                                        \
          bn = 32;                                                        \
          bk = 32;                                                        \
          wm = 2;                                                         \
          wn = 2;                                                         \
        } else { /* nn */                                                 \
          bm = 64;                                                        \
          bn = 64;                                                        \
          bk = 16;                                                        \
          wm = 1;                                                         \
          wn = 2;                                                         \
        }                                                                 \
      } else { /* floats */                                               \
        if (!transpose_a && transpose_b) { /* nt */                       \
          bm = 32;                                                        \
          bn = 64;                                                        \
          bk = 16;                                                        \
          wm = 1;                                                         \
          wn = 2;                                                         \
        } else { /* nn */                                                 \
          bm = 64;                                                        \
          bn = 32;                                                        \
          bk = 32;                                                        \
          wm = 2;                                                         \
          wn = 2;                                                         \
        }                                                                 \
      }                                                                   \
    }                                                                     \
  } else { /* Medium device */                                            \
    bm = 64;                                                              \
    bn = 64;                                                              \
    bk = 16;                                                              \
    wm = 2;                                                               \
    wn = 2;                                                               \
  }

///////////////////////////////////////////////////////////////////////////////
// Regular steel matmul dispatch
///////////////////////////////////////////////////////////////////////////////

template <bool CHECK_AB>
void steel_matmul_regular_axpby_nax(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape,
    Strides batch_strides,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t matrix_stride_out,
    int64_t C_batch_stride /* = 0*/,
    float alpha /* = 1.0f */,
    float beta /* = 0.0f */) {
  using namespace mlx::steel;

  // Determine dispatch kernel
  int bm = 128, bn = 128, bk = 512;
  int wm = 4, wn = 4;

  // Prepare kernel name
  std::ostringstream kname;

  // clang-format off
  kname << "steel_gemm_fused_nax_"
        << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n')
        << "_" << type_to_name(a)
        << "_" << type_to_name(out)
        << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn; // clang-format on

  std::string base_name = kname.str();

  const bool has_batch = (batch_shape.size() > 1);
  const bool use_out_source = CHECK_AB && (alpha != 0.0f || beta != 1.0f);
  const bool do_axpby = use_out_source && (alpha != 1.0f || beta != 1.0f);
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_nax_kernel(
      /* metal::Device& d = */ d,
      /* const std::string& kernel_name = */ base_name,
      /* const std::string& hash_name = */ hash_name,
      /* const metal::MTLFCList& func_consts = */ func_consts,
      /* const array& out = */ out,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* int bm = */ bm,
      /* int bn = */ bn,
      /* int bk = */ bk,
      /* int wm = */ wm,
      /* int wn = */ wn);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Use problem size to determine threadblock swizzle
  int tn = (N + bn - 1) / bn;
  int tm = (M + bm - 1) / bm;

  // TODO: Explore device-based tuning for swizzle
  int swizzle_log = tm <= 3 ? 0 : 1;

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
      /* const int64_t batch_stride_a = */ A_batch_stride,
      /* const int64_t batch_stride_b = */ B_batch_stride,
      /* const int64_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

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

  compute_encoder.set_bytes(params, 4);

  if (has_batch) {
    compute_encoder.set_vector_bytes(batch_shape, 6);
    compute_encoder.set_vector_bytes(batch_strides, 7);
  }

  if (use_out_source) {
    int ldc = c.strides()[c.ndim() - 2];
    int fdc = c.strides()[c.ndim() - 1];

    GEMMAddMMParams params{
        /* const int ldc = */ ldc,
        /* const int fdc = */ fdc,
        /* const int64_t batch_stride_c = */ C_batch_stride,
        /* const float alpha = */ alpha,
        /* const float beta = */ beta};

    compute_encoder.set_input_array(c, 2);
    compute_encoder.set_bytes(params, 5);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Record copies
  d.add_temporaries(std::move(copies), s.index);
}

template <bool CHECK_AB>
void steel_matmul_regular_axpby(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape,
    Strides batch_strides,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t matrix_stride_out,
    int64_t C_batch_stride /* = 0*/,
    float alpha /* = 1.0f */,
    float beta /* = 0.0f */) {
  if (metal::is_nax_available() && !issubdtype(a.dtype(), complexfloating) &&
      (env::enable_tf32() || a.dtype() != float32)) {
    return steel_matmul_regular_axpby_nax<CHECK_AB>(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& a = */ a,
        /* const array& b = */ b,
        /* const array& c = */ c,
        /* array& out = */ out,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* int batch_size_out = */ batch_size_out,
        /* int lda = */ lda,
        /* int ldb = */ ldb,
        /* int ldd = */ ldd,
        /* bool transpose_a = */ transpose_a,
        /* bool transpose_b = */ transpose_b,
        /* std::vector<array>& copies = */ copies,
        /* Shape batch_shape = */ batch_shape,
        /* Strides batch_strides = */ batch_strides,
        /* int64_t A_batch_stride = */ A_batch_stride,
        /* int64_t B_batch_stride = */ B_batch_stride,
        /* int64_t matrix_stride_out = */ matrix_stride_out,
        /* int64_t C_batch_stride = */ C_batch_stride,
        /* float alpha = */ alpha,
        /* float beta = */ beta);
  }

  using namespace mlx::steel;

  // Determine dispatch kernel
  int bm = 64, bn = 64, bk = 16;
  int wm = 2, wn = 2;

  char devc = d.get_architecture().back();
  GEMM_TPARAM_MACRO(devc)

  // Prepare kernel name
  std::ostringstream kname;

  // clang-format off
  kname << "steel_gemm_fused_"
        << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n')
        << "_" << type_to_name(a)
        << "_" << type_to_name(out)
        << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn; // clang-format on

  std::string base_name = kname.str();

  const bool has_batch = (batch_shape.size() > 1);
  const bool use_out_source = CHECK_AB && (alpha != 0.0f || beta != 1.0f);
  const bool do_axpby = use_out_source && (alpha != 1.0f || beta != 1.0f);
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // clang-format off
  kname << "_has_batch_" << (has_batch ? 't' : 'n')
        << "_use_out_source_" << (use_out_source ? 't' : 'n')
        << "_do_axpby_" << (do_axpby ? 't' : 'n')
        << "_align_M_" << (align_M ? 't' : 'n')
        << "_align_N_" << (align_N ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_fused_kernel(
      /* metal::Device& d = */ d,
      /* const std::string& kernel_name = */ base_name,
      /* const std::string& hash_name = */ hash_name,
      /* const metal::MTLFCList& func_consts = */ func_consts,
      /* const array& out = */ out,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* int bm = */ bm,
      /* int bn = */ bn,
      /* int bk = */ bk,
      /* int wm = */ wm,
      /* int wn = */ wn);

  compute_encoder.set_compute_pipeline_state(kernel);

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
      /* const int64_t batch_stride_a = */ A_batch_stride,
      /* const int64_t batch_stride_b = */ B_batch_stride,
      /* const int64_t batch_stride_d = */ matrix_stride_out,
      /* const int swizzle_log = */ swizzle_log,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ int(batch_shape.size())};

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

  compute_encoder.set_bytes(params, 4);

  if (has_batch) {
    compute_encoder.set_vector_bytes(batch_shape, 6);
    compute_encoder.set_vector_bytes(batch_strides, 7);
  }

  if (use_out_source) {
    int ldc = c.strides()[c.ndim() - 2];
    int fdc = c.strides()[c.ndim() - 1];

    GEMMAddMMParams params{
        /* const int ldc = */ ldc,
        /* const int fdc = */ fdc,
        /* const int64_t batch_stride_c = */ C_batch_stride,
        /* const float alpha = */ alpha,
        /* const float beta = */ beta};

    compute_encoder.set_input_array(c, 2);
    compute_encoder.set_bytes(params, 5);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Record copies
  d.add_temporaries(std::move(copies), s.index);
}

///////////////////////////////////////////////////////////////////////////////
// Split k steel matmul
///////////////////////////////////////////////////////////////////////////////

template <bool CHECK_AB = true>
void steel_gemm_splitk_axpby(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
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
  using namespace mlx::steel;

  int _tk = K / 16;

  int bm = M < 40 ? 16 : 32;
  int bn = N < 40 ? 16 : 32;
  int bk = 16;
  int wm = 2, wn = 2;

  int split_k_partitions = _tk < 16 ? 2 : (_tk < 32 ? 4 : (_tk < 64 ? 8 : 16));
  int split_k_partition_stride = M * N;
  int gemm_k_iterations = (K / bk) / split_k_partitions;
  int split_k_partition_size = gemm_k_iterations * bk;

  array C_split(
      {split_k_partitions, M, N},
      issubdtype(out.dtype(), complexfloating) ? complex64 : float32,
      nullptr,
      {});
  C_split.set_data(allocator::malloc(C_split.nbytes()));
  copies.push_back(C_split);

  bool mn_aligned = M % bm == 0 && N % bn == 0;
  bool k_aligned = K % bk == 0;
  std::ostringstream kname;

  // clang-format off
  kname << "steel_gemm_splitk_"
        << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n')
        << "_" << type_to_name(a)
        << "_" << type_to_name(C_split)
        << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn
        << "_MN_" << (mn_aligned ? "t" : "n") << "aligned"
        << "_K_" << (k_aligned ? "t" : "n") << "aligned"; // clang-format on

  // Encode and dispatch gemm kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_splitk_kernel(
      /* metal::Device& d = */ d,
      /* const std::string& kernel_name = */ kname.str(),
      /* const array& in = */ a,
      /* const array& out = */ C_split,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* int bm = */ bm,
      /* int bn = */ bn,
      /* int bk = */ bk,
      /* int wm = */ wm,
      /* int wn = */ wn,
      /* bool mn_aligned = */ mn_aligned,
      /* bool k_aligned = */ k_aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

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

  compute_encoder.set_bytes(params, 3);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Do accum kernel
  {
    const bool do_axpby = CHECK_AB && (alpha != 1.0f || beta != 0.0f);

    auto kernel_name = "steel_gemm_splitk_accum_" + type_to_name(out) + "_" +
        type_to_name(C_split);

    if (do_axpby) {
      kernel_name = kernel_name + "_axbpy";
    }

    auto kernel = get_steel_gemm_splitk_accum_kernel(
        /* metal::Device& d = */ d,
        /* const std::string& kernel_name = */ kernel_name,
        /* const array& in = */ C_split,
        /* const array& out = */ out,
        /* bool axbpy = */ do_axpby);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Set the arguments for the kernel
    compute_encoder.set_input_array(C_split, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(split_k_partitions, 2);
    compute_encoder.set_bytes(split_k_partition_stride, 3);
    compute_encoder.set_bytes(N, 4);

    if (do_axpby) {
      int ldc = c.strides()[c.ndim() - 2];
      int fdc = c.strides()[c.ndim() - 1];

      compute_encoder.set_input_array(c, 5);
      compute_encoder.set_bytes(ldc, 6);
      compute_encoder.set_bytes(fdc, 7);
      compute_encoder.set_bytes(alpha, 8);
      compute_encoder.set_bytes(beta, 9);
    }

    // Launch enough thread groups for each output
    MTL::Size grid_dims = MTL::Size(N, M, 1);
    auto group_dims = get_block_dims(N, M, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

///////////////////////////////////////////////////////////////////////////////
// Split matmul routing
///////////////////////////////////////////////////////////////////////////////

template <bool CHECK_AB>
void steel_matmul_axpby(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
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
    Shape batch_shape /* = {} */,
    Strides A_batch_stride /* = {} */,
    Strides B_batch_stride /* = {} */,
    Strides C_batch_stride /* = {} */,
    float alpha /* = 1.0f */,
    float beta /* = 0.0f */) {
  if (batch_shape.empty()) {
    /////////////////////////////////////////////////////////////////////////////
    // Check and collapse batch dimensions
    if constexpr (CHECK_AB) {
      auto [batch_shape_, A_bstride_, B_bstride_, C_bstride_] =
          collapse_batches(a, b, c);

      batch_shape = batch_shape_;
      A_batch_stride = A_bstride_;
      B_batch_stride = B_bstride_;
      C_batch_stride = C_bstride_;
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
    } else {
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
  }

  /////////////////////////////////////////////////////////////////////////////
  // Split K specialization

  int _tm = M / 16;
  int _tn = N / 16;
  int _tk = K / 16;

  if (batch_size_out == 1 && (_tm * _tn) <= 32 && _tk >= 8) {
    return steel_gemm_splitk_axpby<CHECK_AB>(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& a = */ a,
        /* const array& b = */ b,
        /* const array& c = */ c,
        /* array& out = */ out,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* int batch_size_out = */ batch_size_out,
        /* int lda = */ lda,
        /* int ldb = */ ldb,
        /* bool transpose_a = */ transpose_a,
        /* bool transpose_b = */ transpose_b,
        /* std::vector<array>& copies = */ copies,
        /* float alpha = */ alpha,
        /* float beta = */ beta);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch
  auto batch_strides = A_batch_stride;
  batch_strides.insert(
      batch_strides.end(), B_batch_stride.begin(), B_batch_stride.end());
  if (CHECK_AB && !C_batch_stride.empty()) {
    batch_strides.insert(
        batch_strides.end(), C_batch_stride.begin(), C_batch_stride.end());
  }

  int64_t A_batch_stride_ = A_batch_stride.empty() ? 0 : A_batch_stride.back();
  int64_t B_batch_stride_ = B_batch_stride.empty() ? 0 : B_batch_stride.back();
  int64_t C_batch_stride_ = C_batch_stride.empty() ? 0 : C_batch_stride.back();

  return steel_matmul_regular_axpby<CHECK_AB>(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* const array& c = */ c,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* int ldd = */ N,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ std::move(batch_shape),
      /* Strides batch_strides = */ std::move(batch_strides),
      /* int64_t A_batch_stride = */ A_batch_stride_,
      /* int64_t B_batch_stride = */ B_batch_stride_,
      /* int64_t matrix_stride_out = */ int64_t(M) * N,
      /* int64_t C_batch_stride = */ C_batch_stride_,
      /* float alpha = */ alpha,
      /* float beta = */ beta);
}

///////////////////////////////////////////////////////////////////////////////
// GEMV dispatch
///////////////////////////////////////////////////////////////////////////////

template <bool CHECK_AB = true>
void gemv_axbpy(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    const array& c,
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
    Shape batch_shape = {},
    Strides A_batch_stride = {},
    Strides B_batch_stride = {},
    Strides C_batch_stride = {},
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Collect problem info
  bool is_b_matrix = N != 1;

  auto& mat = is_b_matrix ? b : a;
  auto& vec = is_b_matrix ? a : b;
  bool transpose_mat = is_b_matrix ? !transpose_b : transpose_a;
  int in_vector_len = K;
  int out_vector_len = is_b_matrix ? N : M;

  int mat_ld = is_b_matrix ? ldb : lda;

  auto batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
  auto batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;

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

    if (K <= 64) {
      bm = 1;
      sm = 8;
      sn = 4;
    } else if (K >= 16 * out_vector_len) {
      bm = 1;
      bn = 8;
    }

    // Specialized kernel for very small outputs
    tm = out_vector_len < tm ? 1 : tm;

    n_out_per_tgp = bm * sm * tm;
    kname << "gemv_" << type_to_name(out);
  }

  const bool do_axpby = CHECK_AB && (alpha != 1.0f || beta != 0.0f);

  // clang-format off
  kname << "_bm" << bm << "_bn" << bn
        << "_sm" << sm << "_sn" << sn
        << "_tm" << tm << "_tn" << tn
        << "_nc" << !contiguous_kernel
        << "_axpby" << do_axpby; // clang-format on

  // Encode and dispatch kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder.set_compute_pipeline_state(kernel);

  int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
  MTL::Size group_dims = MTL::Size(32, bn, bm);
  MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

  compute_encoder.set_input_array(mat, 0);
  compute_encoder.set_input_array(vec, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder.set_bytes(in_vector_len, 4);
  compute_encoder.set_bytes(out_vector_len, 5);
  compute_encoder.set_bytes(mat_ld, 6);

  compute_encoder.set_bytes(batch_ndim, 9);
  compute_encoder.set_vector_bytes(batch_shape, 10);
  compute_encoder.set_vector_bytes(batch_strides_vec, 11);
  compute_encoder.set_vector_bytes(batch_strides_mat, 12);

  if (do_axpby) {
    compute_encoder.set_input_array(c, 2);

    compute_encoder.set_bytes(alpha, 7);
    compute_encoder.set_bytes(beta, 8);

    compute_encoder.set_vector_bytes(C_batch_stride, 13);

    int bias_stride = c.strides()[c.ndim() - 1];
    compute_encoder.set_bytes(bias_stride, 14);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

inline void gemv(
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
    Shape batch_shape = {},
    Strides A_batch_stride = {},
    Strides B_batch_stride = {}) {
  return gemv_axbpy<false>(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* const array& c = */ b,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ batch_shape,
      /* Strides A_batch_stride = */ A_batch_stride,
      /* Strides B_batch_stride = */ B_batch_stride);
}

///////////////////////////////////////////////////////////////////////////////
// Matmul implementation
///////////////////////////////////////////////////////////////////////////////

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  if (!issubdtype(out.dtype(), inexact)) {
    throw std::runtime_error("[matmul] dtype must be inexact.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero = array(0, a_pre.dtype());
    fill_gpu(zero, out, s);
    d.add_temporary(std::move(zero), s.index);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto [a_transposed, a_cols, a] = check_transpose(copies, s, a_pre, M == 1);
  auto [b_transposed, b_cols, b] = check_transpose(copies, s, b_pre, N == 1);

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
    return gemv(
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
        /* std::vector<array>& copies = */ copies,
        /* Shape batch_shape = */ std::move(batch_shape),
        /* Strides A_batch_stride = */ std::move(A_batch_stride),
        /* Strides B_batch_stride = */ std::move(B_batch_stride));
  }

  /////////////////////////////////////////////////////////////////////////////
  // Gemm specialization

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
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ std::move(batch_shape),
      /* Strides A_batch_stride = */ std::move(A_batch_stride),
      /* Strides B_batch_stride = */ std::move(B_batch_stride));
}

///////////////////////////////////////////////////////////////////////////////
// AddMM implementation
///////////////////////////////////////////////////////////////////////////////

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[matmul] Does not yet support non-floating point types.");
  }

  // Return 0s if either input is empty
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);

  // Handle empty matrix case (K=0)
  if (inputs[0].shape(-1) == 0) {
    auto& c = inputs[2];
    if (beta_ == 1.0f) {
      copy_gpu(
          c,
          out,
          c.flags().row_contiguous ? CopyType::Vector : CopyType::General,
          s);
    } else {
      array beta_scalar = array(beta_, c.dtype());
      binary_op_gpu({c, beta_scalar}, out, "Multiply", s);
      d.add_temporary(std::move(beta_scalar), s.index);
    }
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

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
  auto [transpose_a, a_cols, a] = check_transpose(copies, s, a_pre, M == 1);
  auto [transpose_b, b_cols, b] = check_transpose(copies, s, b_pre, N == 1);

  array c = c_pre;

  int lda = a_cols;
  int ldb = b_cols;

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions
  auto [batch_shape, A_batch_stride, B_batch_stride, C_batch_stride] =
      collapse_batches(a, b, c);

  int64_t matrix_stride_out = M * static_cast<int64_t>(N);
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
    return gemv_axbpy(
        /* const Stream& s = */ s,
        /* metal::Device& d = */ d,
        /* const array& a = */ a,
        /* const array& b = */ b,
        /* const array& c = */ c,
        /* array& out = */ out,
        /* int M = */ M,
        /* int N = */ N,
        /* int K = */ K,
        /* int batch_size_out = */ batch_size_out,
        /* int lda = */ lda,
        /* int ldb = */ ldb,
        /* bool transpose_a = */ transpose_a,
        /* bool transpose_b = */ transpose_b,
        /* std::vector<array>& copies = */ copies,
        /* Shape batch_shape = */ batch_shape,
        /* Strides A_batch_stride = */ A_batch_stride,
        /* Strides B_batch_stride = */ B_batch_stride,
        /* Strides C_batch_stride = */ C_batch_stride,
        /* float alpha = */ alpha_,
        /* float beta = */ beta_);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular addmm dispatch

  return steel_matmul_axpby(
      /* const Stream& s = */ s,
      /* metal::Device& d = */ d,
      /* const array& a = */ a,
      /* const array& b = */ b,
      /* const array& c = */ c,
      /* array& out = */ out,
      /* int M = */ M,
      /* int N = */ N,
      /* int K = */ K,
      /* int batch_size_out = */ batch_size_out,
      /* int lda = */ lda,
      /* int ldb = */ ldb,
      /* bool transpose_a = */ transpose_a,
      /* bool transpose_b = */ transpose_b,
      /* std::vector<array>& copies = */ copies,
      /* Shape batch_shape = */ batch_shape,
      /* Strides A_batch_stride = */ A_batch_stride,
      /* Strides B_batch_stride = */ B_batch_stride,
      /* Strides B_batch_stride = */ C_batch_stride,
      /* float alpha = */ alpha_,
      /* float beta = */ beta_);
}

///////////////////////////////////////////////////////////////////////////////
// BlockMaskedMM implementation
///////////////////////////////////////////////////////////////////////////////

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
    fill_gpu(zero, out, s);
    d.add_temporary(std::move(zero), s.index);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  std::vector<array> copies;
  auto [transpose_a, a_cols, a] = check_transpose(copies, s, a_pre, M == 1);
  auto [transpose_b, b_cols, b] = check_transpose(copies, s, b_pre, N == 1);

  int lda = a_cols;
  int ldb = b_cols;

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  bool has_op_mask = inputs.size() > 3;
  bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;

  // Prepare kernel name
  std::string out_mask_nm = has_out_mask ? type_to_name(inputs[2]) : "nomask";
  std::string op_mask_nm = has_op_mask ? type_to_name(inputs.back()) : "nomask";

  Shape batch_shape{1};
  Strides A_batch_stride{0};
  Strides B_batch_stride{0};
  Strides outmask_bstride{0};
  Strides Amask_bstride{0};
  Strides Bmask_bstride{0};
  int64_t A_batch_str = 0;
  int64_t B_batch_str = 0;

  Strides batch_strides;

  if (out.ndim() > 2) {
    Shape bshape{out.shape().begin(), out.shape().end() - 2};
    std::vector<Strides> bstrides;

    for (auto& arr : inputs) {
      bstrides.emplace_back(arr.strides().begin(), arr.strides().end() - 2);
    }

    // auto [bshape_c, bstrides_c] = collapse_contiguous_dims(bshape, bstrides);
    batch_shape = bshape;
    A_batch_str = bstrides[0].back();
    B_batch_str = bstrides[1].back();

    for (auto& bstr : bstrides) {
      batch_strides.insert(batch_strides.end(), bstr.begin(), bstr.end());
    }

    A_batch_stride = bstrides[0];
    B_batch_stride = bstrides[1];

    if (has_out_mask) {
      outmask_bstride = bstrides[2];
    }
    if (has_op_mask) {
      Amask_bstride = bstrides[has_out_mask + 2];
      Bmask_bstride = bstrides[has_out_mask + 3];
    }

  } else {
    batch_strides = Strides(inputs.size(), 0);
  }

  int64_t matrix_stride_out = static_cast<int64_t>(M) * N;
  size_t batch_size_out = out.size() / (matrix_stride_out);

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

    int mat_ld = is_b_matrix ? b_cols : a_cols;

    auto batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
    auto batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;

    auto mask_bstrides_mat = is_b_matrix ? Bmask_bstride : Amask_bstride;
    auto mask_bstrides_vec = is_b_matrix ? Amask_bstride : Bmask_bstride;

    auto mat_mask_idx = int(has_out_mask) + (is_b_matrix ? 3 : 2);
    auto vec_mask_idx = int(has_out_mask) + (is_b_matrix ? 2 : 3);

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
      sm = 8;
      sn = 4;
      bm = 1;
      bn = (block_size_ == 64 && out_vector_len >= 2048) ? 4 : 2;
      tm = block_size_ == 32 ? 4 : 8;
      tn = 4;

      // Specialized kernel for very small outputs
      tn = out_vector_len < tn ? 1 : tn;

      n_out_per_tgp = bn * sn * tn;
      kname << "gemv_t";

    } else {
      if (block_size_ == 32) {
        sm = 4;
        sn = 8;
        bm = 2;
      } else {
        sm = 2;
        sn = 16;
        bm = out_vector_len >= 512 ? 4 : 2;
      }

      // Specialized kernel for very small outputs
      tm = out_vector_len < tm ? 1 : tm;

      n_out_per_tgp = bm * sm * tm;
      kname << "gemv";
    }

    kname << "_outmask_" << out_mask_nm;
    kname << "_opmask_" << op_mask_nm;
    kname << "_" << type_to_name(out);
    kname << "_bm" << bm << "_bn" << bn;
    kname << "_sm" << sm << "_sn" << sn;
    kname << "_tm" << tm << "_tn" << tn;
    kname << "_nc" << !contiguous_kernel;

    // Encode and dispatch kernel
    auto kernel = get_gemv_masked_kernel(
        d,
        kname.str(),
        out,
        has_out_mask ? std::optional<array>{inputs[2]} : std::nullopt,
        has_op_mask ? std::optional<array>{inputs.back()} : std::nullopt,
        transpose_mat,
        bm,
        bn,
        sm,
        sn,
        tm,
        tn,
        contiguous_kernel);

    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
    MTL::Size group_dims = MTL::Size(32, bn, bm);
    MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

    // Get mask params
    std::vector<int> mask_strides;
    Strides mask_batch_strides;
    if (has_out_mask) {
      auto& out_mask = inputs[2];

      if (transpose_mat) {
        mask_strides.push_back(out_mask.strides(out.shape(-2) == 1 ? -1 : -2));
        mask_strides.push_back(out_mask.strides(out.shape(-2) == 1 ? -2 : -1));
      } else {
        mask_strides.push_back(out_mask.strides(out.shape(-1) == 1 ? -1 : -2));
        mask_strides.push_back(out_mask.strides(out.shape(-1) == 1 ? -2 : -1));
      }

      mask_batch_strides.insert(
          mask_batch_strides.end(),
          outmask_bstride.begin(),
          outmask_bstride.end());

      compute_encoder.set_input_array(out_mask, 20);
    }

    if (has_op_mask) {
      auto& mat_mask = inputs[mat_mask_idx];

      if (transpose_mat) {
        mask_strides.push_back(mat_mask.strides(!is_b_matrix ? -2 : -1));
        mask_strides.push_back(mat_mask.strides(!is_b_matrix ? -1 : -2));
      } else {
        mask_strides.push_back(mat_mask.strides(is_b_matrix ? -2 : -1));
        mask_strides.push_back(mat_mask.strides(is_b_matrix ? -1 : -2));
      }

      mask_batch_strides.insert(
          mask_batch_strides.end(),
          mask_bstrides_mat.begin(),
          mask_bstrides_mat.end());

      compute_encoder.set_input_array(mat_mask, 21);

      auto& vec_mask = inputs[vec_mask_idx];
      if (transpose_mat) {
        mask_strides.push_back(vec_mask.strides(vec.shape(-2) == 1 ? -1 : -2));
        mask_strides.push_back(vec_mask.strides(vec.shape(-2) == 1 ? -2 : -1));
      } else {
        mask_strides.push_back(vec_mask.strides(vec.shape(-1) == 1 ? -1 : -2));
        mask_strides.push_back(vec_mask.strides(vec.shape(-1) == 1 ? -2 : -1));
      }

      mask_batch_strides.insert(
          mask_batch_strides.end(),
          mask_bstrides_vec.begin(),
          mask_bstrides_vec.end());

      compute_encoder.set_input_array(vec_mask, 22);
    }

    // Get gemv params
    compute_encoder.set_input_array(mat, 0);
    compute_encoder.set_input_array(vec, 1);
    compute_encoder.set_output_array(out, 3);

    compute_encoder.set_bytes(in_vector_len, 4);
    compute_encoder.set_bytes(out_vector_len, 5);
    compute_encoder.set_bytes(mat_ld, 6);
    compute_encoder.set_bytes(batch_ndim, 9);
    compute_encoder.set_vector_bytes(batch_shape, 10);
    compute_encoder.set_vector_bytes(batch_strides_vec, 11);
    compute_encoder.set_vector_bytes(batch_strides_mat, 12);

    compute_encoder.set_vector_bytes(mask_strides, 23);
    compute_encoder.set_vector_bytes(mask_batch_strides, 24);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

    d.add_temporaries(std::move(copies), s.index);
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Regular kernel dispatch

  // Determine dispatch kernel
  int bm = block_size_, bn = block_size_, bk = 16;
  int wm = 2, wn = 2;
  bool mn_aligned = M % bm == 0 && N % bn == 0;
  bool k_aligned = K % bk == 0;

  std::ostringstream kname;
  kname << "steel_gemm_block_outmask_" << out_mask_nm << "_opmask_"
        << op_mask_nm << "_" << (transpose_a ? 't' : 'n')
        << (transpose_b ? 't' : 'n') << "_" << type_to_name(a) << "_"
        << type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
        << "_wm" << wm << "_wn" << wn << "_MN_" << (mn_aligned ? "t" : "n")
        << "aligned"
        << "_K_" << (k_aligned ? "t" : "n") << "aligned";

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
  compute_encoder.set_compute_pipeline_state(kernel);

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
      /* const int64_t batch_stride_a = */ A_batch_str,
      /* const int64_t batch_stride_b = */ B_batch_str,
      /* const int64_t batch_stride_d = */ matrix_stride_out,
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

  compute_encoder.set_bytes(params, 4);

  compute_encoder.set_vector_bytes(batch_shape, 6);
  compute_encoder.set_vector_bytes(batch_strides, 7);

  compute_encoder.set_vector_bytes(mask_strides, 13);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

///////////////////////////////////////////////////////////////////////////////
// GatherMM implementation
///////////////////////////////////////////////////////////////////////////////

void gather_mm_rhs(
    const array& a_,
    const array& b_,
    const array& indices_,
    array& out,
    metal::Device& d,
    const Stream& s) {
  array indices = ensure_row_contiguous(indices_, d, s);
  auto [transpose_b, ldb, b] = ensure_batch_contiguous(b_, d, s);

  // Broadcast a with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of a broadcasted
  // with rhs_indices. We need only broadcast a and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };
  array a = broadcast_with_indices(a_);

  // Extract the matmul shapes
  int K = a.shape(-1);
  int M = a.size() / K;
  int N = b.shape(-1);
  int lda = a.strides()[a.ndim() - 2]; // should be K

  // Define the dispatch blocks
  int bm = 16, bn = 64, bk = 16;
  int wm = 1, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Define the kernel name
  std::string base_name;
  base_name.reserve(64);
  concatenate(
      base_name,
      "steel_gather_mm_rhs_n",
      transpose_b ? 't' : 'n',
      '_',
      type_to_name(a),
      '_',
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      base_name,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_gather_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      false,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn,
      true);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare the matmul params
  auto batch_stride_b = b.ndim() > 2 ? b.strides()[b.ndim() - 3] : b.size();
  steel::GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ static_cast<int>(ldb),
      /* const int ldd = */ N,
      /* const int tiles_n = */ (N + bn - 1) / bn,
      /* const int tiles_m = */ (M + bm - 1) / bm,
      /* const int64_t batch_stride_a = */ 0,
      /* const int64_t batch_stride_b = */ static_cast<int64_t>(batch_stride_b),
      /* const int64_t batch_stride_d = */ 0,
      /* const int swizzle_log = */ 0,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ 0};

  // Prepare the grid
  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(params.tiles_n, params.tiles_m, 1);

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(indices, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(params, 4);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_mm_rhs_nax(
    const array& a_,
    const array& b_,
    const array& indices_,
    array& out,
    metal::Device& d,
    const Stream& s) {
  array indices = ensure_row_contiguous(indices_, d, s);
  auto [transpose_b, ldb, b] = ensure_batch_contiguous(b_, d, s);

  // Broadcast a with indices. If we are here that means lhs_indices were not
  // provided so the lhs_indices are implied to be the shape of a broadcasted
  // with rhs_indices. We need only broadcast a and copy it as if applying the
  // lhs_indices.
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };
  array a = broadcast_with_indices(a_);

  // Extract the matmul shapes
  int K = a.shape(-1);
  int M = a.size() / K;
  int N = b.shape(-1);
  int lda = a.strides()[a.ndim() - 2]; // should be K
  int E = b.shape(0);

  // Define the dispatch blocks
  int bm, bn = 128, bk = 128, wm, wn = 4;
  if (M / E > 48) {
    bm = 64;
    wm = 2;
  } else if (M / E > 24) {
    bm = 32l;
    wm = 1;
  } else {
    bm = 16;
    wm = 1;
  }

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Define the kernel name
  std::string base_name;
  base_name.reserve(64);
  concatenate(
      base_name,
      "steel_gather_mm_rhs_nax_n",
      transpose_b ? 't' : 'n',
      '_',
      type_to_name(a),
      '_',
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      base_name,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_gather_nax_kernel(
      d,
      base_name,
      hash_name,
      func_consts,
      out,
      false,
      transpose_b,
      bm,
      bn,
      bk,
      wm,
      wn,
      true);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare the matmul params
  auto batch_stride_b = b.ndim() > 2 ? b.strides()[b.ndim() - 3] : b.size();
  steel::GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ lda,
      /* const int ldb = */ static_cast<int>(ldb),
      /* const int ldd = */ N,
      /* const int tiles_n = */ (N + bn - 1) / bn,
      /* const int tiles_m = */ (M + bm - 1) / bm,
      /* const int64_t batch_stride_a = */ 0,
      /* const int64_t batch_stride_b = */ static_cast<int64_t>(batch_stride_b),
      /* const int64_t batch_stride_d = */ 0,
      /* const int swizzle_log = */ 0,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ 0};

  // Prepare the grid
  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(params.tiles_n, params.tiles_m, 1);

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(indices, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(params, 4);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_mv(
    const array& mat_,
    const array& vec_,
    const array& mat_indices_,
    const array& vec_indices_,
    array& out,
    int N,
    int K,
    bool is_mv,
    metal::Device& d,
    const Stream& s) {
  // Copy if needed
  std::vector<array> copies;
  auto [transpose_mat, mat_cols, mat] =
      check_transpose(copies, s, mat_, N == 1);
  auto [transpose_vec, vec_cols, vec] = check_transpose(copies, s, vec_, true);
  d.add_temporaries(std::move(copies), s.index);

  // If we are doing vector matrix instead of matrix vector we need to flip the
  // matrix transposition. Basically m @ v = v @ m.T assuming that v is treated
  // as a one dimensional array.
  transpose_mat = (!is_mv) ^ transpose_mat;

  // Define some shapes
  int in_vector_len = K;
  int out_vector_len = N;
  int mat_ld = mat_cols;

  int batch_size_out = out.size() / N;
  int batch_ndim = out.ndim() - 2;
  int batch_ndim_mat = mat.ndim() - 2;
  int batch_ndim_vec = vec.ndim() - 2;
  Strides index_strides = vec_indices_.strides();
  index_strides.insert(
      index_strides.end(),
      mat_indices_.strides().begin(),
      mat_indices_.strides().end());

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
  compute_encoder.set_compute_pipeline_state(kernel);

  int n_tgp = (out_vector_len + n_out_per_tgp - 1) / n_out_per_tgp;
  MTL::Size group_dims = MTL::Size(32, bn, bm);
  MTL::Size grid_dims = MTL::Size(n_tgp, 1, batch_size_out);

  compute_encoder.set_input_array(mat, 0);
  compute_encoder.set_input_array(vec, 1);
  compute_encoder.set_output_array(out, 3);

  compute_encoder.set_bytes(in_vector_len, 4);
  compute_encoder.set_bytes(out_vector_len, 5);
  compute_encoder.set_bytes(mat_ld, 6);

  compute_encoder.set_bytes(batch_ndim, 9);
  compute_encoder.set_vector_bytes(out.shape(), 10);
  compute_encoder.set_vector_bytes(index_strides, 11);

  compute_encoder.set_bytes(batch_ndim_vec, 12);
  compute_encoder.set_vector_bytes(vec.shape(), 13);
  compute_encoder.set_vector_bytes(vec.strides(), 14);

  compute_encoder.set_bytes(batch_ndim_mat, 15);
  compute_encoder.set_vector_bytes(mat.shape(), 16);
  compute_encoder.set_vector_bytes(mat.strides(), 17);

  compute_encoder.set_input_array(vec_indices_, 18);
  compute_encoder.set_input_array(mat_indices_, 19);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_mm(
    const array& a_,
    const array& b_,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  // Copy if needed
  std::vector<array> copies;
  auto [transpose_a, lda, a] = check_transpose(copies, s, a_, false);
  auto [transpose_b, ldb, b] = check_transpose(copies, s, b_, false);
  d.add_temporaries(std::move(copies), s.index);

  // Determine dispatch kernel
  int bm = 64, bn = 64, bk = 16;
  int wm = 2, wn = 2;
  size_t batch_size_out = out.size() / M / N;
  int batch_ndim = out.ndim() - 2;
  int batch_ndim_a = a.ndim() - 2;
  int batch_ndim_b = b.ndim() - 2;

  char devc = d.get_architecture().back();
  GEMM_TPARAM_MACRO(devc)

  const bool has_batch = batch_ndim > 1;
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Define the kernel name
  std::string base_name;
  base_name.reserve(128);
  concatenate(
      base_name,
      "steel_gather_mm_",
      transpose_a ? 't' : 'n',
      transpose_b ? 't' : 'n',
      "_",
      type_to_name(a),
      "_",
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn);

  metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      base_name,
      "_has_batch_",
      has_batch ? 't' : 'n',
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_gather_kernel(
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
      wn,
      false);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare the matmul params
  steel::GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ static_cast<int>(lda),
      /* const int ldb = */ static_cast<int>(ldb),
      /* const int ldd = */ N,
      /* const int tiles_n = */ (N + bn - 1) / bn,
      /* const int tiles_m = */ (M + bm - 1) / bm,
      /* const int64_t batch_stride_a = */
      (batch_ndim > 0) ? lhs_indices.strides()[0] : 0,
      /* const int64_t batch_stride_b = */
      (batch_ndim > 0) ? rhs_indices.strides()[0] : 0,
      /* const int64_t batch_stride_d = */ M * N,
      /* const int swizzle_log = */ 0,
      /* const int gemm_k_iterations_aligned = */ (K / bk),
      /* const int batch_ndim = */ batch_ndim};

  // Prepare the grid
  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims =
      MTL::Size(params.tiles_n, params.tiles_m, batch_size_out);

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(lhs_indices, 2);
  compute_encoder.set_input_array(rhs_indices, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(params, 5);
  compute_encoder.set_vector_bytes(lhs_indices.shape(), 6);
  compute_encoder.set_vector_bytes(lhs_indices.strides(), 7);
  compute_encoder.set_vector_bytes(rhs_indices.strides(), 8);
  compute_encoder.set_bytes(batch_ndim_a, 9);
  compute_encoder.set_vector_bytes(a.shape(), 10);
  compute_encoder.set_vector_bytes(a.strides(), 11);
  compute_encoder.set_bytes(batch_ndim_b, 12);
  compute_encoder.set_vector_bytes(b.shape(), 13);
  compute_encoder.set_vector_bytes(b.strides(), 14);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  // Return 0s if either input is empty
  if (a.size() == 0 || b.size() == 0) {
    array zero = array(0, a.dtype());
    fill_gpu(zero, out, s);
    d.add_temporary(std::move(zero), s.index);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  // We are walking a in order and b is also in order so we can batch up the
  // matmuls and reuse reading a and b.
  if (M == 1 && right_sorted_ == true) {
    if (metal::is_nax_available() &&
        (env::enable_tf32() || a.dtype() != float32)) {
      return gather_mm_rhs_nax(a, b, rhs_indices, out, d, s);
    }
    gather_mm_rhs(a, b, rhs_indices, out, d, s);
    return;
  }

  // Route to gather gemv if any of a or b are vectors
  if (M == 1) {
    gather_mv(b, a, rhs_indices, lhs_indices, out, N, K, false, d, s);
    return;
  }
  if (N == 1) {
    gather_mv(a, b, lhs_indices, rhs_indices, out, M, K, true, d, s);
    return;
  }

  // Route to non specialized gather mm
  gather_mm(a, b, lhs_indices, rhs_indices, out, M, N, K, d, s);
}

void segmented_mm(
    const array& a_,
    const array& b_,
    const array& segments_,
    array& out,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s) {
  auto check_segments_layout = [&d, &s](const array& x) {
    // Contiguous so return early
    if (x.flags().row_contiguous) {
      return std::make_tuple(true, x);
    }

    bool rc = true;
    for (int i = 0; i < x.ndim() - 2; i++) {
      rc &=
          (x.strides(i + 1) * x.shape(i) == x.strides(i)) || (x.shape(i) == 1);
    }
    rc &= x.strides(x.ndim() - 1) == 1;
    if (x.ndim() > 1) {
      rc &= x.strides(x.ndim() - 2) == 1;
    }

    if (rc) {
      return std::make_tuple(false, x);
    }

    array x_copy = contiguous_copy_gpu(x, s);
    d.add_temporary(x_copy, s.index);
    return std::make_tuple(true, x_copy);
  };

  // Copy if needed
  std::vector<array> copies;
  auto [transpose_a, lda, a] = check_transpose(copies, s, a_, false);
  auto [transpose_b, ldb, b] = check_transpose(copies, s, b_, false);
  auto [segments_contiguous, segments] = check_segments_layout(segments_);
  d.add_temporaries(std::move(copies), s.index);

  // Determine dispatch kernel
  int bm = 64, bn = 64, bk = 16;
  int wm = 2, wn = 2;
  size_t batch_size_out = out.size() / M / N;

  char devc = d.get_architecture().back();
  GEMM_TPARAM_MACRO(devc)

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;

  // Define the kernel name
  std::string base_name;
  base_name.reserve(128);
  concatenate(
      base_name,
      "steel_segmented_mm_",
      transpose_a ? 't' : 'n',
      transpose_b ? 't' : 'n',
      "_",
      type_to_name(a),
      "_",
      type_to_name(out),
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn);

  metal::MTLFCList func_consts = {
      {&segments_contiguous, MTL::DataType::DataTypeBool, 199},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      base_name,
      "_segments_contiguous_",
      segments_contiguous ? 't' : 'n',
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = get_steel_gemm_segmented_kernel(
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
  compute_encoder.set_compute_pipeline_state(kernel);

  // Prepare the matmul params
  steel::GEMMParams params{
      /* const int M = */ M,
      /* const int N = */ N,
      /* const int K = */ K,
      /* const int lda = */ static_cast<int>(lda),
      /* const int ldb = */ static_cast<int>(ldb),
      /* const int ldd = */ N,
      /* const int tiles_n = */ (N + bn - 1) / bn,
      /* const int tiles_m = */ (M + bm - 1) / bm,
      /* const int64_t batch_stride_a = */ 0,
      /* const int64_t batch_stride_b = */ 0,
      /* const int64_t batch_stride_d = */ M * N,
      /* const int swizzle_log = */ 0,
      /* const int gemm_k_iterations_aligned = */ 0,
      /* const int batch_ndim = */ 0};

  // Prepare the grid
  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims =
      MTL::Size(params.tiles_n, params.tiles_m, batch_size_out);

  // Launch kernel
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(segments, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(params, 4);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& segments = inputs[2];

  out.set_data(allocator::malloc(out.nbytes()));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  segmented_mm(a, b, segments, out, M, N, K, d, s);
}

} // namespace mlx::core
