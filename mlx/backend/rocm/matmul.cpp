// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/primitives.h"
#include "mlx/types/half_types.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstring>
#include <numeric>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(rocm::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

void gemm_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  auto& device = encoder.device();
  rocblas_handle handle = device.get_rocblas_handle();

  // rocBLAS uses column-major, so we swap A and B and compute B^T * A^T = (A *
  // B)^T But since we want row-major output, we compute C = A * B by doing C^T
  // = B^T * A^T
  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_none : rocblas_operation_transpose;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_none : rocblas_operation_transpose;

  encoder.launch_kernel([&](hipStream_t stream) {
    rocblas_set_stream(handle, stream);

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        rocblas_sgemm(
            handle,
            trans_a,
            trans_b,
            N, // m (rows of op(B))
            M, // n (cols of op(A))
            K, // k
            &alpha_f,
            b.data<float>(),
            b_transposed ? K : N, // lda for B
            a.data<float>(),
            a_transposed ? M : K, // ldb for A
            &beta_f,
            out.data<float>(),
            N); // ldc
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        rocblas_dgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_d,
            b.data<double>(),
            b_transposed ? K : N,
            a.data<double>(),
            a_transposed ? M : K,
            &beta_d,
            out.data<double>(),
            N);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        // Convert float to rocblas_half using memcpy
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(b.data<float16_t>()),
            b_transposed ? K : N,
            reinterpret_cast<const rocblas_half*>(a.data<float16_t>()),
            a_transposed ? M : K,
            &beta_h,
            reinterpret_cast<rocblas_half*>(out.data<float16_t>()),
            N);
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for matmul on ROCm");
    }
  });
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Check batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);
  auto batch_count = out.size() / (M * N);

  if (batch_count == 1) {
    // Simple single GEMM
    gemm_rocblas(
        encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
  } else {
    // Batched GEMM - for now, loop over batches
    // TODO: Use rocblas_sgemm_strided_batched for better performance
    for (int64_t batch = 0; batch < batch_count; ++batch) {
      // Calculate offsets
      int64_t a_offset = 0, b_offset = 0;
      int64_t batch_idx = batch;
      for (int i = batch_shape.size() - 1; i >= 0; --i) {
        int64_t idx = batch_idx % batch_shape[i];
        batch_idx /= batch_shape[i];
        a_offset += idx * a_batch_strides[i];
        b_offset += idx * b_batch_strides[i];
      }

      // Create views for this batch
      // For simplicity, we use pointer arithmetic in the kernel
      encoder.launch_kernel([&, a_offset, b_offset, batch](hipStream_t stream) {
        auto& device = encoder.device();
        rocblas_handle handle = device.get_rocblas_handle();
        rocblas_set_stream(handle, stream);

        rocblas_operation trans_a =
            b_transposed ? rocblas_operation_none : rocblas_operation_transpose;
        rocblas_operation trans_b =
            a_transposed ? rocblas_operation_none : rocblas_operation_transpose;

        float alpha = 1.0f, beta = 0.0f;

        if (a.dtype() == float32) {
          rocblas_sgemm(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha,
              b.data<float>() + b_offset,
              b_transposed ? K : N,
              a.data<float>() + a_offset,
              a_transposed ? M : K,
              &beta,
              out.data<float>() + batch * M * N,
              N);
        }
      });
    }
  }
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Copy C into out first, then do GEMM with beta
  copy_gpu(c, out, CopyType::General, s);

  // Do GEMM with alpha and beta
  gemm_rocblas(
      encoder,
      M,
      N,
      K,
      a_transposed,
      lda,
      b_transposed,
      ldb,
      out,
      a,
      b,
      alpha_,
      beta_);
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(allocator::malloc(out.nbytes()));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto [transposed_a, lda, a_] = check_transpose(encoder, s, a);
  auto [transposed_b, ldb, b_] = check_transpose(encoder, s, b);
  
  auto use_gemv = can_use_gemv(M, N, K, transposed_a, transposed_b);
  
  if (M == 1 && use_gemv) {
    gather_mv(b_, a_, rhs_indices, lhs_indices, out, N, K, encoder);
    return;
  }

  if (N == 1 && use_gemv) {
    gather_mv(a_, b_, lhs_indices, rhs_indices, out, M, K, encoder);
    return;
  }

  // Fallback: loop over batches
  int batch_size = lhs_indices.size();
  for (int i = 0; i < batch_size; ++i) {
    // For now, use CPU to get indices and dispatch individual GEMMs
    // This is not optimal but provides correctness
    throw std::runtime_error(
        "GatherMM with M > 1 and N > 1 not yet optimized for ROCm. "
        "Consider using GEMV path (M=1 or N=1).");
  }
}

} // namespace mlx::core
