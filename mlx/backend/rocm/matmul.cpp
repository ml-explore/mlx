// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/gemms/gemv.h"
#include "mlx/backend/rocm/gemms/hipblaslt_gemm.h"
#include "mlx/backend/rocm/gemms/naive_gemm.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/utils.h"
#include "mlx/primitives.h"
#include "mlx/types/half_types.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <cstring>
#include <numeric>
#include <vector>

namespace mlx::core {

namespace {

// Async MoE segment launch job (filled on main thread, consumed in host-func).
struct MoeAsyncJob {
  uint32_t* pin{nullptr};
  int batch{0};
  int N{0};
  int K{0};
  bool b_transposed{false};
  int64_t ldb{0};
  size_t esz{0};
  const char* aB{nullptr};
  const char* bB{nullptr};
  char* cB{nullptr};
  int b_ndim{0};
  int64_t b_shape[4]{};
  int64_t b_strides[4]{};
  Dtype dtype{float32};
  int device_id{0};
  hipStream_t stream{nullptr};
};

static void moe_async_host_cb(void* user) {
  auto* j = static_cast<MoeAsyncJob*>(user);
  auto expert_off = [&](uint32_t expert) -> int64_t {
    if (j->b_ndim <= 0)
      return 0;
    if (j->b_ndim == 1)
      return static_cast<int64_t>(expert) * j->b_strides[0];
    int64_t off = 0;
    int64_t idx = static_cast<int64_t>(expert);
    for (int d = j->b_ndim - 1; d >= 0; --d) {
      int64_t coord = idx % j->b_shape[d];
      idx /= j->b_shape[d];
      off += coord * j->b_strides[d];
    }
    return off;
  };

  int start = 0;
  while (start < j->batch) {
    uint32_t e = j->pin[static_cast<size_t>(start)];
    int end = start + 1;
    while (end < j->batch && j->pin[static_cast<size_t>(end)] == e)
      ++end;
    int Mseg = end - start;
    if (Mseg > 0) {
      try {
        rocm::hipblaslt_gemm_rowmajor_on_stream(
            j->stream,
            j->device_id,
            /*transpose_a=*/false,
            j->b_transposed,
            Mseg,
            j->N,
            j->K,
            1.0f,
            j->aB + static_cast<size_t>(start) * j->K * j->esz,
            /*lda=*/j->K,
            j->bB + static_cast<size_t>(expert_off(e)) * j->esz,
            static_cast<int>(j->ldb),
            0.0f,
            j->cB + static_cast<size_t>(start) * j->N * j->esz,
            /*ldc=*/j->N,
            j->dtype);
      } catch (...) {
        // Host callbacks must not throw into the HIP runtime.
      }
    }
    start = end;
  }
  (void)hipHostFree(j->pin);
  delete j;
}

std::tuple<bool, int64_t, array>
check_transpose(rocm::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  // Contiguous last-two dims: hipBLASLt/rocBLAS can take strided batch views
  // without packing. Avoiding contiguous_copy_gpu here is the main GEMM-side
  // "don't force a copy before the fused library GEMM" win on ROCm.
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else if (
      sty == 1 && stx > 0 &&
      static_cast<int64_t>(stx) >= static_cast<int64_t>(arr.shape(-1))) {
    // Row-major last dim with padded leading dimension (common after slice /
    // broadcast expand). Still a valid GEMM without materialize.
    return std::make_tuple(false, stx, arr);
  } else if (
      stx == 1 && sty > 0 &&
      static_cast<int64_t>(sty) >= static_cast<int64_t>(arr.shape(-2))) {
    // Column-major last-two with padded ldb.
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

std::tuple<bool, int64_t, array> ensure_batch_contiguous(
    const array& x,
    rocm::CommandEncoder& encoder,
    Stream s) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides(-2), x);
  }

  bool rc = true;
  for (int i = 0; i < x.ndim() - 3; i++) {
    rc &= (x.strides(i + 1) * x.shape(i)) == x.strides(i);
  }
  if (rc) {
    return check_transpose(encoder, s, x);
  }

  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return std::make_tuple(false, x_copy.strides(-2), x_copy);
}

std::pair<bool, int64_t> get_uniform_batch_stride(
    const Shape& batch_shape,
    const Strides& batch_strides) {
  if (batch_shape.empty() || batch_shape.size() != batch_strides.size()) {
    return {false, 0};
  }

  if (batch_shape.size() == 1) {
    return {true, batch_strides.back()};
  }

  for (int i = batch_shape.size() - 2; i >= 0; --i) {
    int64_t cur = batch_strides[i];
    int64_t next = batch_strides[i + 1];
    if (cur == 0 && next == 0) {
      continue;
    }
    if (cur != next * batch_shape[i + 1]) {
      return {false, 0};
    }
  }

  return {true, batch_strides.back()};
}

int parse_non_negative_int_env(const char* env_name, int default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }

  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0) {
    return default_value;
  }
  return static_cast<int>(value);
}

int gemm_solution_index_f32(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_F32_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_F32_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
}

int gemm_solution_index_bf16(bool batched) {
  static int single_index =
      parse_non_negative_int_env("MLX_ROCM_GEMM_BF16_SOLUTION_INDEX", 0);
  static int batched_index = parse_non_negative_int_env(
      "MLX_ROCM_GEMM_BF16_BATCHED_SOLUTION_INDEX", -1);
  if (!batched) {
    return single_index;
  }
  return batched_index >= 0 ? batched_index : single_index;
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
  // Try hipBLASLt for bf16/fp16 GEMMs -- it often picks faster kernels than
  // rocBLAS for half-precision on RDNA 3/3.5/4 and CDNA GPUs.
  if ((a.dtype() == bfloat16 || a.dtype() == float16) &&
      rocm::is_hipblaslt_available()) {
    try {
      rocm::hipblaslt_gemm(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          K,
          alpha,
          a,
          lda,
          b,
          ldb,
          beta,
          out,
          N, // ldc = N for row-major output
          a.dtype());
      return;
    } catch (...) {
      // hipBLASLt failed (unsupported config, etc.) -- fall through to rocBLAS.
    }
  }

  auto& device = encoder.device();
  rocblas_handle handle = device.get_rocblas_handle();

  // rocBLAS uses column-major, so we swap A and B and compute B^T * A^T = (A *
  // B)^T But since we want row-major output, we compute C = A * B by doing C^T
  // = B^T * A^T
  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_transpose : rocblas_operation_none;

  // We pass B then A (swapped) to compute C^T = B^T * A^T. The leading
  // dimensions come directly from check_transpose() for each operand.
  const int64_t ld_b = ldb;
  const int64_t ld_a = lda;
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);

  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(false);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ld_b,
              a_ptr,
              rocblas_datatype_f32_r,
              ld_a,
              &beta_f,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm(
                handle,
                trans_a,
                trans_b,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ld_b,
                static_cast<const float*>(a_ptr),
                ld_a,
                &beta_f,
                static_cast<float*>(out_ptr),
                N);
          }
        } else {
          rocblas_sgemm(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ld_b,
              static_cast<const float*>(a_ptr),
              ld_a,
              &beta_f,
              static_cast<float*>(out_ptr),
              N);
        }
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
            static_cast<const double*>(b_ptr),
            ld_b,
            static_cast<const double*>(a_ptr),
            ld_a,
            &beta_d,
            static_cast<double*>(out_ptr),
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
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(b_ptr)),
            ld_b,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(a_ptr)),
            ld_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<float16_t*>(out_ptr)),
            N);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_bf16(false);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_ex(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            static_cast<const bfloat16_t*>(b_ptr),
            rocblas_datatype_bf16_r,
            ld_b,
            static_cast<const bfloat16_t*>(a_ptr),
            rocblas_datatype_bf16_r,
            ld_a,
            &beta_f,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);
        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const bfloat16_t*>(b_ptr),
              rocblas_datatype_bf16_r,
              ld_b,
              static_cast<const bfloat16_t*>(a_ptr),
              rocblas_datatype_bf16_r,
              ld_a,
              &beta_f,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported dtype for matmul on ROCm");
    }
  });
}

void gemm_strided_batched_rocblas(
    rocm::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    array& out,
    const array& a,
    const array& b,
    float alpha = 1.0f,
    float beta = 0.0f) {
  // Try hipBLASLt for bf16/fp16 batched GEMMs.
  if ((a.dtype() == bfloat16 || a.dtype() == float16) &&
      rocm::is_hipblaslt_available()) {
    try {
      rocm::hipblaslt_gemm_batched(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          K,
          alpha,
          a,
          lda,
          stride_a,
          b,
          ldb,
          stride_b,
          beta,
          out,
          N, // ldc = N for row-major output
          stride_c,
          batch_count,
          a.dtype());
      return;
    } catch (...) {
      // hipBLASLt failed -- fall through to rocBLAS.
    }
  }

  auto& device = encoder.device();
  rocblas_handle handle = device.get_rocblas_handle();

  rocblas_operation trans_a =
      b_transposed ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation trans_b =
      a_transposed ? rocblas_operation_transpose : rocblas_operation_none;

  const int64_t ld_b = ldb;
  const int64_t ld_a = lda;
  const void* a_ptr = gpu_ptr<void>(a);
  const void* b_ptr = gpu_ptr<void>(b);
  void* out_ptr = gpu_ptr<void>(out);

  encoder.launch_kernel([&, a_ptr, b_ptr, out_ptr](hipStream_t stream) {
    encoder.device().set_rocblas_stream(stream);

    switch (a.dtype()) {
      case float32: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_f32(true);
        static std::atomic<bool> solution_valid{true};

        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          rocblas_status status = rocblas_gemm_strided_batched_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              b_ptr,
              rocblas_datatype_f32_r,
              ld_b,
              stride_b,
              a_ptr,
              rocblas_datatype_f32_r,
              ld_a,
              stride_a,
              &beta_f,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              stride_c,
              out_ptr,
              rocblas_datatype_f32_r,
              N,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index,
              solution_index,
              0);
          if (status != rocblas_status_success) {
            solution_valid.store(false, std::memory_order_relaxed);
            rocblas_sgemm_strided_batched(
                handle,
                trans_a,
                trans_b,
                N,
                M,
                K,
                &alpha_f,
                static_cast<const float*>(b_ptr),
                ld_b,
                stride_b,
                static_cast<const float*>(a_ptr),
                ld_a,
                stride_a,
                &beta_f,
                static_cast<float*>(out_ptr),
                N,
                stride_c,
                batch_count);
          }
        } else {
          rocblas_sgemm_strided_batched(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const float*>(b_ptr),
              ld_b,
              stride_b,
              static_cast<const float*>(a_ptr),
              ld_a,
              stride_a,
              &beta_f,
              static_cast<float*>(out_ptr),
              N,
              stride_c,
              batch_count);
        }
        break;
      }
      case float64: {
        double alpha_d = static_cast<double>(alpha);
        double beta_d = static_cast<double>(beta);
        rocblas_dgemm_strided_batched(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_d,
            static_cast<const double*>(b_ptr),
            ld_b,
            stride_b,
            static_cast<const double*>(a_ptr),
            ld_a,
            stride_a,
            &beta_d,
            static_cast<double*>(out_ptr),
            N,
            stride_c,
            batch_count);
        break;
      }
      case float16: {
        rocblas_half alpha_h, beta_h;
        float16_t alpha_f16 = static_cast<float16_t>(alpha);
        float16_t beta_f16 = static_cast<float16_t>(beta);
        std::memcpy(&alpha_h, &alpha_f16, sizeof(rocblas_half));
        std::memcpy(&beta_h, &beta_f16, sizeof(rocblas_half));
        rocblas_hgemm_strided_batched(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_h,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(b_ptr)),
            ld_b,
            stride_b,
            reinterpret_cast<const rocblas_half*>(
                static_cast<const float16_t*>(a_ptr)),
            ld_a,
            stride_a,
            &beta_h,
            reinterpret_cast<rocblas_half*>(static_cast<float16_t*>(out_ptr)),
            N,
            stride_c,
            batch_count);
        break;
      }
      case bfloat16: {
        float alpha_f = alpha;
        float beta_f = beta;
        int solution_index = gemm_solution_index_bf16(true);
        static std::atomic<bool> solution_valid{true};

        rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
        if (solution_index > 0 &&
            solution_valid.load(std::memory_order_relaxed)) {
          algo = rocblas_gemm_algo_solution_index;
        } else {
          solution_index = 0;
        }

        rocblas_status status = rocblas_gemm_strided_batched_ex(
            handle,
            trans_a,
            trans_b,
            N,
            M,
            K,
            &alpha_f,
            static_cast<const bfloat16_t*>(b_ptr),
            rocblas_datatype_bf16_r,
            ld_b,
            stride_b,
            static_cast<const bfloat16_t*>(a_ptr),
            rocblas_datatype_bf16_r,
            ld_a,
            stride_a,
            &beta_f,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            stride_c,
            static_cast<bfloat16_t*>(out_ptr),
            rocblas_datatype_bf16_r,
            N,
            stride_c,
            batch_count,
            rocblas_datatype_f32_r,
            algo,
            solution_index,
            0);
        if (status != rocblas_status_success &&
            algo == rocblas_gemm_algo_solution_index) {
          solution_valid.store(false, std::memory_order_relaxed);
          rocblas_gemm_strided_batched_ex(
              handle,
              trans_a,
              trans_b,
              N,
              M,
              K,
              &alpha_f,
              static_cast<const bfloat16_t*>(b_ptr),
              rocblas_datatype_bf16_r,
              ld_b,
              stride_b,
              static_cast<const bfloat16_t*>(a_ptr),
              rocblas_datatype_bf16_r,
              ld_a,
              stride_a,
              &beta_f,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              stride_c,
              static_cast<bfloat16_t*>(out_ptr),
              rocblas_datatype_bf16_r,
              N,
              stride_c,
              batch_count,
              rocblas_datatype_f32_r,
              rocblas_gemm_algo_standard,
              0,
              0);
        }
        break;
      }
      default:
        throw std::runtime_error(
            "Unsupported dtype for batched matmul on ROCm");
    }
  });
}

void gemm_and_bias(
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
  // Check and collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  // Use GEMV when possible
  if (rocm::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    rocm::gemv(
        a,
        b,
        out,
        M,
        N,
        K,
        batch_count,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        encoder);
    return;
  }

  // Check if rocBLAS is available
  bool use_rocblas = encoder.device().is_rocblas_available();
  auto [a_uniform_batch, a_uniform_stride] =
      get_uniform_batch_stride(batch_shape, a_batch_strides);
  auto [b_uniform_batch, b_uniform_stride] =
      get_uniform_batch_stride(batch_shape, b_batch_strides);

  if (batch_count == 1) {
    // Simple single GEMM
    if (use_rocblas) {
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
          alpha,
          beta);
    } else {
      // Use naive GEMM fallback
      rocm::naive_gemm(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          b_transposed,
          ldb,
          alpha,
          beta);
    }
  } else if (a_uniform_batch && b_uniform_batch) {
    // Use strided batched GEMM for uniform batches
    if (use_rocblas) {
      gemm_strided_batched_rocblas(
          encoder,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_uniform_stride,
          b_transposed,
          ldb,
          b_uniform_stride,
          M * N,
          batch_count,
          out,
          a,
          b,
          alpha,
          beta);
    } else {
      // Use naive batched GEMM fallback
      rocm::naive_gemm_batched(
          encoder,
          a,
          b,
          out,
          M,
          N,
          K,
          a_transposed,
          lda,
          a_uniform_stride,
          b_transposed,
          ldb,
          b_uniform_stride,
          M * N,
          batch_count,
          alpha,
          beta);
    }
  } else {
    // Fallback: loop over batches for non-uniform strides
    if (use_rocblas) {
      const void* a_ptr_base = gpu_ptr<void>(a);
      const void* b_ptr_base = gpu_ptr<void>(b);
      void* out_ptr_base = gpu_ptr<void>(out);
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        encoder.launch_kernel([&,
                               a_offset,
                               b_offset,
                               batch,
                               a_ptr_base,
                               b_ptr_base,
                               out_ptr_base](hipStream_t stream) {
          auto& device = encoder.device();
          device.set_rocblas_stream(stream);
          rocblas_handle handle = device.get_rocblas_handle();

          rocblas_operation trans_a = b_transposed ? rocblas_operation_transpose
                                                   : rocblas_operation_none;
          rocblas_operation trans_b = a_transposed ? rocblas_operation_transpose
                                                   : rocblas_operation_none;

          const int64_t ld_b = ldb;
          const int64_t ld_a = lda;

          switch (a.dtype()) {
            case float32: {
              float alpha_f = alpha, beta_f = beta;
              rocblas_sgemm(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_f,
                  static_cast<const float*>(b_ptr_base) + b_offset,
                  ld_b,
                  static_cast<const float*>(a_ptr_base) + a_offset,
                  ld_a,
                  &beta_f,
                  static_cast<float*>(out_ptr_base) + batch * M * N,
                  N);
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
                  static_cast<const double*>(b_ptr_base) + b_offset,
                  ld_b,
                  static_cast<const double*>(a_ptr_base) + a_offset,
                  ld_a,
                  &beta_d,
                  static_cast<double*>(out_ptr_base) + batch * M * N,
                  N);
              break;
            }
            case float16: {
              rocblas_half alpha_h, beta_h;
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
                  reinterpret_cast<const rocblas_half*>(
                      static_cast<const float16_t*>(b_ptr_base) + b_offset),
                  ld_b,
                  reinterpret_cast<const rocblas_half*>(
                      static_cast<const float16_t*>(a_ptr_base) + a_offset),
                  ld_a,
                  &beta_h,
                  reinterpret_cast<rocblas_half*>(
                      static_cast<float16_t*>(out_ptr_base) + batch * M * N),
                  N);
              break;
            }
            case bfloat16: {
              float alpha_f = alpha;
              float beta_f = beta;
              auto* out_ptr =
                  static_cast<bfloat16_t*>(out_ptr_base) + batch * M * N;
              rocblas_gemm_ex(
                  handle,
                  trans_a,
                  trans_b,
                  N,
                  M,
                  K,
                  &alpha_f,
                  static_cast<const bfloat16_t*>(b_ptr_base) + b_offset,
                  rocblas_datatype_bf16_r,
                  ld_b,
                  static_cast<const bfloat16_t*>(a_ptr_base) + a_offset,
                  rocblas_datatype_bf16_r,
                  ld_a,
                  &beta_f,
                  out_ptr,
                  rocblas_datatype_bf16_r,
                  N,
                  out_ptr,
                  rocblas_datatype_bf16_r,
                  N,
                  rocblas_datatype_f32_r,
                  rocblas_gemm_algo_standard,
                  0,
                  0);
              break;
            }
            default:
              throw std::runtime_error(
                  "Unsupported dtype for non-uniform batched matmul on ROCm");
          }
        });
      }
    } else {
      // Use naive GEMM for each batch when rocBLAS is not available
      // This is less efficient but provides correctness
      for (int64_t batch = 0; batch < batch_count; ++batch) {
        int64_t a_offset = 0, b_offset = 0;
        int64_t batch_idx = batch;
        for (int i = batch_shape.size() - 1; i >= 0; --i) {
          int64_t idx = batch_idx % batch_shape[i];
          batch_idx /= batch_shape[i];
          a_offset += idx * a_batch_strides[i];
          b_offset += idx * b_batch_strides[i];
        }

        // Use naive GEMM with explicit offsets
        rocm::naive_gemm_with_offset(
            encoder,
            a,
            b,
            out,
            M,
            N,
            K,
            a_transposed,
            lda,
            a_offset,
            b_transposed,
            ldb,
            b_offset,
            batch * M * N,
            alpha,
            beta);
      }
    }
  }
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

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
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

  // Fused Linear: alpha*(A@B) + bias when C is a 1-D bias of length N.
  // hipBLASLt EPILOGUE_BIAS collapses the separate bias-add kernel that Metal
  // often fuses via epilogue / graph fusion. GELU/Swish act epilogues are
  // available via hipblaslt_gemm_epilogue API (act-only or *_BIAS combos) for
  // future graph pattern-matching; Linear bias is the hot path today.
  // Opt-out: MLX_ROCM_NO_HIPBLASLT_EPILOGUE=1.
  const bool bias_vec = c.ndim() == 1 && static_cast<int>(c.size()) == N &&
      c.dtype() == a.dtype() && alpha_ == 1.0f && beta_ == 1.0f;
  if (bias_vec && (a.dtype() == bfloat16 || a.dtype() == float16) &&
      rocm::is_hipblaslt_available()) {
    try {
      out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
      encoder.set_input_array(a);
      encoder.set_input_array(b);
      encoder.set_input_array(c);
      encoder.set_output_array(out);
      // HIPBLASLT_EPILOGUE_BIAS == 4 (hipblaslt.h); keep numeric to avoid
      // pulling the hipBLASLt header into matmul.cpp.
      constexpr int kEpilogueBias = 4;
      rocm::hipblaslt_gemm_epilogue(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          K,
          /*alpha=*/1.0f,
          a,
          lda,
          b,
          ldb,
          /*beta=*/0.0f,
          out,
          N,
          a.dtype(),
          &c,
          kEpilogueBias);
      return;
    } catch (...) {
      // Fall through to copy-C + GEMM path.
    }
  }

  // Copy C into out only when beta uses it.
  if (beta_ != 0.0f) {
    copy_gpu(c, out, CopyType::General, s);
  } else {
    out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));
  }

  // Check if rocBLAS is available
  if (encoder.device().is_rocblas_available()) {
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
  } else {
    // Use naive GEMM fallback
    rocm::naive_gemm(
        encoder,
        a,
        b,
        out,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        alpha_,
        beta_);
  }
}

// MoE-style gather_mm (M==1): collapse consecutive tokens that share an expert
// into one LDS-tiled GEMM per run. Correct when lhs is the identity arange
// (token rows already ordered in `a` — lemonseed pre-gathers via flat[src]).
// Does not require right_sorted_ (that flag is false when Python passes an
// explicit arange lhs). Disable with MLX_ROCM_SORTED_GATHER=0.
static bool sorted_gather_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("MLX_ROCM_SORTED_GATHER");
    return !(e && std::string(e) == "0");
  }();
  return on;
}

// Minimum average tokens per distinct expert run to prefer segment GEMMs over
// gemv_gather. Default 1: even short runs use tiled GEMM (large K×N MoE).
static int moe_segment_min_avg() {
  static const int v = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_SEG_MIN");
    if (!e || !*e)
      return 1;
    char* end = nullptr;
    long x = std::strtol(e, &end, 10);
    return (end != e && x >= 1) ? static_cast<int>(x) : 1;
  }();
  return v;
}

static bool try_moe_segment_gather_mm(
    rocm::CommandEncoder& encoder,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    bool assume_identity_lhs) {
  (void)lda;
  if (!sorted_gather_enabled()) {
    return false;
  }
  // For M==1 the transpose flag is spurious ([1,K] == [K,1] data); the batch is
  // still [batch,K] contiguous (verified by the strideLast==1 guard below), so
  // the segment GEMM math is valid. Only reject a genuine M>1 transpose.
  if (a_transposed && a.shape(-2) != 1) {
    return false;
  }

  // out is [..., M=1, N] → batch of vectors
  const int batch = static_cast<int>(out.size() / static_cast<size_t>(N));
  if (batch <= 0) {
    return false;
  }

  // Contiguous batch of 1×K token rows (pre-gathered features).
  // shape (..., 1, K). Unit dims often have stride 0 in MLX — do NOT require
  // strides[-2]==K (that false-rejected lemonseed's [B,1,1,K] layout and left
  // us on gemv_gather for ~55% of GPU time).
  if (a.shape(-2) != 1 || a.shape(-1) != K) {
    return false;
  }
  if (a.strides()[a.ndim() - 1] != 1) {
    return false;
  }
  // Tight pack: batch successive panels at offsets 0, K, 2K, ...
  if (a.size() != static_cast<size_t>(batch) * static_cast<size_t>(K)) {
    return false;
  }

  // Need flat index buffers (one index per batch row). The host copy below
  // reads `batch` *contiguous* uint32; a non-row-contiguous index view (which
  // the gather_mm VJP produces on the backward pass) would read out of bounds
  // -> illegal memory access. Bail to the generic gather path in that case.
  // Require an EXACT one-index-per-row match, not just >=: the backward
  // gather_mm VJP reuses this primitive with index arrays whose logical size no
  // longer equals `batch` (or that alias a smaller physical buffer), and reading
  // `batch` contiguous uint32 from them overruns the allocation -> illegal
  // access. Exact size + row-contiguous keeps this on the true forward pattern
  // and routes everything else to the generic gather path.
  if (static_cast<size_t>(rhs_indices.size()) != static_cast<size_t>(batch) ||
      !rhs_indices.flags().row_contiguous) {
    return false;
  }
  if (!assume_identity_lhs &&
      (static_cast<size_t>(lhs_indices.size()) != static_cast<size_t>(batch) ||
       !lhs_indices.flags().row_contiguous)) {
    return false;
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(rhs_indices);
  if (!assume_identity_lhs) {
    encoder.set_input_array(lhs_indices);
  }
  encoder.set_output_array(out);

  mlx::core::Shape b_batch_shape{b.shape().begin(), b.shape().end() - 2};
  mlx::core::Strides b_batch_strides{b.strides().begin(), b.strides().end() - 2};

  // Pack + strided-batched hipBLASLt (default ON for bf16 1-D MoE).
  // ZERO-SYNC: M_pad = align_up(batch, 32) is host-known (worst-case one
  // expert takes all tokens) → NO D2H / StreamSynchronize. Kill-switch:
  // MLX_ROCM_MOE_PACK=0. VALU: MLX_ROCM_MOE_DEVICE_SEG=1.
  // Legacy exact-max-run via D2H: MLX_ROCM_MOE_PACK_EXACT=1 (slower, syncs).
  static const bool use_pack = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_PACK");
    if (!e || !*e)
      return true; // default ON
    return !(e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n' ||
             e[0] == 'N');
  }();
  static const bool use_device_seg = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_DEVICE_SEG");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();
  static const bool pack_exact_d2h = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_PACK_EXACT");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();
  const int n_experts =
      (b_batch_shape.size() == 1) ? static_cast<int>(b_batch_shape[0]) : 0;
  if (use_pack && assume_identity_lhs && n_experts > 0 &&
      n_experts <= 256 && a.dtype() == bfloat16 &&
      rocm::is_hipblaslt_available() && !use_device_seg) {
    int M_fixed;
    if (pack_exact_d2h) {
      // Legacy: D2H ids, measure max run (syncs — avoid for train).
      hipStream_t hs_pack = static_cast<hipStream_t>(encoder.stream());
      static thread_local uint32_t* pin_pack = nullptr;
      static thread_local size_t pin_pack_cap = 0;
      const size_t need_pack = static_cast<size_t>(batch);
      if (need_pack > pin_pack_cap) {
        if (pin_pack)
          (void)hipHostFree(pin_pack);
        pin_pack_cap = need_pack + need_pack / 2 + 1024;
        CHECK_HIP_ERROR(hipHostMalloc(
            reinterpret_cast<void**>(&pin_pack),
            pin_pack_cap * sizeof(uint32_t),
            hipHostMallocDefault));
      }
      CHECK_HIP_ERROR(hipMemcpyAsync(
          pin_pack,
          gpu_ptr<const uint32_t>(rhs_indices),
          need_pack * sizeof(uint32_t),
          hipMemcpyDeviceToHost,
          hs_pack));
      CHECK_HIP_ERROR(hipStreamSynchronize(hs_pack));
      M_fixed = 1;
      for (int i = 0; i < batch;) {
        uint32_t e = pin_pack[static_cast<size_t>(i)];
        int j = i + 1;
        while (j < batch && pin_pack[static_cast<size_t>(j)] == e)
          ++j;
        M_fixed = std::max(M_fixed, j - i);
        i = j;
      }
      M_fixed = std::min(M_fixed, batch);
      M_fixed = (M_fixed + 31) & ~31;
    } else {
      // Tight pad: device max expert-run + 4B D2H (avoids full-T OOM/FLOPs).
      M_fixed = rocm::moe_max_run_length_sync(encoder, rhs_indices, batch, n_experts);
    }
    if (M_fixed < 32)
      M_fixed = 32;
    // Temporaries: packed_a [E,M,K], packed_c [E,M,N], slot_map [E,M], counts [E]
    array packed_a(
        mlx::core::Shape{n_experts, M_fixed, K}, a.dtype(), nullptr, {});
    array packed_c(
        mlx::core::Shape{n_experts, M_fixed, N}, a.dtype(), nullptr, {});
    array slot_map(
        mlx::core::Shape{n_experts, M_fixed}, int32, nullptr, {});
    array counts(mlx::core::Shape{n_experts}, int32, nullptr, {});
    packed_a.set_data(mlx::core::rocm::malloc_async(packed_a.nbytes(), encoder));
    packed_c.set_data(mlx::core::rocm::malloc_async(packed_c.nbytes(), encoder));
    slot_map.set_data(mlx::core::rocm::malloc_async(slot_map.nbytes(), encoder));
    counts.set_data(mlx::core::rocm::malloc_async(counts.nbytes(), encoder));
    encoder.add_temporary(packed_a);
    encoder.add_temporary(packed_c);
    encoder.add_temporary(slot_map);
    encoder.add_temporary(counts);

    // a is [batch, 1, K] or similar — need flat [batch, K] view for pack.
    // Contiguous batch*K was verified earlier (a.size() == batch*K).
    rocm::moe_pack_tokens(
        encoder, a, rhs_indices, packed_a, slot_map, counts, batch, K,
        n_experts, M_fixed);

    // B: [E, K, N] or transposed [E, N, K]. Stride between experts.
    const int64_t b_expert_stride = b_batch_strides.empty()
        ? 0
        : static_cast<int64_t>(b_batch_strides[0]);
    const int64_t stride_a = static_cast<int64_t>(M_fixed) * K;
    const int64_t stride_c = static_cast<int64_t>(M_fixed) * N;
    // hipblaslt_gemm_batched expects row-major MLX arrays; lda = K for A [M,K].
    rocm::hipblaslt_gemm_batched(
        encoder,
        /*transpose_a=*/false,
        b_transposed,
        M_fixed,
        N,
        K,
        1.0f,
        packed_a,
        /*lda=*/K,
        stride_a,
        b,
        ldb,
        b_expert_stride,
        0.0f,
        packed_c,
        /*ldc=*/N,
        stride_c,
        n_experts,
        a.dtype());

    // Zero out then scatter packed results (out may have garbage).
    {
      void* op = gpu_ptr<void>(out);
      size_t nbytes = out.nbytes();
      encoder.launch_kernel([op, nbytes](hipStream_t stream) {
        (void)hipMemsetAsync(op, 0, nbytes, stream);
      });
    }
    rocm::moe_unpack_tokens(
        encoder, packed_c, slot_map, out, n_experts, M_fixed, N);
    return true;
  }
  if (use_device_seg && assume_identity_lhs && n_experts > 0 &&
      n_experts <= 256 &&
      (a.dtype() == bfloat16 || a.dtype() == float16 || a.dtype() == float32)) {
    const int64_t b_expert_stride = b_batch_strides.empty()
        ? 0
        : static_cast<int64_t>(b_batch_strides[0]);
    rocm::moe_sorted_expert_gemm(
        encoder,
        a,
        b,
        rhs_indices,
        out,
        batch,
        N,
        K,
        n_experts,
        b_transposed,
        ldb,
        b_expert_stride);
    return true;
  }

  hipStream_t hs = static_cast<hipStream_t>(encoder.stream());

  auto expert_offset_fn = [&](uint32_t expert) -> int64_t {
    if (b_batch_shape.empty()) {
      return 0;
    }
    if (b_batch_shape.size() == 1) {
      return static_cast<int64_t>(expert) * b_batch_strides[0];
    }
    int64_t off = 0;
    int64_t idx = static_cast<int64_t>(expert);
    for (int d = static_cast<int>(b_batch_shape.size()) - 1; d >= 0; --d) {
      int64_t coord = idx % b_batch_shape[d];
      idx /= b_batch_shape[d];
      off += coord * b_batch_strides[d];
    }
    return off;
  };

  // Async host path (opt-in MLX_ROCM_MOE_ASYNC=1): D2H + RLE + hipBLASLt inside
  // hipLaunchHostFunc so the main thread never StreamSynchronize. Can deadlock
  // on some ROCm builds when the main thread also syncs the same stream — keep
  // OFF by default until validated; default remains blocking host path.
  static const bool use_async_moe = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_ASYNC");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();

  if (use_async_moe && assume_identity_lhs &&
      rocm::is_hipblaslt_available()) {
    auto* job = new MoeAsyncJob();
    job->batch = batch;
    job->N = N;
    job->K = K;
    job->b_transposed = b_transposed;
    job->ldb = ldb;
    job->esz = a.itemsize();
    job->aB = static_cast<const char*>(gpu_ptr<void>(a));
    job->bB = static_cast<const char*>(gpu_ptr<void>(b));
    job->cB = static_cast<char*>(gpu_ptr<void>(out));
    job->dtype = a.dtype();
    job->device_id = encoder.device().hip_device();
    job->stream = hs;
    job->b_ndim = static_cast<int>(b_batch_shape.size());
    for (int d = 0; d < job->b_ndim && d < 4; ++d) {
      job->b_shape[d] = b_batch_shape[d];
      job->b_strides[d] = b_batch_strides[d];
    }

    const size_t need = static_cast<size_t>(batch);
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&job->pin),
        need * sizeof(uint32_t),
        hipHostMallocDefault));

    const uint32_t* rhs_dev = gpu_ptr<const uint32_t>(rhs_indices);
    // Enqueue D2H + host callback that launches segment GEMMs. Main thread
    // returns immediately so MLX can keep building the step graph.
    encoder.launch_kernel([job, need, rhs_dev](hipStream_t stream) {
      (void)hipMemcpyAsync(
          job->pin,
          rhs_dev,
          need * sizeof(uint32_t),
          hipMemcpyDeviceToHost,
          stream);
      (void)hipLaunchHostFunc(stream, moe_async_host_cb, job);
    });
    return true;
  }

  // Synchronous host path. Pin-cache: SwiGLU does gather_mm ×3 (gate/up/down)
  // with the same sorted rhs indices. Reuse the pinned copy for 2 more calls
  // after each D2H so we pay one stream sync per MoE proj trio instead of three
  // — the main train-step pipeline drain. reuses_left hits 0 before the next
  // layer so a recycled device pointer with new data cannot serve a stale pin.
  // Kill-switch: MLX_ROCM_MOE_NO_PIN_CACHE=1.
  static thread_local uint32_t* pin_rhs = nullptr;
  static thread_local uint32_t* pin_lhs = nullptr;
  static thread_local size_t pin_cap = 0;
  static thread_local const void* pin_rhs_dev = nullptr;
  static thread_local size_t pin_rhs_n = 0;
  static thread_local int pin_reuses_left = 0;
  static const bool no_pin_cache = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_NO_PIN_CACHE");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();

  const size_t need = static_cast<size_t>(batch);
  if (need > pin_cap) {
    if (pin_rhs)
      (void)hipHostFree(pin_rhs);
    if (pin_lhs)
      (void)hipHostFree(pin_lhs);
    pin_rhs = pin_lhs = nullptr;
    pin_cap = need + need / 2 + 1024;
    pin_rhs_dev = nullptr;
    pin_reuses_left = 0;
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&pin_rhs),
        pin_cap * sizeof(uint32_t),
        hipHostMallocDefault));
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&pin_lhs),
        pin_cap * sizeof(uint32_t),
        hipHostMallocDefault));
  }

  const void* rhs_dev = gpu_ptr<const uint32_t>(rhs_indices);
  const bool cache_hit = !no_pin_cache && assume_identity_lhs &&
      pin_rhs_dev == rhs_dev && pin_rhs_n == need && pin_reuses_left > 0;

  static const bool pin_stats = std::getenv("MLX_ROCM_MOE_PIN_STATS") != nullptr;
  static thread_local long pin_hits = 0, pin_misses = 0;

  if (cache_hit) {
    --pin_reuses_left;
    ++pin_hits;
  } else {
    ++pin_misses;
    if (pin_stats && ((pin_hits + pin_misses) % 200) == 0) {
      fprintf(
          stderr,
          "[moe-pin] hits=%ld misses=%ld hit_rate=%.1f%% assume_id=%d\n",
          pin_hits,
          pin_misses,
          100.0 * pin_hits / std::max(1L, pin_hits + pin_misses),
          (int)assume_identity_lhs);
    }
    CHECK_HIP_ERROR(hipMemcpyAsync(
        pin_rhs,
        rhs_dev,
        need * sizeof(uint32_t),
        hipMemcpyDeviceToHost,
        hs));

    if (!assume_identity_lhs) {
      CHECK_HIP_ERROR(hipMemcpyAsync(
          pin_lhs,
          gpu_ptr<const uint32_t>(lhs_indices),
          need * sizeof(uint32_t),
          hipMemcpyDeviceToHost,
          hs));
    }
    CHECK_HIP_ERROR(hipStreamSynchronize(hs));

    if (!assume_identity_lhs) {
      for (int i = 0; i < batch; ++i) {
        if (pin_lhs[static_cast<size_t>(i)] != static_cast<uint32_t>(i)) {
          pin_rhs_dev = nullptr;
          pin_reuses_left = 0;
          return false; // non-identity lhs → cannot treat a as dense [batch,K]
        }
      }
    }
    // Allow gate→up→down to share this pin (2 further hits after this call).
    pin_rhs_dev = rhs_dev;
    pin_rhs_n = need;
    pin_reuses_left = assume_identity_lhs && !no_pin_cache ? 2 : 0;
  }

  int n_runs = 0;
  for (int i = 0; i < batch;) {
    uint32_t e = pin_rhs[static_cast<size_t>(i)];
    int j = i + 1;
    while (j < batch && pin_rhs[static_cast<size_t>(j)] == e) {
      ++j;
    }
    ++n_runs;
    i = j;
  }
  if (n_runs <= 0) {
    return false;
  }
  if (batch / n_runs < moe_segment_min_avg()) {
    return false;
  }

  const bool moe_use_blaslt = rocm::is_hipblaslt_available();
  const size_t moe_esz = a.itemsize();
  const char* moe_aB = static_cast<const char*>(gpu_ptr<void>(a));
  const char* moe_bB = static_cast<const char*>(gpu_ptr<void>(b));
  char* moe_cB = static_cast<char*>(gpu_ptr<void>(out));
  int start = 0;
  while (start < batch) {
    uint32_t e = pin_rhs[static_cast<size_t>(start)];
    int end = start + 1;
    while (end < batch && pin_rhs[static_cast<size_t>(end)] == e) {
      ++end;
    }
    int Mseg = end - start;
    if (moe_use_blaslt) {
      rocm::hipblaslt_gemm_ptrs(
          encoder, /*transpose_a=*/false, b_transposed, Mseg, N, K, 1.0f,
          moe_aB + static_cast<size_t>(start) * K * moe_esz, /*lda=*/K,
          moe_bB + static_cast<size_t>(expert_offset_fn(e)) * moe_esz, ldb,
          0.0f, moe_cB + static_cast<size_t>(start) * N * moe_esz, /*ldc=*/N,
          a.dtype());
    } else {
      rocm::naive_gemm_with_offset(
          encoder, a, b, out, Mseg, N, K, /*a_transposed=*/false, /*lda=*/K,
          static_cast<int64_t>(start) * K, b_transposed, ldb, expert_offset_fn(e),
          static_cast<int64_t>(start) * N, 1.0f, 0.0f);
    }
    start = end;
  }
  return true;
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

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  auto [transposed_a, lda, a_] = check_transpose(encoder, s, a);
  auto [transposed_b, ldb, b_] = check_transpose(encoder, s, b);

  // MoE win: M==1 + identity/pre-gathered a + long rhs runs → segment GEMMs.
  // right_sorted_ means sequential a (lhs omitted). With explicit identity
  // arange (common for MoE 4-D shapes) the flag is false; we D2H-check lhs.
  if (M == 1) {
    const bool assume_id = right_sorted_; // sequential a when flag set
    if (try_moe_segment_gather_mm(
            encoder,
            a_,
            b_,
            lhs_indices,
            rhs_indices,
            out,
            N,
            K,
            transposed_a,
            lda,
            transposed_b,
            ldb,
            /*assume_identity_lhs=*/assume_id)) {
      return;
    }
  }

  auto use_gemv = rocm::can_use_gemv(M, N, K, transposed_a, transposed_b);

  if (M == 1 && use_gemv) {
    rocm::gather_mv(b_, a_, rhs_indices, lhs_indices, out, N, K, encoder);
    return;
  }

  if (N == 1 && use_gemv) {
    rocm::gather_mv(a_, b_, lhs_indices, rhs_indices, out, M, K, encoder);
    return;
  }

  // Keep gather indices on device and resolve per-batch matrix offsets inside
  // the kernel to avoid host synchronization.
  rocm::naive_gemm_gather(
      encoder,
      a_,
      b_,
      lhs_indices,
      rhs_indices,
      out,
      M,
      N,
      K,
      transposed_a,
      lda,
      transposed_b,
      ldb,
      1.0f,
      0.0f);
}

// SegmentedMM: out[i] = A[:, k0:k1] @ B[k0:k1, :] for each segment [k0, k1).
// Used by gather_mm weight VJP when indices are sorted (MoE training).
void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto& segments_pre = inputs[2];

  if (out.size() == 0 || a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(mlx::core::rocm::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int num_segments = static_cast<int>(segments_pre.size() / 2);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  array segments = segments_pre;
  if (!segments_pre.flags().row_contiguous) {
    segments = contiguous_copy_gpu(segments_pre, s);
    encoder.add_temporary(segments);
  }

  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(segments);
  encoder.set_output_array(out);

  const int64_t a_k_stride = a.strides()[a.ndim() - 1];
  const int64_t b_k_stride = b.strides()[b.ndim() - 2];
  const int64_t out_stride = static_cast<int64_t>(M) * N;

  // Opt-in device-side SegmentedMM (no stream sync). Enable with
  // MLX_SEGMM_DEVICE=1. Default host+hipBLASLt is faster on MI300X today.
  static const bool seg_device = [] {
    const char* e = std::getenv("MLX_SEGMM_DEVICE");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();
  if (seg_device &&
      (a.dtype() == bfloat16 || a.dtype() == float16 || a.dtype() == float32)) {
    rocm::segmented_mm_device(
        encoder,
        a,
        b,
        segments,
        out,
        M,
        N,
        num_segments,
        a_transposed,
        lda,
        a_k_stride,
        b_transposed,
        ldb,
        b_k_stride,
        out_stride);
    return;
  }

  // Host path (default): D2H segment table + one hipBLASLt/naive GEMM per segment.
  // Pin-cache: MoE weight VJP does 3× segmented_mm with the same segments
  // (dW_gate/up/down). Reuse pin for 2 more hits after each D2H.
  static thread_local uint32_t* pin_segs = nullptr;
  static thread_local size_t pin_segs_cap = 0;
  static thread_local const void* pin_segs_dev = nullptr;
  static thread_local size_t pin_segs_n = 0;
  static thread_local int pin_segs_reuses = 0;
  static const bool no_seg_cache = [] {
    const char* e = std::getenv("MLX_ROCM_MOE_NO_PIN_CACHE");
    return e && (e[0] == '1' || e[0] == 'o' || e[0] == 'O' || e[0] == 't' ||
                 e[0] == 'T');
  }();
  const size_t nseg_u32 = static_cast<size_t>(num_segments) * 2;
  if (nseg_u32 > pin_segs_cap) {
    if (pin_segs)
      (void)hipHostFree(pin_segs);
    pin_segs_cap = nseg_u32 + 64;
    pin_segs_dev = nullptr;
    pin_segs_reuses = 0;
    CHECK_HIP_ERROR(hipHostMalloc(
        reinterpret_cast<void**>(&pin_segs),
        pin_segs_cap * sizeof(uint32_t),
        hipHostMallocDefault));
  }
  hipStream_t hs = static_cast<hipStream_t>(encoder.stream());
  const void* segs_dev = gpu_ptr<const uint32_t>(segments);
  const bool seg_hit = !no_seg_cache && pin_segs_dev == segs_dev &&
      pin_segs_n == nseg_u32 && pin_segs_reuses > 0;
  if (seg_hit) {
    --pin_segs_reuses;
  } else {
    CHECK_HIP_ERROR(hipMemcpyAsync(
        pin_segs,
        segs_dev,
        nseg_u32 * sizeof(uint32_t),
        hipMemcpyDeviceToHost,
        hs));
    CHECK_HIP_ERROR(hipStreamSynchronize(hs));
    pin_segs_dev = segs_dev;
    pin_segs_n = nseg_u32;
    pin_segs_reuses = no_seg_cache ? 0 : 2;
  }

  const size_t esize = size_of(a.dtype());
  char* out_base = static_cast<char*>(gpu_ptr<void>(out));

  for (int i = 0; i < num_segments; ++i) {
    uint32_t k0 = pin_segs[static_cast<size_t>(2 * i)];
    uint32_t k1 = pin_segs[static_cast<size_t>(2 * i + 1)];
    if (k1 <= k0) {
      void* c_ptr = out_base + static_cast<size_t>(i) * out_stride * esize;
      size_t bytes = static_cast<size_t>(out_stride) * esize;
      encoder.launch_kernel([c_ptr, bytes](hipStream_t stream) {
        (void)hipMemsetAsync(c_ptr, 0, bytes, stream);
      });
      continue;
    }
    int Kseg = static_cast<int>(k1 - k0);
    const Dtype seg_dt = a.dtype();
    static const bool seg_use_blaslt = std::getenv("MLX_SEGMM_NAIVE") == nullptr;
    if (seg_use_blaslt && encoder.device().supports_cdna_mfma_gemm() &&
        (seg_dt == bfloat16 || seg_dt == float16 || seg_dt == float32)) {
      const char* a_seg = static_cast<const char*>(gpu_ptr<void>(a)) +
          static_cast<size_t>(static_cast<int64_t>(k0) * a_k_stride) * esize;
      const char* b_seg = static_cast<const char*>(gpu_ptr<void>(b)) +
          static_cast<size_t>(static_cast<int64_t>(k0) * b_k_stride) * esize;
      char* c_seg = out_base + static_cast<size_t>(i) * out_stride * esize;
      rocm::hipblaslt_gemm_ptrs(
          encoder,
          a_transposed,
          b_transposed,
          M,
          N,
          Kseg,
          1.0f,
          a_seg,
          static_cast<int>(lda),
          b_seg,
          static_cast<int>(ldb),
          0.0f,
          c_seg,
          N,
          seg_dt);
    } else {
      rocm::naive_gemm_with_offset(
          encoder,
          a,
          b,
          out,
          M,
          N,
          Kseg,
          a_transposed,
          lda,
          static_cast<int64_t>(k0) * a_k_stride,
          b_transposed,
          ldb,
          static_cast<int64_t>(k0) * b_k_stride,
          static_cast<int64_t>(i) * out_stride,
          1.0f,
          0.0f);
    }
  }
}

} // namespace mlx::core
