// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/cusolver_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

// potrf writes only one triangle; zero the other one to match the CPU op.
template <typename T, typename IdxT>
__global__ void zero_triangle(T* out, IdxT size, int32_t n, bool zero_below) {
  IdxT index = cg::this_grid().thread_rank();
  if (index >= size) {
    return;
  }
  int32_t row = (index / n) % n;
  int32_t col = index % n;
  if (zero_below ? (col < row) : (col > row)) {
    out[index] = T(0);
  }
}

// potrfBatched wants a device array of per matrix pointers.
__global__ void
fill_matrix_pointers(void** ptrs, char* base, size_t stride, int32_t count) {
  int32_t i = cg::this_grid().thread_rank();
  if (i < count) {
    ptrs[i] = base + i * stride;
  }
}

} // namespace cu

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Cholesky::eval_gpu");
  auto& s = stream();
  auto& a = inputs[0];

  // Copy the input to the output; the factorization runs in place.
  copy_gpu(
      a,
      out,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      s);

  int64_t n = a.shape(-1);
  if (a.size() == 0) {
    return;
  }
  int64_t num_matrices = a.size() / (n * n);

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_output_array(out);

  auto handle = get_cusolver_handle(encoder.device());
  CHECK_CUSOLVER_ERROR(cusolverDnSetStream(handle, encoder.stream()));

  // cuSOLVER is column major. The input is symmetric and a column major
  // lower triangle is a row major upper one, so pass the opposite of
  // upper_, same as the CPU op.
  cublasFillMode_t uplo =
      upper_ ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  // Only float32 reaches eval_gpu: float64 is rejected on GPU streams at
  // array construction.
  auto type = CUDA_R_32F;
  auto* out_ptr = gpu_ptr<char>(out);
  size_t matrix_bytes = n * n * out.itemsize();

  // Matching the CPU op, info is never read back: a non positive definite
  // input gives an undefined factor rather than an error.
  auto* info = static_cast<int*>(
      allocate_workspace(encoder, num_matrices * sizeof(int)));

  // A loop of single factorizations beats the batched kernels once the
  // matrices are large enough to fill the device (measured on sm_120).
  if (num_matrices > 1 && num_matrices <= INT32_MAX && n <= 256) {
    auto** ptrs = static_cast<void**>(
        allocate_workspace(encoder, num_matrices * sizeof(void*)));
    auto capture = encoder.capture_context();
    int32_t count = num_matrices;
    cu::fill_matrix_pointers<<<(count + 255) / 256, 256, 0, encoder.stream()>>>(
        ptrs, out_ptr, matrix_bytes, count);
    CHECK_CUSOLVER_ERROR(cusolverDnSpotrfBatched(
        handle,
        uplo,
        n,
        reinterpret_cast<float**>(ptrs),
        /* lda */ n,
        info,
        count));
  } else {
    size_t device_bytes = 0;
    size_t host_bytes = 0;
    CHECK_CUSOLVER_ERROR(cusolverDnXpotrf_bufferSize(
        handle,
        /* params */ nullptr,
        uplo,
        n,
        type,
        out_ptr,
        /* lda */ n,
        type,
        &device_bytes,
        &host_bytes));

    auto* device_ws = allocate_workspace(encoder, device_bytes);
    auto host_ws = std::make_shared<std::vector<char>>(host_bytes);
    if (host_bytes > 0) {
      encoder.add_completed_handler([host_ws]() {});
    }

    auto capture = encoder.capture_context();
    for (int64_t i = 0; i < num_matrices; ++i) {
      CHECK_CUSOLVER_ERROR(cusolverDnXpotrf(
          handle,
          /* params */ nullptr,
          uplo,
          n,
          type,
          out_ptr + i * matrix_bytes,
          /* lda */ n,
          type,
          device_ws,
          device_bytes,
          host_ws->data(),
          host_bytes,
          info + i));
    }
  }

  encoder.set_output_array(out);
  dispatch_bool(out.size() > INT32_MAX, [&](auto large) {
    using IdxT = std::conditional_t<large(), int64_t, int32_t>;
    auto [num_blocks, block_dims] = get_launch_args(out, large());
    encoder.add_kernel_node(
        cu::zero_triangle<float, IdxT>,
        num_blocks,
        block_dims,
        gpu_ptr<float>(out),
        out.size(),
        n,
        upper_);
  });
}

} // namespace mlx::core
