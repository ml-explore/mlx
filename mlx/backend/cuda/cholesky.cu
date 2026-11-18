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
fill_matrix_pointers(void** ptrs, float* base, int64_t n2, int64_t batch) {
  int64_t i = cg::this_grid().thread_rank();
  if (i < batch) {
    ptrs[i] = base + i * n2;
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

  if (a.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);

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
  auto* out_ptr = gpu_ptr<float>(out);

  int64_t n = a.shape(-1);
  int64_t batch = a.size() / (n * n);

  // Matching the CPU op, info is never read back: a non positive definite
  // input gives an undefined factor rather than an error.
  auto* info = static_cast<int*>(
      allocate_workspace(encoder, batch * sizeof(int)));

  // TODO: For some consumer cards serialized loop might be faster.
  if (batch > 1) {
    array ptrs(
        cu::malloc_async(batch * sizeof(void*), encoder),
        {static_cast<int>(batch)},
        uint64);
    encoder.add_temporary(ptrs);
    encoder.set_output_array(ptrs);
    encoder.add_kernel_node(
        &cu::fill_matrix_pointers,
        cuda::ceil_div(batch, 256),
        256,
        gpu_ptr<void*>(ptrs),
        out_ptr,
        n * n,
        batch);

    encoder.set_input_array(ptrs);
    encoder.set_output_array(out);
    auto capture = encoder.capture_context();
    CHECK_CUSOLVER_ERROR(cusolverDnSpotrfBatched(
        handle,
        uplo,
        n,
        gpu_ptr<float*>(ptrs),
        /* lda */ n,
        info,
        batch));
  } else {
    int lwork = 0;
    CHECK_CUSOLVER_ERROR(cusolverDnSpotrf_bufferSize(
        handle, uplo, n, out_ptr, n, &lwork));
    auto* workspace = static_cast<float*>(
        allocate_workspace(encoder, lwork * sizeof(float)));

    encoder.set_input_array(out);
    encoder.set_output_array(out);
    auto capture = encoder.capture_context();
    CHECK_CUSOLVER_ERROR(cusolverDnSpotrf(
        handle, uplo, n, out_ptr, n, workspace, lwork, info));
  }

  encoder.set_output_array(out);
  dispatch_bool(out.size() > INT32_MAX, [&](auto large) {
    using IdxT = std::conditional_t<large(), int64_t, int32_t>;
    auto [num_blocks, block_dims] = get_launch_args(out, large());
    encoder.add_kernel_node(
        cu::zero_triangle<float, IdxT>,
        num_blocks,
        block_dims,
        out_ptr,
        static_cast<IdxT>(out.size()),
        static_cast<int32_t>(n),
        upper_);
  });
}

} // namespace mlx::core
