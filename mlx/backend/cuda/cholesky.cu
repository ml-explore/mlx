// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cusolver_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <nvtx3/nvtx3.hpp>
#include <cuda/cmath>

#include <cassert>

namespace mlx::core {

namespace cg = cooperative_groups;

namespace cu {

template <typename T>
__global__ void set_batch_pointers(T** ptrs, T* base, int64_t n2, int batch) {
  auto i = cg::this_grid().thread_rank();
  if (i < batch) {
    ptrs[i] = base + i * n2;
  }
}

// potrf only writes the requested triangle, so zero the other one.
template <typename T>
__global__ void zero_triangle(T* matrices, int n, bool upper, int64_t total) {
  auto i = cg::this_grid().thread_rank();
  if (i >= total) {
    return;
  }
  int64_t offset = i % (static_cast<int64_t>(n) * n);
  int row = offset / n;
  int col = offset % n;
  if (upper ? (col < row) : (col > row)) {
    matrices[i] = T(0);
  }
}

} // namespace cu

namespace {

template <typename T>
struct Potrf;

template <>
struct Potrf<float> {
  static constexpr auto buffer_size = cusolverDnSpotrf_bufferSize;
  static constexpr auto factor = cusolverDnSpotrf;
  static constexpr auto factor_batched = cusolverDnSpotrfBatched;
};

// Unreachable while array.cpp rejects float64 for any GPU primitive.
template <>
struct Potrf<double> {
  static constexpr auto buffer_size = cusolverDnDpotrf_bufferSize;
  static constexpr auto factor = cusolverDnDpotrf;
  static constexpr auto factor_batched = cusolverDnDpotrfBatched;
};

template <typename T>
void cholesky_impl(
    cu::CommandEncoder& encoder,
    const array& a,
    array& out,
    bool upper,
    Stream s) {
  // The factorization is in place, so work on a copy of the input.
  copy_gpu(
      a,
      out,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      s);

  if (out.size() == 0) {
    return;
  }

  int n = a.shape(-1);
  int batch = static_cast<int>(cusolver_utils::batch_count(a));
  auto uplo = cusolver_utils::uplo_for(upper);
  auto handle = get_cusolver_handle(encoder);

  // `info` is never read back: that needs a stream synchronize, and the CPU
  // backend also declines to raise on a non positive-definite input.
  //
  // Allocate before the captures below, which allocate_workspace asserts.
  auto* info =
      static_cast<int*>(allocate_workspace(encoder, batch * sizeof(int)));

  // insert_graph_dependencies() clears active_deps_ after each node, so every
  // node below has to register the arrays it touches. Registering once leaves
  // later nodes without predecessors, free to run before their inputs are
  // ready.
  if (batch > 1) {
    // A raw workspace pointer has no array identity, so no dependency edge
    // could be expressed between the fill kernel and potrfBatched reading it.
    static_assert(sizeof(T*) == sizeof(uint64_t));
    array ptrs_arr(
        cu::malloc_async(batch * sizeof(T*), encoder), {batch}, uint64);
    encoder.add_temporary(ptrs_arr);
    auto** ptrs = gpu_ptr<T*>(ptrs_arr);

    encoder.set_output_array(ptrs_arr);
    encoder.add_kernel_node(
        cu::set_batch_pointers<T>,
        static_cast<uint32_t>(cuda::ceil_div(batch, 256)),
        256,
        ptrs,
        gpu_ptr<T>(out),
        static_cast<int64_t>(n) * n,
        batch);

    encoder.set_input_array(ptrs_arr);
    encoder.set_input_array(out);
    encoder.set_output_array(out);
    auto capture = encoder.capture_context();
    CHECK_CUSOLVER_ERROR(
        Potrf<T>::factor_batched(handle, uplo, n, ptrs, n, info, batch));
  } else {
    int lwork = 0;
    CHECK_CUSOLVER_ERROR(
        Potrf<T>::buffer_size(handle, uplo, n, gpu_ptr<T>(out), n, &lwork));
    auto* workspace =
        static_cast<T*>(allocate_workspace(encoder, lwork * sizeof(T)));

    encoder.set_input_array(out);
    encoder.set_output_array(out);
    auto capture = encoder.capture_context();
    CHECK_CUSOLVER_ERROR(
        Potrf<T>::factor(
            handle, uplo, n, gpu_ptr<T>(out), n, workspace, lwork, info));
  }

  int64_t total = out.size();
  encoder.set_input_array(out);
  encoder.set_output_array(out);
  encoder.add_kernel_node(
      cu::zero_triangle<T>,
      static_cast<uint32_t>(cuda::ceil_div(total, static_cast<int64_t>(256))),
      256,
      gpu_ptr<T>(out),
      n,
      upper,
      total);
}

} // namespace

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Cholesky::eval_gpu");
  assert(inputs.size() == 1);
  auto& encoder = cu::get_command_encoder(stream());

  switch (inputs[0].dtype()) {
    case float32:
      cholesky_impl<float>(encoder, inputs[0], out, upper_, stream());
      break;
    case float64:
      cholesky_impl<double>(encoder, inputs[0], out, upper_, stream());
      break;
    default:
      throw std::runtime_error(
          "[Cholesky::eval_gpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
