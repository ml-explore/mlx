// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/mps/cholesky.h"
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

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  const auto& a = inputs.at(0);
  if (a.dtype() != float32) {
    throw std::runtime_error("[Cholesky::eval_gpu] only supports float32.");
  }

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  MPS::DataType mps_dtype = MPS::DataTypeFloat32;

  const int num_rows = a.shape(-2);
  const int row_nbytes = a.shape(-1) * sizeof(float);
  const int num_matrices = a.size() / (num_rows * num_rows);

  const auto mat_desc = MPS::MatrixDescriptor::matrixDescriptor(
      /* rows = */ num_rows,
      /* columns = */ num_rows,
      /* rowBytes = */ row_nbytes,
      /* dataType = */ mps_dtype);

  d.end_encoding(s.index);

  // Ensure that the input is contiguous.
  auto input = a;
  if (!a.flags().contiguous) {
    copy(
        a,
        input,
        a.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  }

  // The kernel does not support batching, so we have to loop explicitly.
  for (int i = 0; i < num_matrices; i++) {
    size_t offset = i * num_rows * row_nbytes;
    auto a_buf = static_cast<const MTL::Buffer*>(input.buffer().ptr());
    auto a_mat = MPS::Matrix::alloc()->init(a_buf, offset, mat_desc);

    auto out_buf = static_cast<MTL::Buffer*>(out.buffer().ptr());
    auto out_mat = MPS::Matrix::alloc()->init(out_buf, offset, mat_desc);

    auto kernel = MPS::MatrixDecompositionCholesky::alloc()->init(
        d.mtl_device(), !upper_, num_rows);

    auto command_buffer = d.get_command_buffer(s.index);
    kernel->encodeToCommandBuffer(command_buffer, a_mat, out_mat, nullptr);
    a_mat->release();
    out_mat->release();
    command_buffer->addCompletedHandler(
        [kernel](MTL::CommandBuffer*) mutable { kernel->release(); });
  }

  // Zero out the unwanted part of the output.
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto make_triangular_kernel = d.get_kernel("make_triangular");
  compute_encoder->setComputePipelineState(make_triangular_kernel);
  compute_encoder->setBytes(&upper_, sizeof(upper_), 0);
  compute_encoder->setBytes(&num_rows, sizeof(num_rows), 1);
  compute_encoder.set_output_array(out, 2);

  const MTL::Size threads_per_grid(num_rows, num_rows, num_matrices);
  const int threadgroup_width = std::min(
      threads_per_grid.width, make_triangular_kernel->threadExecutionWidth());
  const int threadgroup_height = std::min(
      threads_per_grid.height,
      make_triangular_kernel->maxTotalThreadsPerThreadgroup() /
          threadgroup_width);
  const MTL::Size threads_per_threadgroup(
      threadgroup_width, threadgroup_height, 1);
  compute_encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
}

} // namespace mlx::core
