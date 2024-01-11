// Copyright © 2023 Apple Inc.

#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  if (ctype == CopyType::Vector) {
    out.set_data(
        allocator::malloc_or_wait(in.data_size() * out.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }
  if (out.size() == 0) {
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu(const array& in, array& out, CopyType ctype) {
  copy_gpu(in, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(in, out);
  auto& strides_in = strides[0];
  auto& strides_out = strides[1];

  auto& d = metal::device(s.device);
  std::ostringstream kname;
  switch (ctype) {
    case CopyType::Scalar:
      kname << "scopy";
      break;
    case CopyType::Vector:
      kname << "vcopy";
      break;
    case CopyType::General:
      kname << "gcopy";
      break;
    case CopyType::GeneralGeneral:
      kname << "ggcopy";
      break;
  }
  kname << type_to_name(in) << type_to_name(out);
  if ((ctype == CopyType::General || ctype == CopyType::GeneralGeneral) &&
      shape.size() <= MAX_COPY_SPECIALIZED_DIMS) {
    kname << "_" << shape.size();
  }
  auto kernel = d.get_kernel(kname.str());
  auto compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    size_t ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 2);
      compute_encoder->setBytes(strides_in.data(), ndim * sizeof(size_t), 3);
      if (ctype == CopyType::GeneralGeneral) {
        compute_encoder->setBytes(strides_out.data(), ndim * sizeof(size_t), 4);
      }
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_in.data(), ndim * sizeof(size_t), 2);
      if (ctype == CopyType::GeneralGeneral) {
        compute_encoder->setBytes(strides_out.data(), ndim * sizeof(size_t), 3);
      }
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(
          &ndim, sizeof(int), (ctype == CopyType::GeneralGeneral) ? 5 : 4);
    }

    int dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    int dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    int rest = in.size() / (dim0 * dim1);

    // NB assuming thread_group_size is a power of 2 larger than 32 x 32
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::copy] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    size_t nthreads = out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }
}

} // namespace mlx::core
