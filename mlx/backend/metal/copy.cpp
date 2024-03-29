// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  if (ctype == CopyType::Vector) {
    // If the input is donateable, we are doing a vector copy and the types
    // have the same size, then the input buffer can hold the output.
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.move_shared_buffer(in);
      // If the output has the same type as the input then there is nothing to
      // copy, just use the buffer.
      if (in.dtype() == out.dtype()) {
        return;
      }
    } else {
      out.set_data(
          allocator::malloc_or_wait(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
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

template <typename stride_t>
void copy_gpu_inplace(
    const array& in,
    array& out,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& strides_in_pre,
    const std::vector<stride_t>& strides_out_pre,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s) {
  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(
      data_shape, std::vector{strides_in_pre, strides_out_pre});
  auto& strides_in_ = strides[0];
  auto& strides_out_ = strides[1];

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
  bool donate_in = in.data_shared_ptr() == nullptr;

  inp_offset *= size_of(in.dtype());
  out_offset *= size_of(out.dtype());

  set_array_buffer(compute_encoder, donate_in ? out : in, inp_offset, 0);
  set_array_buffer(compute_encoder, out, out_offset, 1);

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    int ndim = shape.size();
    std::vector<int64_t> strides_in{strides_in_.begin(), strides_in_.end()};
    std::vector<int64_t> strides_out{strides_out_.begin(), strides_out_.end()};

    if (ndim > 3) {
      set_vector_bytes(compute_encoder, shape, ndim, 2);
    }
    set_vector_bytes(compute_encoder, strides_in, ndim, 3);
    if (ctype == CopyType::GeneralGeneral) {
      set_vector_bytes(compute_encoder, strides_out, ndim, 4);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 5);
    }

    int dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    int dim1 = ndim > 1 ? shape[ndim - 2] : 1;

    size_t data_size = 1;
    for (auto& s : shape)
      data_size *= s;
    int rest = data_size / (dim0 * dim1);

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

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  return copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out.strides(), 0, 0, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const std::vector<int64_t>& istride,
    int64_t ioffset,
    CopyType ctype,
    const Stream& s) {
  std::vector<int64_t> ostrides{out.strides().begin(), out.strides().end()};
  return copy_gpu_inplace(
      in, out, in.shape(), istride, ostrides, ioffset, 0, ctype, s);
}

} // namespace mlx::core
