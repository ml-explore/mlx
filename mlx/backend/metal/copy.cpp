// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_COPY_SPECIALIZED_DIMS = 3;

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
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto maybe_collapse =
      [ctype, &data_shape, &strides_in_pre, &strides_out_pre]() {
        if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
          auto [shape, strides] = collapse_contiguous_dims(
              data_shape,
              std::vector{strides_in_pre, strides_out_pre},
              /* size_cap = */ INT32_MAX);
          return std::make_tuple(shape, strides[0], strides[1]);
        } else {
          std::vector<stride_t> e;
          return std::make_tuple(std::vector<int>{}, e, e);
        }
      };
  auto [shape, strides_in_, strides_out_] = maybe_collapse();
  int ndim = shape.size();
  bool large;
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    // Allow for negative strides
    large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;
  } else {
    large = out.data_size() > UINT32_MAX;
  }
  auto& d = metal::device(s.device);
  int work_per_thread = 1;
  std::string kernel_name;
  switch (ctype) {
    case CopyType::Scalar:
      kernel_name = (large ? "s2" : "s");
      break;
    case CopyType::Vector:
      kernel_name = (large ? "v2" : "v");
      break;
    case CopyType::General:
      kernel_name = "g";
      break;
    case CopyType::GeneralGeneral:
      kernel_name = "gg";
      break;
  }
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    if (shape.size() <= MAX_COPY_SPECIALIZED_DIMS) {
      kernel_name += std::to_string(shape.size());
    } else {
      work_per_thread = large ? 4 : 2;
      concatenate(kernel_name, "n", std::to_string(work_per_thread));
    }
    if (large) {
      kernel_name += "large";
    }
  }
  concatenate(kernel_name, "_copy", type_to_name(in), type_to_name(out));
  auto kernel = get_copy_kernel(d, kernel_name, in, out);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  bool donate_in = in.data_shared_ptr() == nullptr;

  inp_offset *= size_of(in.dtype());
  out_offset *= size_of(out.dtype());

  compute_encoder.set_input_array(donate_in ? out : in, 0, inp_offset);
  compute_encoder.set_output_array(out, 1, out_offset);

  auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    std::vector<int64_t> strides_in{strides_in_.begin(), strides_in_.end()};
    std::vector<int64_t> strides_out{strides_out_.begin(), strides_out_.end()};
    if (ndim > 3) {
      compute_encoder.set_vector_bytes(shape, ndim, 2);
    }
    compute_encoder.set_vector_bytes(strides_in, ndim, 3);
    if (ctype == CopyType::GeneralGeneral) {
      compute_encoder.set_vector_bytes(strides_out, ndim, 4);
    }

    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;

    size_t data_size = 1;
    for (auto& s : shape)
      data_size *= s;
    size_t rest = data_size / (dim0 * dim1);

    if (ndim > MAX_COPY_SPECIALIZED_DIMS) {
      compute_encoder.set_bytes(ndim, 5);
      dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    }

    // NB assuming thread_group_size is a power of 2 larger than 32 x 32
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::copy] Must use 1024 sized block");
    }

    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    size_t nthreads = out.data_size();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    MTL::Size grid_dims = large ? get_2d_grid_dims(out.shape(), out.strides())
                                : MTL::Size(nthreads, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
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
  assert(in.shape() == out.shape());
  std::vector<int64_t> ostrides{out.strides().begin(), out.strides().end()};
  return copy_gpu_inplace(
      in, out, in.shape(), istride, ostrides, ioffset, 0, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  bool large = out.data_size() > UINT32_MAX;
  auto& d = metal::device(s.device);
  std::string kernel_name = std::string(large ? "s2" : "s") + "_copy" +
      type_to_name(val) + type_to_name(out);
  auto kernel = get_copy_kernel(d, kernel_name, val, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(val, 0);
  compute_encoder.set_output_array(out, 1);

  auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  size_t nthreads = out.data_size();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  MTL::Size grid_dims = large ? get_2d_grid_dims(out.shape(), out.strides())
                              : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
