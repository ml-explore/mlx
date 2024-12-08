// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s) {
  assert(inputs.size() == 3);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  TernaryOpType topt = get_ternary_op_type(a, b, c);

  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto maybe_collapse = [topt, &a, &b, &c, &out]() {
    if (topt == TernaryOpType::General) {
      auto [shape, strides] = collapse_contiguous_dims(a, b, c, out);
      return std::make_tuple(
          shape, strides[0], strides[1], strides[2], strides[3]);
    } else {
      Strides e;
      return std::make_tuple(Shape{}, e, e, e, e);
    }
  };
  auto [shape, strides_a, strides_b, strides_c, strides_out] = maybe_collapse();

  bool large = out.data_size() > UINT_MAX;
  auto ndim = shape.size();
  int work_per_thread;
  if (topt == TernaryOpType::General) {
    large |=
        (a.data_size() > UINT32_MAX || b.data_size() > UINT32_MAX ||
         c.data_size() > UINT32_MAX);
    work_per_thread = large ? 4 : 2;
  } else {
    work_per_thread = 1;
  }
  std::string kernel_name;
  if (topt == TernaryOpType::General) {
    kernel_name = "g";
    if (shape.size() <= 3) {
      kernel_name += std::to_string(shape.size());
    } else if (work_per_thread > 1) {
      concatenate(kernel_name, "n", std::to_string(work_per_thread));
    }
    if (large) {
      kernel_name += "large";
    }
  } else if (large) {
    kernel_name = "v2";
  } else {
    kernel_name = "v";
  }
  concatenate(kernel_name, "_", op, type_to_name(b));

  auto& d = metal::device(s.device);

  auto kernel = get_ternary_kernel(d, kernel_name, out.dtype(), op);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  bool donate_c = c.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? out : a, 0);
  compute_encoder.set_input_array(donate_b ? out : b, 1);
  compute_encoder.set_input_array(donate_c ? out : c, 2);
  compute_encoder.set_output_array(out, 3);

  auto thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (topt == TernaryOpType::General) {
    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);

    if (ndim > 3) {
      compute_encoder.set_vector_bytes(shape, 4);
      compute_encoder.set_vector_bytes(strides_a, 5);
      compute_encoder.set_vector_bytes(strides_b, 6);
      compute_encoder.set_vector_bytes(strides_c, 7);

      compute_encoder.set_bytes(ndim, 8);
      dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder.set_vector_bytes(strides_a, 4);
      compute_encoder.set_vector_bytes(strides_b, 5);
      compute_encoder.set_vector_bytes(strides_c, 6);
    }

    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::ternary] Must use 1024 sized block");
    }
    MTL::Size group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    // Launch a 1D or 2D grid of threads
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

void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt, true /* donate_with_move */);
  ternary_op_gpu_inplace(inputs, out, op, s);
}

void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  auto& s = out.primitive().stream();
  ternary_op_gpu(inputs, out, op, s);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  ternary_op_gpu(inputs, out, get_primitive_string(this));
}

} // namespace mlx::core
