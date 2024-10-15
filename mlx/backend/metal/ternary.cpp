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
      std::vector<size_t> e;
      return std::make_tuple(std::vector<int>{}, e, e, e, e);
    }
  };
  auto [shape, strides_a, strides_b, strides_c, strides_out] = maybe_collapse();

  bool use_2d = out.data_size() > UINT_MAX;
  auto ndim = shape.size();
  int work_per_thread = (topt == TernaryOpType::General) ? 4 : 1;
  std::string kernel_name;
  {
    std::ostringstream kname;
    if (topt == TernaryOpType::General) {
      kname << "g";
      if (shape.size() <= 3) {
        kname << shape.size();
      } else if (work_per_thread > 1) {
        kname << "n" << work_per_thread;
      }
    } else if (use_2d) {
      kname << "v2";
    } else {
      kname << "v";
    }
    kname << "_" << op << type_to_name(b);
    kernel_name = kname.str();
  }

  auto& d = metal::device(s.device);

  auto kernel = get_ternary_kernel(d, kernel_name, out.dtype(), op);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  bool donate_c = c.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? out : a, 0);
  compute_encoder.set_input_array(donate_b ? out : b, 1);
  compute_encoder.set_input_array(donate_c ? out : c, 2);
  compute_encoder.set_output_array(out, 3);

  if (topt == TernaryOpType::General) {
    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);

    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 4);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 6);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 7);

      compute_encoder->setBytes(&ndim, sizeof(int), 8);
      dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 6);
    }

    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::ternary] Must use 1024 sized block");
    }
    MTL::Size group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D or 2D grid of threads
    size_t nthreads = out.data_size();
    MTL::Size grid_dims = use_2d ? get_2d_grid_dims(out.shape(), out.strides())
                                 : MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
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
