// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/ternary.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

#ifndef MLX_METAL_JIT
#include <fmt/format.h>
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/compiled_includes.h"
#include "mlx/backend/metal/jit/ternary.h"
#endif

namespace mlx::core {

constexpr int MAX_TERNARY_SPECIALIZED_DIMS = 5;

void ternary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  assert(inputs.size() == 3);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt, true /* donate_with_move */);

  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, c, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_c = strides[2];
  auto& strides_out = strides[3];

  std::string kernel_name;
  {
    std::ostringstream kname;
    if (topt == TernaryOpType::General) {
      kname << "g";
      if (shape.size() <= MAX_TERNARY_SPECIALIZED_DIMS) {
        kname << shape.size();
      }
    } else {
      kname << "v";
    }
    kname << "_" << op << type_to_name(b);
    kernel_name = kname.str();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  MTL::ComputePipelineState* kernel;

  if constexpr (mlx_metal_jit()) {
    std::ostringstream op_t;
    out.primitive().print(op_t);
    auto op_name = op_t.str();
    std::string lib_name = kernel_name.substr(kernel_name.find("_") + 1);
    auto lib = d.get_library(lib_name);
    if (lib == nullptr) {
      std::ostringstream kernel_source;
      kernel_source << metal::utils() << metal::ternary_ops()
                    << metal::ternary()
                    << fmt::format(
                           ternary_kernels,
                           lib_name,
                           get_type_string(out.dtype()),
                           op_name);
      lib = d.get_library(lib_name, kernel_source.str());
    }
    kernel = d.get_kernel(kernel_name, lib);
  } else {
    kernel = d.get_kernel(kernel_name);
  }

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_input_array(c, 2);
  compute_encoder.set_output_array(out, 3);

  if (topt == TernaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 4);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 6);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 7);

      if (ndim > MAX_TERNARY_SPECIALIZED_DIMS) {
        compute_encoder->setBytes(&ndim, sizeof(int), 8);
      }
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_c.data(), ndim * sizeof(size_t), 6);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    MTL::Size group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D grid of threads
    size_t nthreads = out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  ternary_op(inputs, out, "select");
}

} // namespace mlx::core
