// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/binary.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_BINARY_SPECIALIZED_DIMS = 5;

void binary_op(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::string op) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, outputs[0], bopt, true);
  set_binary_op_output_data(a, b, outputs[1], bopt, true);

  auto& out = outputs[0];
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_out = strides[2];

  std::string kernel_name;
  {
    std::ostringstream kname;
    switch (bopt) {
      case BinaryOpType::ScalarScalar:
        kname << "ss";
        break;
      case BinaryOpType::ScalarVector:
        kname << "sv";
        break;
      case BinaryOpType::VectorScalar:
        kname << "vs";
        break;
      case BinaryOpType::VectorVector:
        kname << "vv";
        break;
      case BinaryOpType::General:
        kname << "g";
        if (shape.size() <= MAX_BINARY_SPECIALIZED_DIMS) {
          kname << shape.size();
        } else {
          kname << "n";
        }
        break;
    }
    kname << op << type_to_name(a);
    kernel_name = kname.str();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto kernel = get_binary_two_kernel(d, kernel_name, a, outputs[0]);

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  // - If a is donated it goes to the first output
  // - If b is donated it goes to the first output if a was not donated
  //   otherwise it goes to the second output
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? outputs[0] : a, 0);
  compute_encoder.set_input_array(
      donate_b ? (donate_a ? outputs[1] : outputs[0]) : b, 1);
  compute_encoder.set_output_array(outputs[0], 2);
  compute_encoder.set_output_array(outputs[1], 3);

  if (bopt == BinaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 4);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 5);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 6);
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 7);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
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

void binary_op(
    const std::vector<array>& inputs,
    array& out,
    const std::string op) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  set_binary_op_output_data(a, b, out, bopt, true);
  if (out.size() == 0) {
    return;
  }

  // Try to collapse contiguous dims
  auto [shape, strides] = collapse_contiguous_dims(a, b, out);
  auto& strides_a = strides[0];
  auto& strides_b = strides[1];
  auto& strides_out = strides[2];

  std::string kernel_name;
  {
    std::ostringstream kname;
    switch (bopt) {
      case BinaryOpType::ScalarScalar:
        kname << "ss";
        break;
      case BinaryOpType::ScalarVector:
        kname << "sv";
        break;
      case BinaryOpType::VectorScalar:
        kname << "vs";
        break;
      case BinaryOpType::VectorVector:
        kname << "vv";
        break;
      case BinaryOpType::General:
        kname << "g";
        if (shape.size() <= MAX_BINARY_SPECIALIZED_DIMS) {
          kname << shape.size();
        } else {
          kname << "n";
        }
        break;
    }
    kname << op << type_to_name(a);
    kernel_name = kname.str();
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  auto kernel = get_binary_kernel(d, kernel_name, a, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  bool donate_a = a.data_shared_ptr() == nullptr;
  bool donate_b = b.data_shared_ptr() == nullptr;
  compute_encoder.set_input_array(donate_a ? out : a, 0);
  compute_encoder.set_input_array(donate_b ? out : b, 1);
  compute_encoder.set_output_array(out, 2);

  if (bopt == BinaryOpType::General) {
    auto ndim = shape.size();
    if (ndim > 3) {
      compute_encoder->setBytes(shape.data(), ndim * sizeof(int), 3);
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 4);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 5);
    } else {
      // The shape is implicit in the grid for <= 3D
      compute_encoder->setBytes(strides_a.data(), ndim * sizeof(size_t), 3);
      compute_encoder->setBytes(strides_b.data(), ndim * sizeof(size_t), 4);
    }

    if (ndim > MAX_BINARY_SPECIALIZED_DIMS) {
      compute_encoder->setBytes(&ndim, sizeof(int), 6);
    }

    // Launch up to 3D grid of threads
    size_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
    size_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
    size_t rest = out.size() / (dim0 * dim1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size != 1024) {
      throw std::runtime_error("[Metal::binary] Must use 1024 sized block");
    }
    auto group_dims = get_block_dims(dim0, dim1, rest);
    MTL::Size grid_dims = MTL::Size(dim0, dim1, rest);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    // Launch a 1D grid of threads
    size_t nthreads =
        bopt == BinaryOpType::General ? out.size() : out.data_size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "add");
}

void ArcTan2::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "arctan2");
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  switch (op_) {
    case BitwiseBinary::And:
      binary_op(inputs, out, "bitwise_and");
      break;
    case BitwiseBinary::Or:
      binary_op(inputs, out, "bitwise_or");
      break;
    case BitwiseBinary::Xor:
      binary_op(inputs, out, "bitwise_xor");
      break;
    case BitwiseBinary::LeftShift:
      binary_op(inputs, out, "left_shift");
      break;
    case BitwiseBinary::RightShift:
      binary_op(inputs, out, "right_shift");
      break;
  }
}

void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "div");
}

void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  binary_op(inputs, outputs, "divmod");
}

void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "rem");
}

void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, equal_nan_ ? "naneq" : "eq");
}

void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "ge");
}

void GreaterEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "geq");
}

void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "le");
}

void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "leq");
}

void LogicalAnd::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "land");
}

void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "lor");
}

void LogAddExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "lae");
}

void Maximum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "max");
}

void Minimum::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "min");
}

void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "mul");
}

void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "neq");
}

void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "pow");
}

void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  binary_op(inputs, out, "sub");
}

} // namespace mlx::core
