// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"

#include <cassert>

namespace mlx::core {

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  if (in.size() == 0) {
    init_reduce(encoder, in, out, reduce_type_);
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  bool broadcasted = false;
  for (int i = 0, j = 0; i < in.ndim() && !broadcasted; i++) {
    if (j < axes_.size() && axes_[j] == i) {
      j++;
    } else {
      broadcasted = in.strides(i) == 0;
    }
  }
  if (plan.type == GeneralReduce || broadcasted || !in.flags().contiguous) {
    array in_copy = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  if (plan.type == ContiguousAllReduce) {
    all_reduce(encoder, in, out, reduce_type_);
    return;
  }

  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    row_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  if (plan.type == ContiguousStridedReduce ||
      plan.type == GeneralStridedReduce) {
    col_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  throw std::runtime_error("No plan reached in reduce.");
}

// Initialize output with identity value
void init_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  out.set_data(allocator::malloc(out.nbytes()));
  
  // Fill with identity value based on reduce type
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (reduce_type) {
      case Reduce::Sum:
        hipMemsetAsync(out.data<void>(), 0, out.nbytes(), stream);
        break;
      case Reduce::Prod: {
        // Need to fill with 1
        if (out.dtype() == float32) {
          float one = 1.0f;
          hipMemcpyAsync(out.data<void>(), &one, sizeof(float), hipMemcpyHostToDevice, stream);
        }
        break;
      }
      default:
        // For min/max, we'd need to fill with appropriate values
        hipMemsetAsync(out.data<void>(), 0, out.nbytes(), stream);
        break;
    }
  });
}

// All reduce implementation
void all_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type) {
  out.set_data(allocator::malloc(out.nbytes()));
  
  bool large = in.size() > INT32_MAX;
  int block_size = 256;
  int num_blocks = std::min((in.size() + block_size - 1) / block_size, (size_t)1024);
  
  encoder.launch_kernel([&](hipStream_t stream) {
    // Initialize output to identity
    hipMemsetAsync(out.data<void>(), 0, out.nbytes(), stream);
    
    switch (in.dtype()) {
      case float32:
        if (reduce_type == Reduce::Sum) {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<float, rocm::ReduceSum, int64_t>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in.data<float>(), out.data<float>(), static_cast<int64_t>(in.size()),
                rocm::ReduceSum{});
          } else {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<float, rocm::ReduceSum, int32_t>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in.data<float>(), out.data<float>(), static_cast<int32_t>(in.size()),
                rocm::ReduceSum{});
          }
        }
        break;
      case int32:
        if (reduce_type == Reduce::Sum) {
          if (large) {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<int32_t, rocm::ReduceSum, int64_t>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in.data<int32_t>(), out.data<int32_t>(), static_cast<int64_t>(in.size()),
                rocm::ReduceSum{});
          } else {
            hipLaunchKernelGGL(
                (rocm::all_reduce_kernel<int32_t, rocm::ReduceSum, int32_t>),
                dim3(num_blocks), dim3(block_size), 0, stream,
                in.data<int32_t>(), out.data<int32_t>(), static_cast<int32_t>(in.size()),
                rocm::ReduceSum{});
          }
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for all_reduce");
    }
  });
}

// Row reduce implementation
void row_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  out.set_data(allocator::malloc(out.nbytes()));
  
  int64_t reduce_size = plan.shape.back();
  int64_t out_size = out.size();
  
  int block_size = 256;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (in.dtype()) {
      case float32:
        if (reduce_type == Reduce::Sum) {
          hipLaunchKernelGGL(
              (rocm::row_reduce_kernel<float, rocm::ReduceSum, int64_t>),
              dim3(out_size), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, out_size,
              rocm::ReduceSum{});
        } else if (reduce_type == Reduce::Max) {
          hipLaunchKernelGGL(
              (rocm::row_reduce_kernel<float, rocm::ReduceMax, int64_t>),
              dim3(out_size), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, out_size,
              rocm::ReduceMax{});
        } else if (reduce_type == Reduce::Min) {
          hipLaunchKernelGGL(
              (rocm::row_reduce_kernel<float, rocm::ReduceMin, int64_t>),
              dim3(out_size), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, out_size,
              rocm::ReduceMin{});
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for row_reduce");
    }
  });
}

// Column reduce implementation
void col_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  out.set_data(allocator::malloc(out.nbytes()));
  
  int64_t reduce_size = plan.shape[0];
  int64_t reduce_stride = plan.strides[0];
  int64_t out_size = out.size();
  
  int block_size = 256;
  int num_blocks = (out_size + block_size - 1) / block_size;
  
  encoder.launch_kernel([&](hipStream_t stream) {
    switch (in.dtype()) {
      case float32:
        if (reduce_type == Reduce::Sum) {
          hipLaunchKernelGGL(
              (rocm::col_reduce_kernel<float, rocm::ReduceSum, int64_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, reduce_stride, out_size,
              rocm::ReduceSum{});
        } else if (reduce_type == Reduce::Max) {
          hipLaunchKernelGGL(
              (rocm::col_reduce_kernel<float, rocm::ReduceMax, int64_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, reduce_stride, out_size,
              rocm::ReduceMax{});
        } else if (reduce_type == Reduce::Min) {
          hipLaunchKernelGGL(
              (rocm::col_reduce_kernel<float, rocm::ReduceMin, int64_t>),
              dim3(num_blocks), dim3(block_size), 0, stream,
              in.data<float>(), out.data<float>(), reduce_size, reduce_stride, out_size,
              rocm::ReduceMin{});
        }
        break;
      default:
        throw std::runtime_error("Unsupported type for col_reduce");
    }
  });
}

} // namespace mlx::core
