// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

struct RowReduceArgs {
  // Input shape and strides not including the reduction axes
  std::vector<int> shape;
  std::vector<size_t> strides;
  int ndim;

  // Input shape and strides for the reduction axes
  std::vector<int> reduce_shape;
  std::vector<size_t> reduce_strides;
  int reduce_ndim;

  // The number of rows we are reducing. Namely prod(reduce_shape).
  size_t non_row_reductions;

  // The size of the row.
  int row_size;

  RowReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    row_size = plan.shape.back();

    reduce_shape = plan.shape;
    reduce_strides = plan.strides;
    reduce_shape.pop_back();
    reduce_strides.pop_back();
    reduce_ndim = reduce_shape.size();

    non_row_reductions = 1;
    for (auto s : reduce_shape) {
      non_row_reductions *= s;
    }

    std::tie(shape, strides) = shapes_without_reduction_axes(in, axes);
    std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
    ndim = shape.size();
  }

  void encode(CommandEncoder& compute_encoder) {
    // Push 0s to avoid encoding empty vectors.
    if (reduce_ndim == 0) {
      reduce_shape.push_back(0);
      reduce_strides.push_back(0);
    }
    if (ndim == 0) {
      shape.push_back(0);
      strides.push_back(0);
    }

    compute_encoder->setBytes(&row_size, sizeof(int), 2);
    compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 3);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 5);
    compute_encoder->setBytes(&ndim, sizeof(int), 6);
    compute_encoder->setBytes(
        reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
    compute_encoder->setBytes(
        reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
    compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);

    if (reduce_ndim == 0) {
      reduce_shape.pop_back();
      reduce_strides.pop_back();
    }
    if (ndim == 0) {
      shape.pop_back();
      strides.pop_back();
    }
  }
};

struct ColReduceArgs {
  // Input shape and strides not including the reduction axes
  std::vector<int> shape;
  std::vector<size_t> strides;
  int ndim;

  // Input shape and strides for the reduction axes
  std::vector<int> reduce_shape;
  std::vector<size_t> reduce_strides;
  int reduce_ndim;

  // The number of column reductions we are doing. Namely prod(reduce_shape).
  size_t non_col_reductions;

  // The size of the contiguous column reduction.
  size_t reduction_size;
  size_t reduction_stride;

  ColReduceArgs(
      const array& in,
      const ReductionPlan& plan,
      const std::vector<int>& axes) {
    reduction_size = plan.shape.back();
    reduction_stride = plan.strides.back();

    reduce_shape = plan.shape;
    reduce_strides = plan.strides;
    reduce_shape.pop_back();
    reduce_strides.pop_back();
    reduce_ndim = reduce_shape.size();

    non_col_reductions = 1;
    for (auto s : reduce_shape) {
      non_col_reductions *= s;
    }

    std::tie(shape, strides) = shapes_without_reduction_axes(in, axes);
    while (!shape.empty() && strides.back() < reduction_stride &&
           strides.back() > 0) {
      shape.pop_back();
      strides.pop_back();
    }
    std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
    ndim = shape.size();
  }

  void encode(CommandEncoder& compute_encoder) {
    // Push 0s to avoid encoding empty vectors.
    if (reduce_ndim == 0) {
      reduce_shape.push_back(0);
      reduce_strides.push_back(0);
    }
    if (ndim == 0) {
      shape.push_back(0);
      strides.push_back(0);
    }

    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 5);
    compute_encoder->setBytes(&ndim, sizeof(int), 6);
    compute_encoder->setBytes(
        reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
    compute_encoder->setBytes(
        reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
    compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);
    compute_encoder->setBytes(&non_col_reductions, sizeof(size_t), 10);

    if (reduce_ndim == 0) {
      reduce_shape.pop_back();
      reduce_strides.pop_back();
    }
    if (ndim == 0) {
      shape.pop_back();
      strides.pop_back();
    }
  }
};

} // namespace

inline auto safe_div(size_t n, size_t m) {
  return m == 0 ? 0 : (n + m - 1) / m;
}

inline auto safe_divup(size_t n, size_t m) {
  return safe_div(n, m) * m;
}

inline bool is_64b_int(Dtype dtype) {
  return dtype == int64 || dtype == uint64;
}

inline bool is_64b_dtype(Dtype dtype) {
  return dtype == int64 || dtype == uint64 || dtype == complex64;
}

inline int threadgroup_size_from_row_size(int row_size) {
  // 1 simdgroup per row smallish rows
  if (row_size <= 512) {
    return 32;
  }

  // 2 simdgroups per row for medium rows
  if (row_size <= 1024) {
    return 64;
  }

  // up to 32 simdgroups after that
  int thread_group_size;
  thread_group_size = (row_size + REDUCE_N_READS - 1) / REDUCE_N_READS;
  thread_group_size = ((thread_group_size + 31) / 32) * 32;
  thread_group_size = std::min(1024, thread_group_size);
  return thread_group_size;
}

inline auto output_grid_for_col_reduce(
    const array& out,
    const ColReduceArgs& args) {
  auto out_shape = out.shape();
  auto out_strides = out.strides();
  while (!out_shape.empty() && out_strides.back() < args.reduction_stride) {
    out_shape.pop_back();
    out_strides.pop_back();
  }
  return get_2d_grid_dims(out_shape, out_strides);
}

void init_reduce(
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto kernel =
      get_reduce_init_kernel(d, "i_reduce_" + op_name + type_to_name(out), out);
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_output_array(out, 0);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

// All Reduce
void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  Dtype out_dtype = out.dtype();
  bool is_out_64b_int = is_64b_int(out_dtype);
  std::string kernel_name = "all";
  if (is_out_64b_int) {
    kernel_name += "NoAtomics";
  }
  kernel_name += "_reduce_" + op_name + type_to_name(in);
  auto kernel = get_reduce_kernel(d, kernel_name, op_name, in, out);

  compute_encoder->setComputePipelineState(kernel);

  // We make sure each thread has enough to do by making it read in
  // at least n_reads inputs
  int n_reads = REDUCE_N_READS;
  size_t in_size = in.size();

  // mod_in_size gives us the groups of n_reads needed to go over the entire
  // input
  uint mod_in_size = (in_size + n_reads - 1) / n_reads;
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  thread_group_size =
      mod_in_size > thread_group_size ? thread_group_size : mod_in_size;
  uint simd_size = kernel->threadExecutionWidth();
  thread_group_size =
      ((thread_group_size + simd_size - 1) / simd_size) * simd_size;

  // If the number of thread groups needed exceeds 1024, we reuse threads groups
  uint n_thread_groups = safe_div(mod_in_size, thread_group_size);
  n_thread_groups = std::min(n_thread_groups, 1024u);
  uint nthreads = n_thread_groups * thread_group_size;

  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);

  // Encode buffers and dispatch
  if (is_out_64b_int == false || n_thread_groups == 1) {
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

  } else {
    // Allocate intermediate array to store partial reduction results
    size_t intermediate_size = n_thread_groups;
    array intermediate =
        array({static_cast<int>(intermediate_size)}, out_dtype, nullptr, {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    std::vector<array> intermediates = {intermediate};

    // First dispatch
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(intermediate, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

    // Second pass to reduce intermediate reduction results written to DRAM
    compute_encoder.set_input_array(intermediate, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&intermediate_size, sizeof(size_t), 2);

    mod_in_size = (intermediate_size + n_reads - 1) / n_reads;

    thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    thread_group_size =
        mod_in_size > thread_group_size ? thread_group_size : mod_in_size;
    thread_group_size =
        ((thread_group_size + simd_size - 1) / simd_size) * simd_size;

    // If the number of thread groups needed exceeds 1024, we reuse threads
    // groups
    nthreads = thread_group_size;
    group_dims = MTL::Size(thread_group_size, 1, 1);
    grid_dims = MTL::Size(nthreads, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [intermediates](MTL::CommandBuffer*) mutable {
          intermediates.clear();
        });
  }
}

void row_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  std::ostringstream kname;
  int n = (plan.shape.size() <= 5) ? std::max(1ul, plan.shape.size() - 1) : 0;
  kname << "rowSmall" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Prepare the arguments for the kernel
  RowReduceArgs args(in, plan, axes);

  // Figure out the grid dims
  MTL::Size grid_dims;
  MTL::Size group_dims;
  if ((args.non_row_reductions < 32 && args.row_size <= 8) ||
      args.non_row_reductions <= 8) {
    grid_dims = get_2d_grid_dims(out.shape(), out.strides());
    group_dims =
        MTL::Size((grid_dims.width < 1024) ? grid_dims.width : 1024, 1, 1);
  } else {
    auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
    grid_dims = MTL::Size(32, out_grid_size.width, out_grid_size.height);
    group_dims = MTL::Size(32, 1, 1);
  }

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_simple(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  std::ostringstream kname;
  kname << "rowSimple_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Figure out the grid dims
  int row_size = plan.shape.back();
  size_t out_size = out.size();
  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  out_grid_size.width =
      (out_grid_size.width + REDUCE_N_WRITES - 1) / REDUCE_N_WRITES;
  int threadgroup_size = threadgroup_size_from_row_size(row_size);
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&row_size, sizeof(int), 2);
  compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  std::ostringstream kname;
  int n = (plan.shape.size() <= 5) ? std::max(1ul, plan.shape.size() - 1) : 0;
  kname << "rowLooped" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Prepare the arguments for the kernel
  RowReduceArgs args(in, plan, axes);

  // Figure out the grid
  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  int threadgroup_size = threadgroup_size_from_row_size(args.row_size);
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void row_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Case 1: The row is small
  if (plan.shape.back() <= 64) {
    return row_reduce_small(
        in, out, op_name, plan, axes, compute_encoder, d, s);
  }

  // Case 2: Contiguous reduce without non-row reductions
  if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
    return row_reduce_simple(
        in, out, op_name, plan, axes, compute_encoder, d, s);
  }

  // Case 3: General row reduce including non-row reductions
  return row_reduce_looped(in, out, op_name, plan, axes, compute_encoder, d, s);
}

void strided_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Prepare the arguments for the kernel
  ColReduceArgs args(in, plan, axes);

  // Figure out the grid dims
  MTL::Size grid_dims, group_dims;

  // Case 1: everything is small so launch one thread per col reduce
  if (args.reduction_size * args.non_col_reductions < 64) {
    grid_dims = output_grid_for_col_reduce(out, args);
    int threadgroup_size = (grid_dims.width > 128) ? 128 : grid_dims.width;
    group_dims = MTL::Size(threadgroup_size, 1, 1);
  }

  // Case 2: Reduction in the simdgroup
  else {
    args.reduce_shape.push_back(args.reduction_size);
    args.reduce_strides.push_back(args.reduction_stride);
    args.reduce_ndim++;
    int simdgroups =
        (args.reduction_stride + REDUCE_N_READS - 1) / REDUCE_N_READS;
    int threadgroup_size = simdgroups * 32;
    auto out_grid_dims = output_grid_for_col_reduce(out, args);
    grid_dims =
        MTL::Size(threadgroup_size, out_grid_dims.width, out_grid_dims.height);
    group_dims = MTL::Size(threadgroup_size, 1, 1);
  }

  // Set the kernel
  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  std::ostringstream kname;
  kname << "colSmall" << n << "_reduce_" << op_name << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void strided_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Prepare the arguments for the kernel
  ColReduceArgs args(in, plan, axes);
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  // Figure out the grid dims
  auto out_grid_size = output_grid_for_col_reduce(out, args);
  int BN = (args.reduction_stride <= 256) ? 32 : 128;
  int BM = 1024 / BN;
  int threadgroup_size = 4 * 32;
  MTL::Size grid_dims(
      threadgroup_size * ((args.reduction_stride + BN - 1) / BN),
      out_grid_size.width,
      out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Set the kernel
  int n = (args.reduce_ndim < 5) ? std::max(1, args.reduce_ndim) : 0;
  std::ostringstream kname;
  kname << "colLooped" << n << "_" << BM << "_" << BN << "_reduce_" << op_name
        << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatchThreads(grid_dims, group_dims);
}

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  if (plan.strides.back() < 32) {
    return strided_reduce_small(
        in, out, op_name, plan, axes, compute_encoder, d, s);
  }

  return strided_reduce_looped(
      in, out, op_name, plan, axes, compute_encoder, d, s);
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());
  assert(out.size() != in.size());

  // Continue with reduction operation
  // Minimum of 4 bytes since we use size 4 structs for all reduce
  // and metal will complain o/w
  size_t min_bytes = std::max(out.nbytes(), 4ul);
  out.set_data(allocator::malloc_or_wait(min_bytes));
  std::string op_name;
  switch (reduce_type_) {
    case Reduce::And:
      op_name = "and";
      break;
    case Reduce::Or:
      op_name = "or";
      break;
    case Reduce::Sum:
      op_name = "sum";
      break;
    case Reduce::Prod:
      op_name = out.dtype() == bool_ ? "and" : "prod";
      break;
    case Reduce::Min:
      op_name = out.dtype() == bool_ ? "and" : "min";
      break;
    case Reduce::Max:
      op_name = out.dtype() == bool_ ? "or" : "max";
      break;
  }

  // Initialize output
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s.index);

  // Reduce
  if (in.size() > 0) {
    std::vector<array> copies;
    ReductionPlan plan = get_reduction_plan(in, axes_);

    // If it is a general reduce then copy the input to a contiguous array and
    // recompute the plan.
    //
    // TODO: This can be avoided by making the output have the same strides as
    //       input for the axes with stride smaller than the minimum reduction
    //       stride.
    if (plan.type == GeneralReduce) {
      array in_copy(in.shape(), in.dtype(), nullptr, {});
      copy_gpu(in, in_copy, CopyType::General, s);
      copies.push_back(in_copy);
      in = in_copy;
      plan = get_reduction_plan(in, axes_);
    }

    // Reducing over everything and the data is all there no broadcasting or
    // slicing etc.
    if (plan.type == ContiguousAllReduce) {
      init_reduce(out, op_name, compute_encoder, d, s);
      all_reduce_dispatch(in, out, op_name, compute_encoder, d, s);
    }

    // At least the last dimension is row contiguous and we are reducing over
    // the last dim.
    else if (
        plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
      row_reduce_general_dispatch(
          in, out, op_name, plan, axes_, compute_encoder, d, s);
    }

    // At least the last two dimensions are contiguous and we are doing a
    // strided reduce over these.
    else if (
        plan.type == ContiguousStridedReduce ||
        plan.type == GeneralStridedReduce) {
      strided_reduce_general_dispatch(
          in, out, op_name, plan, axes_, compute_encoder, d, s);
    }

    if (!copies.empty()) {
      d.get_command_buffer(s.index)->addCompletedHandler(
          [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    }
  }

  // Nothing to reduce just set initialize the output
  else {
    init_reduce(out, op_name, compute_encoder, d, s);
  }
}

} // namespace mlx::core
