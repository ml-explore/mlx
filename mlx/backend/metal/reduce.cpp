// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <iostream>
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
  kname << "rowSmall" << (plan.shape.size() - 1) << "_reduce_" << op_name
        << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Prepare the arguments for the kernel
  int row_size = plan.shape.back();
  auto reduce_shape = plan.shape;
  auto reduce_strides = plan.strides;
  reduce_shape.pop_back();
  reduce_strides.pop_back();
  int reduce_ndim = reduce_shape.size();
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  int ndim = shape.size();
  int non_row_reductions = 1;
  for (auto s : reduce_shape) {
    non_row_reductions *= s;
  }

  // Figure out the grid dims
  MTL::Size grid_dims;
  MTL::Size group_dims;
  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
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
  compute_encoder->setBytes(&row_size, sizeof(int), 2);
  compute_encoder->setBytes(&non_row_reductions, sizeof(int), 3);
  compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
  compute_encoder->setBytes(strides.data(), strides.size() * sizeof(size_t), 5);
  compute_encoder->setBytes(&ndim, sizeof(int), 6);
  compute_encoder->setBytes(
      reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
  compute_encoder->setBytes(
      reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
  compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);
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
  kname << "rowLooped" << (plan.shape.size() - 1) << "_reduce_" << op_name
        << type_to_name(in);
  auto kernel = get_reduce_kernel(d, kname.str(), op_name, in, out);
  compute_encoder->setComputePipelineState(kernel);

  // Prepare the arguments for the kernel
  int row_size = plan.shape.back();
  auto reduce_shape = plan.shape;
  auto reduce_strides = plan.strides;
  reduce_shape.pop_back();
  reduce_strides.pop_back();
  int reduce_ndim = reduce_shape.size();
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  int ndim = shape.size();
  size_t non_row_reductions = 1;
  for (auto s : reduce_shape) {
    non_row_reductions *= s;
  }

  // Figure out the grid
  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  int threadgroup_size = threadgroup_size_from_row_size(row_size);
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&row_size, sizeof(int), 2);
  compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 3);
  compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 4);
  compute_encoder->setBytes(strides.data(), strides.size() * sizeof(size_t), 5);
  compute_encoder->setBytes(&ndim, sizeof(int), 6);
  compute_encoder->setBytes(
      reduce_shape.data(), reduce_shape.size() * sizeof(int), 7);
  compute_encoder->setBytes(
      reduce_strides.data(), reduce_strides.size() * sizeof(size_t), 8);
  compute_encoder->setBytes(&reduce_ndim, sizeof(int), 9);
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

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  Dtype out_dtype = out.dtype();

  // Prepare the arguments for the kernel
  size_t reduction_size = plan.shape.back();
  size_t reduction_stride = plan.strides.back();
  size_t out_size = out.size();
  auto shape = plan.shape;
  auto strides = plan.strides;
  shape.pop_back();
  strides.pop_back();
  size_t non_col_reductions = 1;
  for (auto s : shape) {
    non_col_reductions *= static_cast<size_t>(s);
  }

  std::vector<int> non_col_shapes = shape;
  std::vector<size_t> non_col_strides = strides;
  int non_col_ndim = shape.size();

  auto [rem_shape, rem_strides] = shapes_without_reduction_axes(in, axes);
  for (auto s : rem_shape) {
    shape.push_back(s);
  }
  for (auto s : rem_strides) {
    strides.push_back(s);
  }
  int ndim = shape.size();

  // Specialize for small dims
  if (reduction_size * non_col_reductions < 16) {
    // Select kernel
    auto kernel = get_reduce_kernel(
        d, "colSmall_reduce_" + op_name + type_to_name(in), op_name, in, out);
    compute_encoder->setComputePipelineState(kernel);

    // Select block dims
    MTL::Size grid_dims = MTL::Size(out_size, 1, 1);
    MTL::Size group_dims = MTL::Size(256ul, 1, 1);

    if (non_col_ndim == 0) {
      non_col_shapes = {1};
      non_col_strides = {1};
    }

    // Encode arrays
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 4);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);
    compute_encoder->setBytes(&non_col_reductions, sizeof(size_t), 8);
    compute_encoder->setBytes(
        non_col_shapes.data(), non_col_shapes.size() * sizeof(int), 9);
    compute_encoder->setBytes(
        non_col_strides.data(), non_col_shapes.size() * sizeof(size_t), 10);
    compute_encoder->setBytes(&non_col_ndim, sizeof(int), 11);

    // Dispatch threads
    compute_encoder.dispatchThreads(grid_dims, group_dims);

    return;
  }

  // Select kernel
  bool is_out_64b_int = is_64b_int(out_dtype);
  std::string kernel_name = "colGeneral";
  if (is_out_64b_int) {
    kernel_name += "NoAtomics";
  }
  kernel_name += "_reduce_" + op_name + type_to_name(in);
  auto kernel = get_reduce_kernel(d, kernel_name, op_name, in, out);

  compute_encoder->setComputePipelineState(kernel);

  // Select block dimensions
  // Each thread reads 16 inputs to give it more work
  uint n_inputs_per_thread = REDUCE_N_READS;
  uint n_threads_per_output =
      (reduction_size + n_inputs_per_thread - 1) / n_inputs_per_thread;

  // We spread outputs over the x dimension and inputs over the y dimension
  // Threads with the same lid.x in a given threadgroup work on the same
  // output and each thread in the y dimension accumulates for that output

  // Threads with same lid.x, i.e. each column of threads work on same output
  uint threadgroup_dim_x = std::min(out_size, 128ul);

  // Number of threads along y, is dependent on number of reductions needed.
  uint threadgroup_dim_y =
      kernel->maxTotalThreadsPerThreadgroup() / threadgroup_dim_x;
  threadgroup_dim_y = std::min(n_threads_per_output, threadgroup_dim_y);

  // Derive number of thread groups along x, based on how many threads we need
  // along x
  uint n_threadgroups_x =
      (out_size + threadgroup_dim_x - 1) / threadgroup_dim_x;

  // Derive number of thread groups along y based on how many threads we need
  // along y
  uint n_threadgroups_y =
      (n_threads_per_output + threadgroup_dim_y - 1) / threadgroup_dim_y;

  // Launch enough thread groups for each output
  MTL::Size grid_dims =
      MTL::Size(n_threadgroups_x, n_threadgroups_y, non_col_reductions);
  MTL::Size group_dims = MTL::Size(threadgroup_dim_x, threadgroup_dim_y, 1);

  if (is_out_64b_int == false) {
    // Set the arguments for the kernel
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 4);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);

    // We set shared memory to be exploited here for reductions within a
    // threadgroup - each thread must be able to update its accumulated output
    // Note: Each threadgroup should have 32kB of data in threadgroup memory
    //       and threadgroup_dim_x * threadgroup_dim_y <= 1024 by design
    //       This should be fine for floats, but we might need to revisit
    //       if we ever come to doubles. In that case, we should also cut
    //       down the number of threads we launch in a threadgroup
    compute_encoder->setThreadgroupMemoryLength(
        safe_divup(threadgroup_dim_x * threadgroup_dim_y * out.itemsize(), 16),
        0);
    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

  } else {
    // Allocate intermediate array to store reduction results from all thread
    // groups
    array intermediate = array(
        {static_cast<int>(out.size()),
         static_cast<int>(n_threadgroups_y * non_col_reductions)},
        out_dtype,
        nullptr,
        {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    std::vector<array> intermediates = {intermediate};

    // Set the arguments for the kernel
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(intermediate, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 4);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);

    // We set shared memory to be exploited here for reductions within a
    // threadgroup - each thread must be able to update its accumulated output
    // Note: Each threadgroup should have 32kB of data in threadgroup memory
    //       and threadgroup_dim_x * threadgroup_dim_y <= 1024 by design
    //       This should be fine for floats, but we might need to revisit
    //       if we ever come to doubles. In that case, we should also cut
    //       down the number of threads we launch in a threadgroup
    compute_encoder->setThreadgroupMemoryLength(
        safe_divup(threadgroup_dim_x * threadgroup_dim_y * out.itemsize(), 16),
        0);
    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);

    // Perform second pass of reductions
    // Reduce results of threadgroups along y, z from first pass, that
    // collectively work on each output element.
    reduction_size = n_threadgroups_y * non_col_reductions;
    out_size = 1;

    // Shape of axes that aren't participating in reduction remains unchanged.
    std::vector<int> new_shape = rem_shape;

    // Update their strides since they'll be different after a partial reduction
    // post first compute dispatch.
    std::vector<size_t> new_strides = rem_strides;
    new_strides.back() = reduction_size;
    for (int i = new_shape.size() - 2; i >= 0; i--) {
      new_strides[i] = new_shape[i + 1] * new_strides[i + 1];
    }
    ndim = new_shape.size();

    std::string kernel_name =
        "rowGeneralNoAtomics_reduce_" + op_name + type_to_name(intermediate);
    auto row_reduce_kernel =
        get_reduce_kernel(d, kernel_name, op_name, intermediate, out);

    compute_encoder->setComputePipelineState(row_reduce_kernel);
    compute_encoder.set_input_array(intermediate, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 4);
    compute_encoder->setBytes(
        new_shape.data(), new_shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        new_strides.data(), new_strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);

    // Each thread group is responsible for 1 output
    size_t n_reads = REDUCE_N_READS;
    size_t thread_group_size =
        row_reduce_kernel->maxTotalThreadsPerThreadgroup();
    thread_group_size =
        std::min((reduction_size + n_reads - 1) / n_reads, thread_group_size);

    // Align thread group size with simd_size
    uint simd_size = row_reduce_kernel->threadExecutionWidth();
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    // Launch enough thread groups for each output
    uint n_threads = thread_group_size;
    grid_dims = MTL::Size(n_threads, out.size(), 1);
    group_dims = MTL::Size(thread_group_size, 1, 1);

    compute_encoder.dispatchThreads(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [intermediates](MTL::CommandBuffer*) mutable {
          intermediates.clear();
        });
  }
}

//////////////////////////////////////////////////////////////////////
// Main reduce dispatch
//////////////////////////////////////////////////////////////////////

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());

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
      init_reduce(out, op_name, compute_encoder, d, s);
      strided_reduce_general_dispatch(
          in, out, op_name, plan, axes_, compute_encoder, d, s);
    }

    if (!copies.empty()) {
      d.get_command_buffer(s.index)->addCompletedHandler(
          [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
    }
  }
}

} // namespace mlx::core
