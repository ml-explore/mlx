// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

//////////////////////////////////////////////////////////////////////
// Case wise reduce dispatch
//////////////////////////////////////////////////////////////////////

inline auto safe_div(size_t n, size_t m) {
  return m == 0 ? 0 : (n + m - 1) / m;
}

inline auto safe_divup(size_t n, size_t m) {
  return safe_div(n, m) * m;
}

inline bool is_64b_int(Dtype dtype) {
  return dtype == int64 || dtype == uint64;
}

// All Reduce
void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d,
    const Stream& s) {
  Dtype out_dtype = out.dtype();
  bool is_out_64b_int = is_64b_int(out_dtype);
  auto kernel = (is_out_64b_int)
      ? d.get_kernel("all_reduce_no_atomics_" + op_name + type_to_name(in))
      : d.get_kernel("all_reduce_" + op_name + type_to_name(in));

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
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder->dispatchThreads(grid_dims, group_dims);

  } else {
    // Allocate intermediate array to store partial reduction results
    size_t intermediate_size = n_thread_groups;
    array intermediate =
        array({static_cast<int>(intermediate_size)}, out_dtype, nullptr, {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    std::vector<array> intermediates = {intermediate};

    // First dispatch
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, intermediate, 1);
    compute_encoder->setBytes(&in_size, sizeof(size_t), 2);
    compute_encoder->dispatchThreads(grid_dims, group_dims);

    // Second pass to reduce intermediate reduction results written to DRAM
    set_array_buffer(compute_encoder, intermediate, 0);
    set_array_buffer(compute_encoder, out, 1);
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
    compute_encoder->dispatchThreads(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [intermediates](MTL::CommandBuffer*) mutable {
          intermediates.clear();
        });
  }
}

void row_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d,
    const Stream& s) {
  Dtype out_dtype = out.dtype();
  bool is_out_64b_int = is_64b_int(out_dtype);

  // Prepare the arguments for the kernel
  size_t reduction_size = plan.shape.back();
  auto shape = plan.shape;
  auto strides = plan.strides;

  shape.pop_back();
  strides.pop_back();

  size_t non_row_reductions = 1;
  for (auto s : shape) {
    non_row_reductions *= static_cast<size_t>(s);
  }
  size_t out_size = out.size();
  auto [rem_shape, rem_strides] = shapes_without_reduction_axes(in, axes);
  for (auto s : rem_shape) {
    shape.push_back(s);
  }
  for (auto s : rem_strides) {
    strides.push_back(s);
  }
  int ndim = shape.size();

  // Determine dispatch kernel
  std::ostringstream kname;

  bool is_small = non_row_reductions * reduction_size < 32;
  bool is_med = non_row_reductions * reduction_size <= 256;
  is_out_64b_int &= !is_small && !is_med;

  std::string small_desc = "_";
  if (is_small) {
    small_desc = "_small_";
  } else if (is_med) {
    small_desc = "_med_";
  }

  small_desc = is_out_64b_int ? "_no_atomics_" : small_desc;

  kname << "row_reduce_general" << small_desc << op_name << type_to_name(in);

  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  // Get dispatch grid dims
  MTL::Size grid_dims;
  MTL::Size group_dims;

  // Each thread handles one output
  if (is_small) {
    grid_dims = MTL::Size(out.size(), 1, 1);
    group_dims = MTL::Size(std::min(1024ul, out.size()), 1, 1);
  }
  // Each simdgroup handles one output
  else if (is_med) {
    grid_dims = MTL::Size(out.size() * 32, 1, 1);
    group_dims = MTL::Size(std::min(8ul, out.size()) * 32, 1, 1);
  }
  // Each theadgroup handles one output
  else {
    int n_reads = REDUCE_N_READS;
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    thread_group_size =
        std::min((reduction_size + n_reads - 1) / n_reads, thread_group_size);

    // Align thread group size with simd_size
    uint simd_size = kernel->threadExecutionWidth();
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    // Launch enough thread groups for each output
    size_t n_threads = out.size() * thread_group_size;
    grid_dims = MTL::Size(n_threads, non_row_reductions, 1);
    group_dims = MTL::Size(thread_group_size, 1, 1);
  }

  // Dispatch kernel
  if (!is_out_64b_int || non_row_reductions == 1) {
    // Set the arguments for the kernel
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
    compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 4);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);
    compute_encoder->dispatchThreads(grid_dims, group_dims);

  } else {
    // Allocate intermediate array to store partial reduction results
    array intermediate = array(
        {static_cast<int>(out.size()), static_cast<int>(non_row_reductions)},
        out_dtype,
        nullptr,
        {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    std::vector<array> intermediates = {intermediate};

    // Set the arguments for the kernel
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, intermediate, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
    compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 4);
    compute_encoder->setBytes(shape.data(), shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        strides.data(), strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);
    compute_encoder->dispatchThreads(grid_dims, group_dims);

    // Set up second dispatch
    reduction_size = non_row_reductions;
    out_size = 1;

    // Shape of axes that aren't participating in reduction remains unchanged.
    std::vector<int> new_shape = rem_shape;

    // Update their strides since they'll be different post partial reduction in
    // first compute dispatch.
    std::vector<size_t> new_strides = rem_strides;
    new_strides.back() = reduction_size;
    for (int i = new_shape.size() - 2; i >= 0; i--) {
      new_strides[i] = new_shape[i + 1] * new_strides[i + 1];
    }
    ndim = new_shape.size();

    // Set the arguments for the kernel
    set_array_buffer(compute_encoder, intermediate, 0);
    set_array_buffer(compute_encoder, out, 1);
    compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
    compute_encoder->setBytes(&out_size, sizeof(size_t), 3);
    compute_encoder->setBytes(&non_row_reductions, sizeof(size_t), 4);
    compute_encoder->setBytes(
        new_shape.data(), new_shape.size() * sizeof(int), 5);
    compute_encoder->setBytes(
        new_strides.data(), new_strides.size() * sizeof(size_t), 6);
    compute_encoder->setBytes(&ndim, sizeof(int), 7);

    // Each thread group is responsible for 1 output
    int n_reads = REDUCE_N_READS;
    size_t thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    thread_group_size =
        std::min((reduction_size + n_reads - 1) / n_reads, thread_group_size);

    // Align thread group size with simd_size
    uint simd_size = kernel->threadExecutionWidth();
    thread_group_size =
        (thread_group_size + simd_size - 1) / simd_size * simd_size;
    assert(thread_group_size <= kernel->maxTotalThreadsPerThreadgroup());

    // Launch enough thread groups for each output
    size_t n_threads = thread_group_size;
    grid_dims = MTL::Size(n_threads, out.size(), 1);
    group_dims = MTL::Size(thread_group_size, 1, 1);

    compute_encoder->dispatchThreads(grid_dims, group_dims);

    d.get_command_buffer(s.index)->addCompletedHandler(
        [intermediates](MTL::CommandBuffer*) mutable {
          intermediates.clear();
        });
  }
}

void strided_reduce_general_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const ReductionPlan& plan,
    const std::vector<int>& axes,
    MTL::ComputeCommandEncoder* compute_encoder,
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
    auto kernel =
        d.get_kernel("col_reduce_small_" + op_name + type_to_name(in));
    compute_encoder->setComputePipelineState(kernel);

    // Select block dims
    MTL::Size grid_dims = MTL::Size(out_size, 1, 1);
    MTL::Size group_dims = MTL::Size(256ul, 1, 1);

    if (non_col_ndim == 0) {
      non_col_shapes = {1};
      non_col_strides = {1};
    }

    // Encode arrays
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
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
    compute_encoder->dispatchThreads(grid_dims, group_dims);

    return;
  }

  // Select kernel
  bool is_out_64b_int = is_64b_int(out_dtype);
  auto kernel = (is_out_64b_int)
      ? d.get_kernel(
            "col_reduce_general_no_atomics_" + op_name + type_to_name(in))
      : d.get_kernel("col_reduce_general_" + op_name + type_to_name(in));

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
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
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
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

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
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, intermediate, 1);
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
    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);

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

    auto row_reduce_kernel = d.get_kernel(
        "row_reduce_general_no_atomics_" + op_name +
        type_to_name(intermediate));
    compute_encoder->setComputePipelineState(row_reduce_kernel);
    set_array_buffer(compute_encoder, intermediate, 0);
    set_array_buffer(compute_encoder, out, 1);
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

    compute_encoder->dispatchThreads(grid_dims, group_dims);

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
      op_name = out.dtype() == bool_ ? "and" : "min_";
      break;
    case Reduce::Max:
      op_name = out.dtype() == bool_ ? "or" : "max_";
      break;
  }

  // Initialize output
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto compute_encoder = d.get_command_encoder(s.index);
  {
    auto kernel = d.get_kernel("i" + op_name + type_to_name(out));
    size_t nthreads = out.size();
    MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
    NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > nthreads) {
      thread_group_size = nthreads;
    }
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(compute_encoder, out, 0);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }

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
}

} // namespace mlx::core
