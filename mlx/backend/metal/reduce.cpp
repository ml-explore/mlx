#include <algorithm>
#include <cassert>
#include <sstream>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

//////////////////////////////////////////////////////////////////////
// Case wise reduce dispatch
//////////////////////////////////////////////////////////////////////

namespace {

// All Reduce
void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d) {
  // Get kernel and encode buffers
  size_t in_size = in.size();
  auto kernel = d.get_kernel("all_reduce_" + op_name + type_to_name(in));

  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(&in_size, sizeof(size_t), 2);

  // Set grid dimensions

  // We make sure each thread has enough to do by making it read in
  // atleast n_reads inputs
  int n_reads = REDUCE_N_READS;

  // mod_in_size gives us the groups of n_reads needed to go over the entire
  // input
  uint mod_in_size = (in_size + n_reads - 1) / n_reads;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  thread_group_size =
      mod_in_size > thread_group_size ? thread_group_size : mod_in_size;

  // If the number of thread groups needed exceeds 1024, we reuse threads groups
  uint n_thread_groups =
      (mod_in_size + thread_group_size - 1) / thread_group_size;
  n_thread_groups = std::min(n_thread_groups, 1024u);
  uint nthreads = n_thread_groups * thread_group_size;

  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);

  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

void row_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const std::vector<int>& axes_,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d) {
  auto kernel = d.get_kernel("row_reduce_" + op_name + type_to_name(in));

  int n_reads = REDUCE_N_READS;
  size_t reduction_size = in.size() / out.size();

  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);

  // Each thread group is responsible for 1 output
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
  MTL::Size grid_dims = MTL::Size(n_threads, 1, 1);
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);

  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

void col_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const std::vector<int>& axes_,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d) {
  std::ostringstream kernel_name;

  bool encode_in_shape = false;
  bool encode_ndim = false;

  // If the slowest moving axis can be merged into the reductions,
  // we call the column reduce kernel
  // In this case, a linear index in the output corresponds to the
  // linear index in the input where the reduction starts
  if (axes_[axes_.size() - 1] == (axes_.size() - 1)) {
    kernel_name << "col_reduce_" << op_name << type_to_name(in);
  }
  // Otherwise, while all the reduction axes can be merged, the mapping between
  // indices in the output and input require resolving using shapes and strides
  else {
    kernel_name << "contiguous_strided_reduce_" << op_name << type_to_name(in);
    encode_in_shape = true;

    // We check for a viable template with the required number of dimensions
    // we only care about encoding non-reduced shapes and strides in the input
    size_t non_reducing_dims = in.ndim() - axes_.size();
    if (non_reducing_dims >= 1 &&
        non_reducing_dims <= MAX_REDUCE_SPECIALIZED_DIMS) {
      kernel_name << "_dim_" << non_reducing_dims;
    } else {
      encode_ndim = true;
    }
  }

  auto kernel = d.get_kernel(kernel_name.str());
  size_t in_size = in.size();
  size_t out_size = out.size();

  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);

  // Calculate the number of inputs to reduce and the stride b/w them
  size_t reduction_size = 1;
  size_t in_ndim = in.ndim();
  size_t reduction_stride = in_size;

  for (int i : axes_) {
    reduction_size *= in.shape(i);
    reduction_stride = std::min(reduction_stride, in.strides()[i]);
  }

  compute_encoder->setBytes(&reduction_size, sizeof(size_t), 2);
  compute_encoder->setBytes(&reduction_stride, sizeof(size_t), 3);
  compute_encoder->setBytes(&out_size, sizeof(size_t), 4);
  if (encode_in_shape) {
    // Obtain the non-reducing shape and strides of the input to encode
    std::vector<int> inp_shape_mod;
    std::vector<size_t> inp_strides_mod;

    for (size_t i = 0, j = 0; i < in.ndim(); i++) {
      if (j < axes_.size() && axes_[j] == i) {
        j++;
      } else {
        inp_shape_mod.push_back(in.shape(i));
        inp_strides_mod.push_back(in.strides()[i]);
      }
    }

    size_t ndim = inp_shape_mod.size();

    compute_encoder->setBytes(inp_shape_mod.data(), ndim * sizeof(int), 5);
    compute_encoder->setBytes(inp_strides_mod.data(), ndim * sizeof(size_t), 6);

    if (encode_ndim) {
      compute_encoder->setBytes(&ndim, sizeof(size_t), 7);
    }
  }

  // Select block dimensions

  // Each thread reads 16 inputs to give it more work
  uint n_inputs_per_thread = REDUCE_N_READS;
  uint n_threads_per_output =
      (reduction_size + n_inputs_per_thread - 1) / n_inputs_per_thread;

  // We spread outputs over the x dimension and inputs over the y dimension
  // Threads with the same lid.x in a given threadgroup work on the same
  // output and each thread in the y dimension accumlates for that output
  uint threadgroup_dim_x = std::min(out_size, 128ul);
  uint threadgroup_dim_y =
      kernel->maxTotalThreadsPerThreadgroup() / threadgroup_dim_x;
  threadgroup_dim_y = std::min(n_threads_per_output, threadgroup_dim_y);

  uint n_threadgroups_x =
      (out_size + threadgroup_dim_x - 1) / threadgroup_dim_x;

  uint n_threadgroups_y =
      (n_threads_per_output + threadgroup_dim_y - 1) / threadgroup_dim_y;

  // Launch enough thread groups for each output
  MTL::Size grid_dims = MTL::Size(n_threadgroups_x, n_threadgroups_y, 1);
  MTL::Size group_dims = MTL::Size(threadgroup_dim_x, threadgroup_dim_y, 1);

  // We set shared memory to be exploited here for reductions within a
  // threadgroup - each thread must be able to update its accumulated output
  // Note: Each threadgroup should have 32kB of data in threadgroup memory
  //       and threadgroup_dim_x * threadgroup_dim_y <= 1024 by design
  //       This should be fine for floats, but we might need to revisit
  //       if we ever come to doubles. In that case, we should also cut
  //       down the number of threads we launch in a threadgroup
  compute_encoder->setThreadgroupMemoryLength(
      threadgroup_dim_x * threadgroup_dim_y * out.itemsize(), 0);

  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
}

void general_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    const std::vector<int>& axes_,
    MTL::ComputeCommandEncoder* compute_encoder,
    metal::Device& d) {
  bool encode_ndim = true;
  std::ostringstream kernel_name;
  kernel_name << "general_reduce_" << op_name << type_to_name(in);

  // Check for specialzed kernels for input ndim
  if (in.ndim() >= 1 && in.ndim() <= MAX_REDUCE_SPECIALIZED_DIMS) {
    kernel_name << "_dim_" << in.ndim();
    encode_ndim = false;
  }
  auto kernel = d.get_kernel(kernel_name.str());
  size_t in_size = in.size();
  size_t ndim = in.ndim();

  // We set the reducing strides to 0 to induce collisions for the reduction
  std::vector<size_t> out_strides(ndim);
  size_t stride = 1;
  for (int i = ndim - 1, j = axes_.size() - 1; i >= 0; --i) {
    if (j >= 0 && axes_[j] == i) {
      out_strides[i] = 0;
      --j;
    } else {
      out_strides[i] = stride;
      stride *= in.shape(i);
    }
  }

  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(in.shape().data(), ndim * sizeof(int), 2);
  compute_encoder->setBytes(in.strides().data(), ndim * sizeof(size_t), 3);
  compute_encoder->setBytes(out_strides.data(), ndim * sizeof(size_t), 4);
  if (encode_ndim) {
    compute_encoder->setBytes(&ndim, sizeof(size_t), 5);
  }

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > in_size) {
    thread_group_size = in_size;
  }
  size_t nthreads = in_size;

  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace

//////////////////////////////////////////////////////////////////////
// Main reduce dispatch
//////////////////////////////////////////////////////////////////////

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto& in = inputs[0];

  // TODO: Allow specific row and column reductions with types disabled
  // due to atomics ?
  if (size_of(in.dtype()) == 8) {
    std::ostringstream msg;
    msg << "[Reduce::eval_gpu] Does not support " << in.dtype();
    throw std::runtime_error(msg.str());
  }

  // Make sure no identity reductions trickle down here
  assert(!axes_.empty());

  // Continue with reduction operation
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
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
  {
    // Check for contiguous data
    if (in.size() == in.data_size() &&
        (in.flags().row_contiguous || in.flags().col_contiguous)) {
      // Go to all reduce if reducing over all axes
      if (axes_.size() == in.ndim()) {
        all_reduce_dispatch(in, out, op_name, compute_encoder, d);
        return;
      }
      // Use specialized kernels if the input is row contiguous and
      // the reducing axes can be merged into one
      else if (
          in.flags().row_contiguous && in.strides().back() == 1 &&
          (axes_.back() - axes_.front()) == axes_.size() - 1) {
        // If the fastest moving axis is being reduced, go to row reduce
        if (axes_[0] == (in.ndim() - axes_.size())) {
          row_reduce_dispatch(in, out, op_name, axes_, compute_encoder, d);
          return;
        }
        // Otherwise go to to generalized strided reduce
        // Note: bool isn't support here yet due to the use of atomics
        //       once that is updated, this should be the else condition of this
        //       branch
        else if (in.dtype() != bool_) {
          col_reduce_dispatch(in, out, op_name, axes_, compute_encoder, d);
          return;
        }
      }
    }
    // Fall back to the general case
    general_reduce_dispatch(in, out, op_name, axes_, compute_encoder, d);
  }
}

} // namespace mlx::core
