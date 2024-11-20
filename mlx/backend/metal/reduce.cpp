// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <cassert>

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
  size_t row_size;

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

    compute_encoder.set_bytes(row_size, 2);
    compute_encoder.set_bytes(non_row_reductions, 3);
    compute_encoder.set_vector_bytes(shape, 4);
    compute_encoder.set_vector_bytes(strides, 5);
    compute_encoder.set_bytes(ndim, 6);
    compute_encoder.set_vector_bytes(reduce_shape, 7);
    compute_encoder.set_vector_bytes(reduce_strides, 8);
    compute_encoder.set_bytes(reduce_ndim, 9);

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

    // We 'll use a stride_back variable because strides.back() could be 0 but
    // yet we may have removed the appropriate amount of elements. It is safe
    // to compute the stride by multiplying shapes (while < reduction_stride)
    // because it is a contiguous section.
    size_t stride_back = 1;
    std::tie(shape, strides) = shapes_without_reduction_axes(in, axes);
    while (!shape.empty() && stride_back < reduction_stride) {
      stride_back *= shape.back();
      shape.pop_back();
      strides.pop_back();
    }
    std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
    ndim = shape.size();
  }

  /**
   * Create the col reduce arguments for reducing the 1st axis of the row
   * contiguous intermediate array.
   */
  ColReduceArgs(const array& intermediate) {
    assert(intermediate.flags().row_contiguous);

    reduction_size = intermediate.shape(0);
    reduction_stride = intermediate.size() / reduction_size;
    non_col_reductions = 1;
    reduce_ndim = 0;
    ndim = 0;
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

    compute_encoder.set_bytes(reduction_size, 2);
    compute_encoder.set_bytes(reduction_stride, 3);
    compute_encoder.set_vector_bytes(shape, 4);
    compute_encoder.set_vector_bytes(strides, 5);
    compute_encoder.set_bytes(ndim, 6);
    compute_encoder.set_vector_bytes(reduce_shape, 7);
    compute_encoder.set_vector_bytes(reduce_strides, 8);
    compute_encoder.set_bytes(reduce_ndim, 9);
    compute_encoder.set_bytes(non_col_reductions, 10);

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

inline int get_kernel_reduce_ndim(int reduce_ndim) {
  return (reduce_ndim <= 1) ? 1 : 5;
}

inline int threadgroup_size_from_row_size(int row_size) {
  // 1 simdgroup per row smallish rows
  if (row_size <= 512) {
    return 32;
  }

  // 2 simdgroups per row for medium rows
  if (row_size <= 1024) {
    return 128;
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

std::pair<Dtype, Dtype> remap_reduce_types(
    const array& in,
    const std::string& op_name) {
  if (op_name == "sum" || op_name == "prod") {
    if (issubdtype(in.dtype(), integer)) {
      switch (in.dtype().size()) {
        case 1:
          return {int8, int8};
        case 2:
          return {int16, int16};
        case 4:
          return {int32, int32};
        case 8:
          return {int64, int64};
      }
    }
    return {in.dtype(), in.dtype()};
  } else if (op_name == "and" || op_name == "or") {
    if (in.dtype().size() == 1) {
      return {bool_, bool_};
    } else if (in.dtype().size() == 2) {
      return {int16, bool_};
    } else if (in.dtype().size() == 4) {
      return {int32, bool_};
    } else {
      return {int64, bool_};
    }
  }
  return {in.dtype(), in.dtype()};
}

void init_reduce(
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [_, out_type] = remap_reduce_types(out, op_name);
  const std::string func_name = "init_reduce";
  std::string kname = func_name;
  concatenate(kname, "_", op_name, type_to_name(out_type));
  auto kernel = get_reduce_init_kernel(d, kname, func_name, op_name, out);
  size_t nthreads = out.size();
  MTL::Size grid_dims = MTL::Size(nthreads, 1, 1);
  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_output_array(out, 0);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void all_reduce_dispatch(
    const array& in,
    array& out,
    const std::string& op_name,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  auto [in_type, out_type] = remap_reduce_types(in, op_name);
  const std::string func_name = "all_reduce";
  std::string kname = func_name;
  concatenate(kname, "_", op_name, type_to_name(in_type));
  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out);
  compute_encoder.set_compute_pipeline_state(kernel);

  size_t in_size = in.size();

  // Small array so dispatch a single threadgroup
  if (in_size <= REDUCE_N_READS * 1024) {
    int threadgroup_size = (in_size + REDUCE_N_READS - 1) / REDUCE_N_READS;
    threadgroup_size = ((threadgroup_size + 31) / 32) * 32;
    MTL::Size grid_dims(threadgroup_size, 1, 1);

    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(in_size, 2);
    compute_encoder.set_bytes(in_size, 3);
    compute_encoder.dispatch_threads(grid_dims, grid_dims);
  }

  // We need multiple threadgroups so we 'll do it in 2 passes.
  else {
    int n_rows, threadgroup_2nd_pass;
    // Less than 2**26 bytes
    if (in.nbytes() <= (1 << 26)) {
      n_rows = 32 * REDUCE_N_READS;
      threadgroup_2nd_pass = 32;
    }

    // Really large matrix so parallelize as much as possible
    else {
      n_rows = 1024 * REDUCE_N_READS;
      threadgroup_2nd_pass = 1024;
    }

    // Allocate an intermediate tensor to hold results if needed
    array intermediate({n_rows}, out_type, nullptr, {});
    intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
    d.add_temporary(intermediate, s.index);

    // 1st pass
    size_t row_size = (in_size + n_rows - 1) / n_rows;
    int threadgroup_size =
        std::min((row_size + REDUCE_N_READS - 1) / REDUCE_N_READS, 1024ul);
    threadgroup_size = ((threadgroup_size + 31) / 32) * 32;
    MTL::Size grid_dims(threadgroup_size, n_rows, 1);
    MTL::Size group_dims(threadgroup_size, 1, 1);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(intermediate, 1);
    compute_encoder.set_bytes(in_size, 2);
    compute_encoder.set_bytes(row_size, 3);
    compute_encoder.dispatch_threads(grid_dims, group_dims);

    // 2nd pass
    std::string kname_2nd_pass = func_name;
    concatenate(kname_2nd_pass, "_", op_name, type_to_name(intermediate));
    auto kernel_2nd_pass = get_reduce_kernel(
        d, kname_2nd_pass, func_name, op_name, intermediate, out);
    compute_encoder.set_compute_pipeline_state(kernel_2nd_pass);
    size_t intermediate_size = n_rows;
    grid_dims = MTL::Size(threadgroup_2nd_pass, 1, 1);
    group_dims = MTL::Size(threadgroup_2nd_pass, 1, 1);
    compute_encoder.set_input_array(intermediate, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(intermediate_size, 2);
    compute_encoder.set_bytes(intermediate_size, 3);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void row_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  auto [in_type, out_type] = remap_reduce_types(in, op_name);
  const std::string func_name = "row_reduce_small";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out, n);
  compute_encoder.set_compute_pipeline_state(kernel);

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
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void row_reduce_simple(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  // Set the kernel
  auto [in_type, out_type] = remap_reduce_types(in, op_name);
  const std::string func_name = "row_reduce_simple";
  std::string kname = func_name;
  concatenate(kname, "_", op_name, type_to_name(in_type));

  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Figure out the grid dims
  size_t row_size = args.row_size;
  size_t out_size = out.size();
  auto out_grid_size = get_2d_grid_dims(out.shape(), out.strides());
  out_grid_size.width =
      (out_grid_size.width + REDUCE_N_WRITES - 1) / REDUCE_N_WRITES;
  int threadgroup_size = threadgroup_size_from_row_size(row_size);
  if (in.itemsize() == 8) {
    threadgroup_size = std::min(threadgroup_size, 512);
  }
  MTL::Size grid_dims(
      threadgroup_size, out_grid_size.width, out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(row_size, 2);
  compute_encoder.set_bytes(out_size, 3);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void row_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    RowReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [in_type, out_type] = remap_reduce_types(in, op_name);

  // Set the kernel
  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  const std::string func_name = "row_reduce_looped";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out, n);
  compute_encoder.set_compute_pipeline_state(kernel);

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
  compute_encoder.dispatch_threads(grid_dims, group_dims);
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
  // Prepare the arguments for the kernel
  RowReduceArgs args(in, plan, axes);

  // Case 1: The row is small
  if (args.row_size <= 64) {
    return row_reduce_small(in, out, op_name, args, compute_encoder, d, s);
  }

  // Case 2: Contiguous reduce without non-row reductions
  if (plan.type == ContiguousReduce && args.reduce_ndim == 0 &&
      in.size() / args.row_size >= 32) {
    return row_reduce_simple(in, out, op_name, args, compute_encoder, d, s);
  }

  // Case 3: General row reduce including non-row reductions
  return row_reduce_looped(in, out, op_name, args, compute_encoder, d, s);
}

void strided_reduce_small(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [in_type, out_type] = remap_reduce_types(in, op_name);

  // Figure out the grid dims
  MTL::Size grid_dims, group_dims;

  // Prepare the arguments for the kernel
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  const std::string func_name = "col_reduce_small";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out, n);
  compute_encoder.set_compute_pipeline_state(kernel);

  const int n_reads = 4;
  size_t reduction_stride_blocks =
      (args.reduction_stride + n_reads - 1) / n_reads;
  size_t total = args.reduction_size * args.non_col_reductions;
  size_t threadgroup_x = std::min(reduction_stride_blocks, 32ul);
  size_t threadgroup_y = std::min(
      8ul,
      std::min(kernel->maxTotalThreadsPerThreadgroup() / threadgroup_x, total));

  group_dims = MTL::Size(threadgroup_x, threadgroup_y, 1);
  grid_dims = output_grid_for_col_reduce(out, args);
  grid_dims = MTL::Size(
      (reduction_stride_blocks + threadgroup_x - 1) / threadgroup_x,
      grid_dims.width,
      grid_dims.height);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void strided_reduce_longcolumn(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [in_type, out_type] = remap_reduce_types(in, op_name);
  size_t total_reduction_size = args.reduction_size * args.non_col_reductions;
  size_t outer_blocks = 32;
  if (total_reduction_size >= 32768) {
    outer_blocks = 128;
  }

  // Prepare the temporary accumulator
  std::vector<int> intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.push_back(outer_blocks);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end());
  array intermediate(std::move(intermediate_shape), out_type, nullptr, {});
  intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
  d.add_temporary(intermediate, s.index);

  // Prepare the arguments for the kernel
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  // Figure out the grid dims
  size_t out_size = out.size();
  size_t threadgroup_x = args.reduction_stride;
  size_t threadgroup_y =
      (args.non_col_reductions * args.reduction_size + outer_blocks - 1) /
      outer_blocks;
  threadgroup_y = std::min(32ul, threadgroup_y);

  auto out_grid_size = output_grid_for_col_reduce(out, args);
  MTL::Size grid_dims(out_grid_size.width, out_grid_size.height, outer_blocks);
  MTL::Size group_dims(threadgroup_x, threadgroup_y, 1);

  // Set the kernel
  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  std::string func_name = "col_reduce_longcolumn";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel = get_reduce_kernel(d, kname, func_name, op_name, in, out, n);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(intermediate, 1);
  args.encode(compute_encoder);
  compute_encoder.set_bytes(out_size, 11);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Make the 2nd pass arguments and grid_dims
  ColReduceArgs second_args(intermediate);
  second_args.reduce_shape.push_back(outer_blocks);
  second_args.reduce_strides.push_back(out.size());
  second_args.reduce_ndim++;
  int BN = 32;
  grid_dims = MTL::Size(256 * ((out.size() + BN - 1) / BN), 1, 1);
  group_dims = MTL::Size(256, 1, 1);

  // Set the 2nd kernel
  func_name = "col_reduce_looped";
  kname = func_name;
  if (intermediate.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(kname, "_1_32_32_reduce_", op_name, type_to_name(intermediate));
  kernel = get_reduce_kernel(
      d, kname, func_name, op_name, intermediate, out, 1, 32, 32);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_output_array(out, 1);
  second_args.encode(compute_encoder);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void strided_reduce_looped(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [in_type, out_type] = remap_reduce_types(in, op_name);

  // Prepare the arguments for the kernel
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  // Figure out the grid dims
  auto out_grid_size = output_grid_for_col_reduce(out, args);
  int BN = 32;
  int BM = 1024 / BN;
  int threadgroup_size = 8 * 32;
  MTL::Size grid_dims(
      threadgroup_size * ((args.reduction_stride + BN - 1) / BN),
      out_grid_size.width,
      out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Set the kernel
  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  std::string func_name = "col_reduce_looped";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_",
      std::to_string(BM),
      "_",
      std::to_string(BN),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel =
      get_reduce_kernel(d, kname, func_name, op_name, in, out, n, BM, BN);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  args.encode(compute_encoder);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void strided_reduce_2pass(
    const array& in,
    array& out,
    const std::string& op_name,
    ColReduceArgs& args,
    CommandEncoder& compute_encoder,
    metal::Device& d,
    const Stream& s) {
  auto [in_type, out_type] = remap_reduce_types(in, op_name);

  // Prepare the temporary accumulator
  std::vector<int> intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.push_back(32);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end());
  array intermediate(std::move(intermediate_shape), out_type, nullptr, {});
  intermediate.set_data(allocator::malloc_or_wait(intermediate.nbytes()));
  d.add_temporary(intermediate, s.index);

  // Prepare the arguments for the kernel
  args.reduce_shape.push_back(args.reduction_size);
  args.reduce_strides.push_back(args.reduction_stride);
  args.reduce_ndim++;

  // Figure out the grid dims
  size_t out_size = out.size() / args.reduction_stride;
  auto out_grid_size = output_grid_for_col_reduce(out, args);
  int outer_blocks = 32;
  int BN = 32;
  int BM = 1024 / BN;
  int threadgroup_size = 8 * 32;
  MTL::Size grid_dims(
      threadgroup_size * ((args.reduction_stride + BN - 1) / BN),
      out_grid_size.width * outer_blocks,
      out_grid_size.height);
  MTL::Size group_dims(threadgroup_size, 1, 1);

  // Set the kernel
  int n = get_kernel_reduce_ndim(args.reduce_ndim);
  std::string func_name = "col_reduce_2pass";
  std::string kname = func_name;
  if (in.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(
      kname,
      "_",
      std::to_string(n),
      "_",
      std::to_string(BM),
      "_",
      std::to_string(BN),
      "_reduce_",
      op_name,
      type_to_name(in_type));
  auto kernel =
      get_reduce_kernel(d, kname, func_name, op_name, in, out, n, BM, BN);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Launch
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(intermediate, 1);
  args.encode(compute_encoder);
  compute_encoder.set_bytes(out_size, 11);
  compute_encoder.dispatch_threads(grid_dims, group_dims);

  // Make the 2nd pass arguments and grid_dims
  ColReduceArgs second_args(intermediate);
  second_args.reduce_shape.push_back(outer_blocks);
  second_args.reduce_strides.push_back(out.size());
  second_args.reduce_ndim++;
  grid_dims = MTL::Size(threadgroup_size * ((out.size() + BN - 1) / BN), 1, 1);

  // Set the 2nd kernel
  func_name = "col_reduce_looped";
  kname = func_name;
  if (intermediate.size() > UINT32_MAX) {
    kname += "_large";
  }
  concatenate(kname, "_1_32_32_reduce_", op_name, type_to_name(intermediate));
  kernel = get_reduce_kernel(
      d, kname, func_name, op_name, intermediate, out, 1, 32, 32);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_output_array(out, 1);
  second_args.encode(compute_encoder);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
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
  // Prepare the arguments for the kernel
  ColReduceArgs args(in, plan, axes);

  // Small column
  if (args.reduction_size * args.non_col_reductions < 32) {
    return strided_reduce_small(in, out, op_name, args, compute_encoder, d, s);
  }

  // Long column but small row
  if (args.reduction_stride < 32 &&
      args.reduction_size * args.non_col_reductions >= 1024) {
    return strided_reduce_longcolumn(
        in, out, op_name, args, compute_encoder, d, s);
  }

  if (args.reduction_size * args.non_col_reductions > 256 &&
      out.size() / 32 < 1024) {
    return strided_reduce_2pass(in, out, op_name, args, compute_encoder, d, s);
  }

  return strided_reduce_looped(in, out, op_name, args, compute_encoder, d, s);
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
      d.add_temporary(in_copy, s.index);
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
  }

  // Nothing to reduce just initialize the output
  else {
    init_reduce(out, op_name, compute_encoder, d, s);
  }
}

} // namespace mlx::core
