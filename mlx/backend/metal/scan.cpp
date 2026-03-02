// Copyright © 2023 Apple Inc.

#include <cassert>
#include <sstream>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/scan.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void scan_gpu_inplace(
    array in,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s) {
  auto& d = metal::device(s.device);

  bool contiguous = in.strides()[axis] == 1;

  std::string reduce_type_str;
  switch (reduce_type) {
    case Scan::Sum:
      reduce_type_str = "sum";
      break;
    case Scan::Prod:
      reduce_type_str = "prod";
      break;
    case Scan::Max:
      reduce_type_str = "max";
      break;
    case Scan::Min:
      reduce_type_str = "min";
      break;
    case Scan::LogAddExp:
      reduce_type_str = "logaddexp";
      break;
  }

  std::string kname;
  concatenate(
      kname,
      contiguous ? "contig_" : "strided_",
      "scan_",
      reverse ? "reverse_" : "",
      inclusive ? "inclusive_" : "exclusive_",
      reduce_type_str,
      "_",
      type_to_name(in),
      "_",
      type_to_name(out));

  auto kernel =
      get_scan_kernel(d, kname, reverse, inclusive, reduce_type_str, in, out);

  if (contiguous) {
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis);
    compute_encoder.set_bytes(size, 2);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    constexpr int simd_size = 32;
    int elements_per_simd = n_reads * simd_size;
    int thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (size <= n_reads * 1024) {
      thread_group_size =
          ((size + elements_per_simd - 1) / elements_per_simd) * simd_size;
    } else if (size <= n_reads * 2048) {
      thread_group_size =
          ((size / 2 + elements_per_simd - 1) / elements_per_simd) * simd_size;
    }
    thread_group_size = std::min(
        thread_group_size,
        static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()));
    auto tmp_grid_dims =
        get_2d_grid_dims(in.shape(), in.strides(), /*divisor=*/size);
    MTL::Size grid_dims(
        thread_group_size, tmp_grid_dims.width, tmp_grid_dims.height);
    MTL::Size group_dims(thread_group_size, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  } else {
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis);
    size_t stride = in.strides()[axis];
    int bn = 32;
    size_t stride_blocks = (stride + bn - 1) / bn;
    compute_encoder.set_bytes(size, 2);
    compute_encoder.set_bytes(stride, 3);
    compute_encoder.set_bytes(stride_blocks, 4);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    int n_simdgroups = bn / n_reads;
    int thread_group_size = n_simdgroups * 32;
    auto tmp_grid_dims =
        get_2d_grid_dims(in.shape(), in.strides(), /*divisor=*/size * stride);
    if (tmp_grid_dims.width * stride_blocks <= UINT_MAX) {
      tmp_grid_dims.width *= stride_blocks;
    } else {
      tmp_grid_dims.height *= stride_blocks;
    }
    MTL::Size grid_dims(
        thread_group_size, tmp_grid_dims.width, tmp_grid_dims.height);
    MTL::Size group_dims(thread_group_size, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto in = inputs[0];
  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    in = contiguous_copy_gpu(in, stream());
    out.copy_shared_buffer(in);
  }

  scan_gpu_inplace(
      in, out, reduce_type_, axis_, reverse_, inclusive_, stream());
}

} // namespace mlx::core
