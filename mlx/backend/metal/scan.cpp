// Copyright © 2023 Apple Inc.

#include <cassert>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  auto& s = stream();
  auto& d = metal::device(s.device);

  std::vector<array> copies;
  auto in = inputs[0];
  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.move_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc_or_wait(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    array arr_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    in = arr_copy;
    out.move_shared_buffer(in);
  }

  bool contiguous = in.strides()[axis_] == 1;

  std::ostringstream kname;
  kname << (contiguous ? "contig_" : "strided_");
  kname << "scan_";
  if (reverse_) {
    kname << "reverse_";
  }
  kname << ((inclusive_) ? "inclusive_" : "exclusive_");

  std::string reduce_type;
  switch (reduce_type_) {
    case Scan::Sum:
      reduce_type = "sum";
      break;
    case Scan::Prod:
      reduce_type = "prod";
      break;
    case Scan::Max:
      reduce_type = "max";
      break;
    case Scan::Min:
      reduce_type = "min";
      break;
  }
  kname << reduce_type << "_" << type_to_name(in) << "_" << type_to_name(out);
  auto kernel = get_scan_kernel(
      d, kname.str(), reverse_, inclusive_, reduce_type, in, out);

  if (contiguous) {
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(
        in.data_shared_ptr() == nullptr ? out : in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis_);
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
        get_2d_grid_dims(in.shape(), in.strides(), /** divisor= */ size);
    MTL::Size grid_dims(
        thread_group_size, tmp_grid_dims.width, tmp_grid_dims.height);
    MTL::Size group_dims(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(
        in.data_shared_ptr() == nullptr ? out : in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis_);
    size_t stride = in.strides()[axis_];
    int bm = 32;
    int bn = 32;
    size_t stride_blocks = (stride + bn - 1) / bn;
    compute_encoder.set_bytes(size, 2);
    compute_encoder.set_bytes(stride, 3);
    compute_encoder.set_bytes(stride_blocks, 4);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    int n_simdgroups = bn / n_reads;
    int thread_group_size = n_simdgroups * 32;
    auto tmp_grid_dims = get_2d_grid_dims(
        in.shape(), in.strides(), /** divisor= */ size * stride);
    if (tmp_grid_dims.width * stride_blocks <= UINT_MAX) {
      tmp_grid_dims.width *= stride_blocks;
    } else {
      tmp_grid_dims.height *= stride_blocks;
    }
    MTL::Size grid_dims(
        thread_group_size, tmp_grid_dims.width, tmp_grid_dims.height);
    MTL::Size group_dims(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
