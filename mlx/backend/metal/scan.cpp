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
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(
        in.data_shared_ptr() == nullptr ? out : in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis_);
    compute_encoder->setBytes(&size, sizeof(size_t), 2);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    constexpr int simd_size = 32;
    int elements_per_simd = n_reads * simd_size;
    int thread_groups = in.data_size() / size;
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
    MTL::Size grid_dims = MTL::Size(thread_groups * thread_group_size, 1, 1);
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder.set_input_array(
        in.data_shared_ptr() == nullptr ? out : in, 0);
    compute_encoder.set_output_array(out, 1);
    size_t size = in.shape(axis_);
    size_t stride = in.strides()[axis_];
    compute_encoder->setBytes(&size, sizeof(size_t), 2);
    compute_encoder->setBytes(&stride, sizeof(size_t), 3);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    int tile_x = 32;
    int tile_y = 32;
    int elements_per_tile_x = tile_x * n_reads;
    int grid_y = in.size() / size / stride;
    int grid_x = (stride + elements_per_tile_x - 1) / elements_per_tile_x;
    MTL::Size grid_dims = MTL::Size(grid_x * tile_x, grid_y * tile_y, 1);
    MTL::Size group_dims = MTL::Size(tile_x, tile_y, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core
