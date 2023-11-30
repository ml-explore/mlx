// Copyright © 2023 Apple Inc.

#include <cassert>
#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);

  // Ensure contiguity
  std::vector<array> copies;
  auto in = inputs[0];
  if (!in.flags().row_contiguous) {
    array arr_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, arr_copy, CopyType::General, s);
    copies.push_back(arr_copy);
    in = arr_copy;
  }

  std::ostringstream kname;
  if (in.strides()[axis_] == 1) {
    kname << "contiguous_scan_";
    if (reverse_) {
      kname << "reverse_";
    }
    kname << ((inclusive_) ? "inclusive_" : "exclusive_");
    switch (reduce_type_) {
      case Scan::Sum:
        kname << "sum_";
        break;
      case Scan::Prod:
        kname << "prod_";
        break;
      case Scan::Max:
        kname << "max_";
        break;
      case Scan::Min:
        kname << "min_";
        break;
    }
    kname << type_to_name(in) << "_" << type_to_name(out);

    auto kernel = d.get_kernel(kname.str());
    auto compute_encoder = d.get_command_encoder(s.index);
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
    size_t size = in.shape(axis_);
    compute_encoder->setBytes(&size, sizeof(size_t), 2);

    // Compute the thread grid
    int n_reads = (in.itemsize() <= 4) ? 4 : 2;
    int elements_per_simd = n_reads * 32;
    int thread_groups = in.size() / size;
    int thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
    if (size < n_reads * 1024) {
      thread_group_size = ((size + elements_per_simd - 1) / elements_per_simd) *
          elements_per_simd;
    } else if (size < n_reads * 2048) {
      thread_group_size =
          ((size / 2 + elements_per_simd - 1) / elements_per_simd) *
          elements_per_simd;
    }
    thread_group_size = std::min(
        thread_group_size,
        static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()));
    MTL::Size grid_dims = MTL::Size(thread_groups * thread_group_size, 1, 1);
    MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  } else {
    kname << "strided_scan_";
    if (reverse_) {
      kname << "reverse_";
    }
    kname << ((inclusive_) ? "inclusive_" : "exclusive_");
    switch (reduce_type_) {
      case Scan::Sum:
        kname << "sum_";
        break;
      case Scan::Prod:
        kname << "prod_";
        break;
      case Scan::Max:
        kname << "max_";
        break;
      case Scan::Min:
        kname << "min_";
        break;
    }
    kname << type_to_name(in) << "_" << type_to_name(out);

    auto kernel = d.get_kernel(kname.str());
    auto compute_encoder = d.get_command_encoder(s.index);
    compute_encoder->setComputePipelineState(kernel);
    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, out, 1);
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
    compute_encoder->dispatchThreads(grid_dims, group_dims);
  }

  if (copies.size() > 0) {
    auto command_buffer = d.get_command_buffer(s.index);
    command_buffer->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  }
}

} // namespace mlx::core
