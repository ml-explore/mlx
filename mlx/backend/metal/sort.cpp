// Copyright © 2023 Apple Inc.

#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <bool ARGSORT>
void single_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  std::vector<size_t> nc_str = in.strides();
  nc_str.erase(nc_str.begin() + axis);

  std::vector<int> nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int size_sorted_axis = in.shape(axis);
  int stride_sorted_axis = in.strides()[axis];
  int stride_segment_axis = *std::min_element(nc_str.begin(), nc_str.end());

  // Check if remaining strides are contiguous
  bool contiguous_write = true;
  if (axis != in.ndim() - 1 && axis != 0) {
    for (int i = 0; i < nc_str.size() - 1; ++i) {
      size_t expected = nc_str[i + 1] * nc_str[i + 1];
      contiguous_write &= (nc_str[i] == expected);
    }
  }

  // Prepare kernel name
  std::ostringstream kname;
  if (ARGSORT) {
    kname << "arg_";
  }
  kname << "block_merge_sort_" << type_to_name(in) << "_" << type_to_name(out)
        << "_bn" << bn << "_tn" << tn;

  if (!contiguous_write) {
    kname << "_nc";
  }

  // Prepare command encoder
  auto compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname.str());
  compute_encoder->setComputePipelineState(kernel);

  // Set inputs
  set_array_buffer(compute_encoder, in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(&size_sorted_axis, sizeof(int), 2);
  compute_encoder->setBytes(&stride_sorted_axis, sizeof(int), 3);

  if (contiguous_write) {
    compute_encoder->setBytes(&stride_segment_axis, sizeof(int), 4);
  } else {
    compute_encoder->setBytes(&nc_dim, sizeof(int), 4);
    compute_encoder->setBytes(nc_shape.data(), nc_dim * sizeof(int), 5);
    compute_encoder->setBytes(nc_str.data(), nc_dim * sizeof(size_t), 6);
  }

  MTL::Size group_dims = MTL::Size(bn, 1, 1);
  MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

  compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
}

template <bool ARGSORT>
void multi_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn,
    int n_blocks) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  std::vector<size_t> nc_str = in.strides();
  nc_str.erase(nc_str.begin() + axis);

  std::vector<int> nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int size_sorted_axis = in.shape(axis);
  int stride_sorted_axis = in.strides()[axis];

  // Make temporary copies
  array dev_vals_0({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});
  array dev_vals_1({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});

  array dev_idxs_0({n_rows, size_sorted_axis}, uint32, nullptr, {});
  array dev_idxs_1({n_rows, size_sorted_axis}, uint32, nullptr, {});

  array block_partitions({n_rows, n_blocks + 1}, uint32, nullptr, {});

  // Do allocations
  dev_vals_0.set_data(allocator::malloc_or_wait(dev_vals_0.nbytes()));
  dev_vals_1.set_data(allocator::malloc_or_wait(dev_vals_1.nbytes()));
  dev_idxs_0.set_data(allocator::malloc_or_wait(dev_idxs_0.nbytes()));
  dev_idxs_1.set_data(allocator::malloc_or_wait(dev_idxs_1.nbytes()));
  block_partitions.set_data(
      allocator::malloc_or_wait(block_partitions.nbytes()));

  std::vector<array> copies = {
      dev_vals_0, dev_vals_1, dev_idxs_0, dev_idxs_1, block_partitions};

  // Prepare command encoder
  auto compute_encoder = d.get_command_encoder(s.index);

  // Do blockwise sort
  {
    std::ostringstream kname;
    kname << "mb_block_sort_" << type_to_name(dev_vals_0) << "_"
          << type_to_name(dev_idxs_0) << "_bn" << bn << "_tn" << tn;

    auto kernel = d.get_kernel(kname.str());
    compute_encoder->setComputePipelineState(kernel);

    set_array_buffer(compute_encoder, in, 0);
    set_array_buffer(compute_encoder, dev_vals_0, 1);
    set_array_buffer(compute_encoder, dev_idxs_0, 2);
    compute_encoder->setBytes(&size_sorted_axis, sizeof(int), 3);
    compute_encoder->setBytes(&stride_sorted_axis, sizeof(int), 4);
    compute_encoder->setBytes(&nc_dim, sizeof(int), 5);
    compute_encoder->setBytes(nc_shape.data(), nc_dim * sizeof(int), 6);
    compute_encoder->setBytes(nc_str.data(), nc_dim * sizeof(size_t), 7);

    MTL::Size group_dims = MTL::Size(bn, 1, 1);
    MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

    compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
  }

  // Do merges
  bool ping = false;
  array dev_vals_in = dev_vals_0;
  array dev_idxs_in = dev_idxs_0;
  array dev_vals_out = dev_vals_1;
  array dev_idxs_out = dev_idxs_1;
  for (int merge_tiles = 2; merge_tiles <= n_blocks; merge_tiles *= 2) {
    dev_vals_in = ping ? dev_vals_1 : dev_vals_0;
    dev_idxs_in = ping ? dev_idxs_1 : dev_idxs_0;
    dev_vals_out = ping ? dev_vals_0 : dev_vals_1;
    dev_idxs_out = ping ? dev_idxs_0 : dev_idxs_1;
    ping = !ping;

    // Do partition
    {
      std::ostringstream kname;
      kname << "mb_block_partition_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel = d.get_kernel(kname.str());
      compute_encoder->setComputePipelineState(kernel);

      set_array_buffer(compute_encoder, block_partitions, 0);
      set_array_buffer(compute_encoder, dev_vals_in, 1);
      set_array_buffer(compute_encoder, dev_idxs_in, 2);
      compute_encoder->setBytes(&size_sorted_axis, sizeof(int), 3);
      compute_encoder->setBytes(&merge_tiles, sizeof(int), 4);

      MTL::Size group_dims = MTL::Size(n_blocks + 1, 1, 1);
      MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

      compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
    }

    // Do merge
    {
      std::ostringstream kname;
      kname << "mb_block_merge_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel = d.get_kernel(kname.str());
      compute_encoder->setComputePipelineState(kernel);

      set_array_buffer(compute_encoder, block_partitions, 0);
      set_array_buffer(compute_encoder, dev_vals_in, 1);
      set_array_buffer(compute_encoder, dev_idxs_in, 2);
      set_array_buffer(compute_encoder, dev_vals_out, 3);
      set_array_buffer(compute_encoder, dev_idxs_out, 4);
      compute_encoder->setBytes(&size_sorted_axis, sizeof(int), 5);
      compute_encoder->setBytes(&merge_tiles, sizeof(int), 6);
      compute_encoder->setBytes(&n_blocks, sizeof(int), 7);

      MTL::Size group_dims = MTL::Size(bn, 1, 1);
      MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

      compute_encoder->dispatchThreadgroups(grid_dims, group_dims);
    }
  }

  // Copy outputs with appropriate strides
  array strided_out_arr = ARGSORT ? dev_idxs_out : dev_vals_out;

  if (axis == strided_out_arr.ndim() - 1) {
    copy_gpu_inplace(strided_out_arr, out, CopyType::Vector, s);
  } else {
    std::vector<int> strided_out_shape = strided_out_arr.shape();
    std::vector<size_t> strided_out_str = strided_out_arr.strides();

    int out_axis_shape = strided_out_shape[axis];
    int out_axis_str = strided_out_str[axis];

    strided_out_shape.erase(strided_out_shape.begin() + axis);
    strided_out_str.erase(strided_out_str.begin() + axis);

    strided_out_shape.push_back(out_axis_shape);
    strided_out_str.push_back(out_axis_str);

    array strided_out_slice(strided_out_shape, out.dtype(), nullptr, {});
    strided_out_slice.copy_shared_buffer(
        strided_out_arr,
        strided_out_str,
        strided_out_arr.flags(),
        strided_out_arr.size(),
        0);

    copy_gpu_inplace(strided_out_slice, out, CopyType::General, s);
  }

  // Clear copies
  d.get_command_buffer(s.index)->addCompletedHandler(
      [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
}

template <bool ARGSORT>
void gpu_merge_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis_) {
  // Get size info
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);

  // Get kernel size
  int tn = 8;
  int bn = 128;
  int potential_bn = (size_sorted_axis + tn - 1) / tn;

  if (potential_bn > 256) {
    bn = 512;
  } else if (potential_bn > 128) {
    bn = 256;
  } else {
    bn = 128;
  }

  if (bn == 512 && size_of(in.dtype()) > 4) {
    bn = 256;
  }

  int n_per_block = bn * tn;
  int n_blocks = (size_sorted_axis + n_per_block - 1) / n_per_block;

  if (n_blocks > 1) {
    return multi_block_sort<ARGSORT>(s, d, in, out, axis, bn, tn, n_blocks);
  } else {
    return single_block_sort<ARGSORT>(s, d, in, out, axis, bn, tn);
  }
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort<true>(s, d, in, out, axis_);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort<false>(s, d, in, out, axis_);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // We direct arg partition to sort for now
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort<true>(s, d, in, out, axis_);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // We direct partition to sort for now
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort<false>(s, d, in, out, axis_);
}

} // namespace mlx::core
