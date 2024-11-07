// Copyright © 2023-2024 Apple Inc.

#include <algorithm>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

void single_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn,
    bool argsort) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  std::vector<size_t> in_nc_str = in.strides();
  in_nc_str.erase(in_nc_str.begin() + axis);

  std::vector<size_t> out_nc_str = out.strides();
  out_nc_str.erase(out_nc_str.begin() + axis);

  std::vector<int> nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int size_sorted_axis = in.shape(axis);
  int in_stride_sorted_axis = in.strides()[axis];
  int out_stride_sorted_axis = out.strides()[axis];
  int in_stride_segment_axis =
      *std::min_element(in_nc_str.begin(), in_nc_str.end());
  int out_stride_segment_axis =
      *std::min_element(out_nc_str.begin(), out_nc_str.end());

  // We can only use the contiguous kernel if the sorted axis
  // has the largest or smallest stride.
  // We also need the input to be contiguous
  bool contiguous = in.flags().contiguous;
  auto check_strides = [](array x, int sort_stride) {
    int min_stride = *std::min_element(x.strides().begin(), x.strides().end());
    int max_stride = *std::max_element(x.strides().begin(), x.strides().end());
    return sort_stride == min_stride || sort_stride == max_stride;
  };
  contiguous &= check_strides(in, in_stride_sorted_axis);
  contiguous &= check_strides(out, out_stride_sorted_axis);

  // Prepare kernel name
  std::ostringstream kname;
  kname << (contiguous ? "c" : "nc");
  if (argsort) {
    kname << "arg";
  }

  kname << "_block_sort_" << type_to_name(in) << "_" << type_to_name(out)
        << "_bn" << bn << "_tn" << tn;
  auto kernel = get_sort_kernel(d, kname.str(), in, out, bn, tn);

  // Prepare command encoder
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set inputs
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(size_sorted_axis, 2);
  compute_encoder.set_bytes(in_stride_sorted_axis, 3);
  compute_encoder.set_bytes(out_stride_sorted_axis, 4);

  if (contiguous) {
    compute_encoder.set_bytes(in_stride_segment_axis, 5);
    compute_encoder.set_bytes(out_stride_segment_axis, 6);
  } else {
    compute_encoder.set_bytes(nc_dim, 5);
    compute_encoder.set_vector_bytes(nc_shape, 6);
    compute_encoder.set_vector_bytes(in_nc_str, 7);
    compute_encoder.set_vector_bytes(out_nc_str, 8);
  }

  MTL::Size group_dims = MTL::Size(bn, 1, 1);
  MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

  compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
}

void multi_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn,
    int n_blocks,
    bool argsort) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  std::vector<size_t> nc_str = in.strides();
  nc_str.erase(nc_str.begin() + axis);

  std::vector<int> nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  if (nc_dim == 0) {
    nc_shape = {0};
    nc_str = {1};
  }

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
  auto& compute_encoder = d.get_command_encoder(s.index);

  // Do blockwise sort
  {
    std::ostringstream kname;
    kname << "sort_mbsort_" << type_to_name(dev_vals_0) << "_"
          << type_to_name(dev_idxs_0) << "_bn" << bn << "_tn" << tn;
    auto kernel =
        get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(dev_vals_0, 1);
    compute_encoder.set_output_array(dev_idxs_0, 2);
    compute_encoder.set_bytes(size_sorted_axis, 3);
    compute_encoder.set_bytes(stride_sorted_axis, 4);
    compute_encoder.set_bytes(nc_dim, 5);
    compute_encoder.set_vector_bytes(nc_shape, 6);
    compute_encoder.set_vector_bytes(nc_str, 7);

    MTL::Size group_dims = MTL::Size(bn, 1, 1);
    MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

    compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
  }

  // Do merges
  bool ping = false;
  array dev_vals_in = dev_vals_0;
  array dev_idxs_in = dev_idxs_0;
  array dev_vals_out = dev_vals_1;
  array dev_idxs_out = dev_idxs_1;

  int n_thr_per_group = (n_blocks + 1) < 1024 ? (n_blocks + 1) : 1024;

  for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2) {
    dev_vals_in = ping ? dev_vals_1 : dev_vals_0;
    dev_idxs_in = ping ? dev_idxs_1 : dev_idxs_0;
    dev_vals_out = ping ? dev_vals_0 : dev_vals_1;
    dev_idxs_out = ping ? dev_idxs_0 : dev_idxs_1;
    ping = !ping;

    // Do partition
    {
      std::ostringstream kname;
      kname << "partition_mbsort_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel =
          get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_output_array(block_partitions, 0);
      compute_encoder.set_input_array(dev_vals_in, 1);
      compute_encoder.set_input_array(dev_idxs_in, 2);
      compute_encoder.set_bytes(size_sorted_axis, 3);
      compute_encoder.set_bytes(merge_tiles, 4);
      compute_encoder.set_bytes(n_blocks, 5);

      MTL::Size group_dims = MTL::Size(n_thr_per_group, 1, 1);
      MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }

    // Do merge
    {
      std::ostringstream kname;
      kname << "merge_mbsort_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel =
          get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_input_array(block_partitions, 0);
      compute_encoder.set_input_array(dev_vals_in, 1);
      compute_encoder.set_input_array(dev_idxs_in, 2);
      compute_encoder.set_output_array(dev_vals_out, 3);
      compute_encoder.set_output_array(dev_idxs_out, 4);
      compute_encoder.set_bytes(size_sorted_axis, 5);
      compute_encoder.set_bytes(merge_tiles, 6);
      compute_encoder.set_bytes(n_blocks, 7);

      MTL::Size group_dims = MTL::Size(bn, 1, 1);
      MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

      compute_encoder.dispatchThreadgroups(grid_dims, group_dims);
    }
  }

  // Copy outputs with appropriate strides
  auto strides = out.strides();
  for (int ax = axis + 1; ax < strides.size(); ax++) {
    strides[ax] *= out.shape(axis);
  }
  strides[axis] = 1;
  copy_gpu_inplace(
      (argsort) ? dev_idxs_out : dev_vals_out,
      out,
      out.shape(),
      strides,
      out.strides(),
      0,
      0,
      (axis == in.ndim() - 1) ? CopyType::Vector : CopyType::General,
      s);

  d.add_temporaries(std::move(copies), s.index);
}

void gpu_merge_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis_,
    bool argsort) {
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
    return multi_block_sort(s, d, in, out, axis, bn, tn, n_blocks, argsort);
  } else {
    return single_block_sort(s, d, in, out, axis, bn, tn, argsort);
  }
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // We direct arg partition to sort for now
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // We direct partition to sort for now
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, false);
}

} // namespace mlx::core
