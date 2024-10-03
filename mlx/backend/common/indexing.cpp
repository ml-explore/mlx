// Copyright © 2023 Apple Inc.
#include <algorithm>
#include <cassert>
#include <cmath>

#include "mlx/allocator.h"
#include "mlx/primitives.h"

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

template <typename IdxT>
inline size_t offset_neg_idx(IdxT idx, size_t size) {
  return (idx < 0) ? idx + size : idx;
}

template <>
inline size_t offset_neg_idx(bool idx, size_t) {
  return idx;
}

template <>
inline size_t offset_neg_idx(uint32_t idx, size_t) {
  return idx;
}

template <typename T, typename IdxT>
void gather(
    const array& src,
    const std::vector<array>& inds,
    array& out,
    const std::vector<int>& axes,
    const std::vector<int>& slice_sizes) {
  // If the array is row contiguous then we can do a contiguous copy given
  // two conditions on the slice size:
  // - Any number of leading ones in the slice sizes are allowed
  // - All other slice sizes match the corresponding dimension except the
  //   first non-singleton slice size
  // If the array is col contiguous then the reverse is the case:
  // - Any number of trailing ones in the slice sizes are allowed
  // - All other slice sizes match the corresponding dimension except the
  //   first non-singleton slice size from the end

  bool can_copy = false;
  if (src.flags().row_contiguous) {
    can_copy = true;

    // Ignore leading 1s
    int i = 0;
    for (; i < slice_sizes.size() && slice_sizes[i] == 1; ++i)
      ;

    // Check the remaining
    i++;
    for (; i < src.ndim() && can_copy; ++i) {
      can_copy = (src.shape(i) == slice_sizes[i]);
    }
  } else if (src.flags().col_contiguous) {
    can_copy = true;

    // Ignore trailing 1s
    int i = slice_sizes.size() - 1;
    for (; i >= 0 && slice_sizes[i] == 1; --i)
      ;

    // Skip the next slice size and check the remaining
    i--;
    for (; i >= 0 && can_copy; --i) {
      can_copy = (src.shape(i) == slice_sizes[i]);
    }
  }
  size_t slice_size = 1;
  for (auto s : slice_sizes) {
    slice_size *= s;
  }
  size_t ind_size = slice_size == 0 ? 0 : out.size() / slice_size;
  const T* src_ptr = src.data<T>();
  T* dst_ptr = out.data<T>();
  size_t out_idx = 0;

  std::vector<ContiguousIterator<size_t>> its(inds.begin(), inds.end());
  ContiguousIterator<size_t> src_it;
  if (!can_copy && src.ndim() > 0) {
    src_it = std::move(
        ContiguousIterator<size_t>(slice_sizes, src.strides(), src.ndim()));
  }
  for (int idx = 0; idx < ind_size; idx++) {
    size_t src_idx = 0;
    for (int ii = 0; ii < inds.size(); ++ii) {
      auto ax = axes[ii];
      auto idx_loc = its[ii].loc;
      its[ii].step();
      auto idx_val =
          offset_neg_idx(inds[ii].data<IdxT>()[idx_loc], src.shape(ax));
      src_idx += (idx_val * src.strides()[ax]);
    }

    if (slice_size == 1) {
      dst_ptr[out_idx++] = src_ptr[src_idx];
    } else if (can_copy) {
      std::copy(
          src_ptr + src_idx, src_ptr + src_idx + slice_size, dst_ptr + out_idx);
      out_idx += slice_size;
    } else {
      for (int jj = 0; jj < slice_size; jj++) {
        dst_ptr[out_idx++] = src_ptr[src_idx + src_it.loc];
        src_it.step();
      }
      src_it.reset();
    }
  }
}

template <typename IdxT>
void dispatch_gather(
    const array& src,
    const std::vector<array>& inds,
    array& out,
    const std::vector<int>& axes,
    const std::vector<int>& size) {
  switch (out.dtype()) {
    case bool_:
      gather<bool, IdxT>(src, inds, out, axes, size);
      break;
    case uint8:
      gather<uint8_t, IdxT>(src, inds, out, axes, size);
      break;
    case uint16:
      gather<uint16_t, IdxT>(src, inds, out, axes, size);
      break;
    case uint32:
      gather<uint32_t, IdxT>(src, inds, out, axes, size);
      break;
    case uint64:
      gather<uint64_t, IdxT>(src, inds, out, axes, size);
      break;
    case int8:
      gather<int8_t, IdxT>(src, inds, out, axes, size);
      break;
    case int16:
      gather<int16_t, IdxT>(src, inds, out, axes, size);
      break;
    case int32:
      gather<int32_t, IdxT>(src, inds, out, axes, size);
      break;
    case int64:
      gather<int64_t, IdxT>(src, inds, out, axes, size);
      break;
    case float16:
      gather<float16_t, IdxT>(src, inds, out, axes, size);
      break;
    case float32:
      gather<float, IdxT>(src, inds, out, axes, size);
      break;
    case bfloat16:
      gather<bfloat16_t, IdxT>(src, inds, out, axes, size);
      break;
    case complex64:
      gather<complex64_t, IdxT>(src, inds, out, axes, size);
      break;
  }
}

void Gather::eval(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& src = inputs[0];
  std::vector<array> inds(inputs.begin() + 1, inputs.end());

  if (inds.empty()) {
    dispatch_gather<bool>(src, inds, out, axes_, slice_sizes_);
    return;
  }

  switch (inds[0].dtype()) {
    case bool_:
      dispatch_gather<bool>(src, inds, out, axes_, slice_sizes_);
      break;
    case uint8:
      dispatch_gather<uint8_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case uint16:
      dispatch_gather<uint16_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case uint32:
      dispatch_gather<uint32_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case uint64:
      dispatch_gather<uint64_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case int8:
      dispatch_gather<int8_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case int16:
      dispatch_gather<int16_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case int32:
      dispatch_gather<int32_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case int64:
      dispatch_gather<int64_t>(src, inds, out, axes_, slice_sizes_);
      break;
    case float16:
    case float32:
    case bfloat16:
    case complex64:
      throw std::runtime_error(
          "[Gather::eval] Cannot gather with floating point indices.");
      break;
  }
}

template <typename InT, typename IdxT, typename OpT>
void scatter(
    const array& updates,
    array& out,
    const std::vector<array>& inds,
    const std::vector<int>& axes,
    const OpT& op) {
  int nind = inds.size();
  auto inds_ndim = updates.ndim() - out.ndim();
  size_t n_updates = nind ? inds[0].size() : 1;

  std::vector<int> update_shape(
      updates.shape().begin() + inds_ndim, updates.shape().end());
  size_t update_size = 1;
  for (auto us : update_shape) {
    update_size *= us;
  }

  std::vector<ContiguousIterator<size_t>> its(inds.begin(), inds.end());
  ContiguousIterator<size_t> update_it(updates);
  ContiguousIterator<size_t> out_it(update_shape, out.strides(), out.ndim());

  for (int i = 0; i < n_updates; ++i) {
    size_t out_offset = 0;
    for (int j = 0; j < nind; ++j) {
      auto ax = axes[j];
      auto idx_loc = its[j].loc;
      its[j].step();
      auto idx_val =
          offset_neg_idx(inds[j].data<IdxT>()[idx_loc], out.shape(ax));
      out_offset += (idx_val * out.strides()[ax]);
    }
    update_it.seek(i * update_size);
    for (int j = 0; j < update_size; ++j) {
      op(updates.data<InT>()[update_it.loc],
         out.data<InT>() + out_offset + out_it.loc);
      update_it.step();
      out_it.step();
    }
    out_it.reset();
    update_it.reset();
  }
}

template <typename InT, typename IdxT>
void dispatch_scatter_inds(
    array& out,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType rtype) {
  switch (rtype) {
    case Scatter::None:
      scatter<InT, IdxT>(
          updates, out, indices, axes, [](auto x, auto* y) { (*y) = x; });
      break;
    case Scatter::Sum:
      scatter<InT, IdxT>(
          updates, out, indices, axes, [](auto x, auto* y) { (*y) += x; });
      break;
    case Scatter::Prod:
      scatter<InT, IdxT>(
          updates, out, indices, axes, [](auto x, auto* y) { (*y) *= x; });
      break;
    case Scatter::Max:
      scatter<InT, IdxT>(updates, out, indices, axes, [](auto x, auto* y) {
        (*y) = (*y > x) ? *y : x;
      });
      break;
    case Scatter::Min:
      scatter<InT, IdxT>(updates, out, indices, axes, [](auto x, auto* y) {
        (*y) = (*y < x) ? *y : x;
      });
      break;
  }
}

template <typename InT>
void dispatch_scatter(
    array& out,
    const std::vector<array>& inds,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType rtype) {
  if (inds.empty()) {
    dispatch_scatter_inds<InT, bool>(out, inds, updates, axes, rtype);
    return;
  }

  switch (inds[0].dtype()) {
    case bool_:
      dispatch_scatter_inds<InT, bool>(out, inds, updates, axes, rtype);
      break;
    case uint8:
      dispatch_scatter_inds<InT, uint8_t>(out, inds, updates, axes, rtype);
      break;
    case uint16:
      dispatch_scatter_inds<InT, uint16_t>(out, inds, updates, axes, rtype);
      break;
    case uint32:
      dispatch_scatter_inds<InT, uint32_t>(out, inds, updates, axes, rtype);
      break;
    case uint64:
      dispatch_scatter_inds<InT, uint64_t>(out, inds, updates, axes, rtype);
      break;
    case int8:
      dispatch_scatter_inds<InT, int8_t>(out, inds, updates, axes, rtype);
      break;
    case int16:
      dispatch_scatter_inds<InT, int16_t>(out, inds, updates, axes, rtype);
      break;
    case int32:
      dispatch_scatter_inds<InT, int32_t>(out, inds, updates, axes, rtype);
      break;
    case int64:
      dispatch_scatter_inds<InT, int64_t>(out, inds, updates, axes, rtype);
      break;
    case float16:
    case float32:
    case bfloat16:
    case complex64:
      throw std::runtime_error(
          "[Scatter::eval_cpu] Cannot scatter with floating point indices.");
  }
}

void Scatter::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() >= 2);

  auto& src = inputs[0];
  std::vector<array> inds(inputs.begin() + 1, inputs.end() - 1);
  auto& updates = inputs.back();

  // Copy src into out (copy allocates memory for out)
  copy(src, out, CopyType::General);

  switch (src.dtype()) {
    case bool_:
      dispatch_scatter<bool>(out, inds, updates, axes_, reduce_type_);
      break;
    case uint8:
      dispatch_scatter<uint8_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case uint16:
      dispatch_scatter<uint16_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case uint32:
      dispatch_scatter<uint32_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case uint64:
      dispatch_scatter<uint64_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case int8:
      dispatch_scatter<int8_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case int16:
      dispatch_scatter<int16_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case int32:
      dispatch_scatter<int32_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case int64:
      dispatch_scatter<int64_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case float16:
      dispatch_scatter<float16_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case float32:
      dispatch_scatter<float>(out, inds, updates, axes_, reduce_type_);
      break;
    case bfloat16:
      dispatch_scatter<bfloat16_t>(out, inds, updates, axes_, reduce_type_);
      break;
    case complex64:
      dispatch_scatter<complex64_t>(out, inds, updates, axes_, reduce_type_);
      break;
  }
}

} // namespace mlx::core
