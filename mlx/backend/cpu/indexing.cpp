// Copyright © 2023 Apple Inc.
#include <algorithm>
#include <cassert>
#include <cmath>

#include "mlx/allocator.h"
#include "mlx/primitives.h"

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

template <typename IdxT>
inline size_t offset_neg_idx(IdxT idx, size_t size) {
  return (idx < 0) ? idx + size : idx;
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
    const Shape& slice_sizes,
    Stream stream) {
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

  std::vector<ContiguousIterator> its(inds.begin(), inds.end());
  ContiguousIterator src_it;
  if (!can_copy && src.ndim() > 0) {
    src_it = ContiguousIterator(slice_sizes, src.strides(), src.ndim());
  }

  std::vector<const IdxT*> ind_ptrs;
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  for (auto& idx : inds) {
    ind_ptrs.push_back(idx.data<IdxT>());
    encoder.set_input_array(idx);
  }
  encoder.set_output_array(out);
  encoder.dispatch([src_ptr,
                    dst_ptr,
                    ind_ptrs = std::move(ind_ptrs),
                    axes,
                    ind_size,
                    slice_size,
                    src_shape = src.shape(),
                    src_strides = src.strides(),
                    src_it = std::move(src_it),
                    its = std::move(its),
                    can_copy]() mutable {
    size_t out_idx = 0;
    for (int idx = 0; idx < ind_size; idx++) {
      size_t src_idx = 0;
      for (int ii = 0; ii < ind_ptrs.size(); ++ii) {
        auto ax = axes[ii];
        auto idx_loc = its[ii].loc;
        its[ii].step();
        auto idx_val = offset_neg_idx(ind_ptrs[ii][idx_loc], src_shape[ax]);
        src_idx += (idx_val * src_strides[ax]);
      }

      if (slice_size == 1) {
        dst_ptr[out_idx++] = src_ptr[src_idx];
      } else if (can_copy) {
        std::copy(
            src_ptr + src_idx,
            src_ptr + src_idx + slice_size,
            dst_ptr + out_idx);
        out_idx += slice_size;
      } else {
        for (int jj = 0; jj < slice_size; jj++) {
          dst_ptr[out_idx++] = src_ptr[src_idx + src_it.loc];
          src_it.step();
        }
        src_it.reset();
      }
    }
  });
}

template <typename IdxT>
void dispatch_gather(
    const array& src,
    const std::vector<array>& inds,
    array& out,
    const std::vector<int>& axes,
    const Shape& size,
    Stream stream) {
  switch (out.dtype()) {
    case bool_:
      gather<bool, IdxT>(src, inds, out, axes, size, stream);
      break;
    case uint8:
      gather<uint8_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case uint16:
      gather<uint16_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case uint32:
      gather<uint32_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case uint64:
      gather<uint64_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case int8:
      gather<int8_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case int16:
      gather<int16_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case int32:
      gather<int32_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case int64:
      gather<int64_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case float16:
      gather<float16_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case float32:
      gather<float, IdxT>(src, inds, out, axes, size, stream);
      break;
    case float64:
      gather<double, IdxT>(src, inds, out, axes, size, stream);
      break;
    case bfloat16:
      gather<bfloat16_t, IdxT>(src, inds, out, axes, size, stream);
      break;
    case complex64:
      gather<complex64_t, IdxT>(src, inds, out, axes, size, stream);
      break;
  }
}

void Gather::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto& src = inputs[0];
  std::vector<array> inds(inputs.begin() + 1, inputs.end());

  if (inds.empty()) {
    dispatch_gather<uint8_t>(src, inds, out, axes_, slice_sizes_, stream());
    return;
  }

  switch (inds[0].dtype()) {
    case uint8:
      dispatch_gather<uint8_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case uint16:
      dispatch_gather<uint16_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case uint32:
      dispatch_gather<uint32_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case uint64:
      dispatch_gather<uint64_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case int8:
      dispatch_gather<int8_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case int16:
      dispatch_gather<int16_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case int32:
      dispatch_gather<int32_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    case int64:
      dispatch_gather<int64_t>(src, inds, out, axes_, slice_sizes_, stream());
      break;
    default:
      throw std::runtime_error(
          "[Gather::eval_cpu] Cannot gather with indices type.");
      break;
  }
}
template <typename T, typename IdxT>
void gather_axis(
    const array& src,
    const array& ind,
    array& out,
    const int axis,
    Stream stream) {
  auto strides = ind.strides();
  strides.erase(strides.begin() + axis);
  auto shape = ind.shape();
  shape.erase(shape.begin() + axis);
  ContiguousIterator ind_it(shape, strides, src.ndim() - 1);

  strides = src.strides();
  strides.erase(strides.begin() + axis);
  ContiguousIterator src_it(shape, strides, src.ndim() - 1);

  auto ind_ptr = ind.data<IdxT>();
  auto src_ptr = src.data<T>();
  auto dst_ptr = out.data<T>();
  auto ind_ax_stride = ind.strides(axis);
  auto src_ax_stride = src.strides(axis);
  auto dst_ax_stride = out.strides(axis);
  auto ind_ax_size = ind.shape(axis);
  auto src_ax_size = src.shape(axis);

  size_t size_pre = 1;
  size_t size_post = 1;
  for (int i = 0; i < axis; ++i) {
    size_pre *= ind.shape(i);
  }
  for (int i = axis + 1; i < ind.ndim(); ++i) {
    size_post *= ind.shape(i);
  }

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_input_array(ind);
  encoder.set_output_array(out);

  encoder.dispatch([ind_ptr,
                    src_ptr,
                    dst_ptr,
                    size_pre,
                    size_post,
                    ind_ax_size,
                    src_ax_size,
                    ind_ax_stride,
                    src_ax_stride,
                    dst_ax_stride,
                    ind_it = std::move(ind_it),
                    src_it = std::move(src_it)]() mutable {
    size_t stride_pre = size_post * ind_ax_size;
    for (size_t i = 0; i < size_pre; i++) {
      for (size_t k = 0; k < size_post; k++) {
        for (int j = 0; j < ind_ax_size; ++j) {
          auto ind_val = offset_neg_idx(
              ind_ptr[ind_it.loc + j * ind_ax_stride], src_ax_size);
          dst_ptr[k + j * dst_ax_stride] =
              src_ptr[src_it.loc + ind_val * src_ax_stride];
        }
        ind_it.step();
        src_it.step();
      }
      dst_ptr += stride_pre;
    }
  });
}

template <typename IdxT>
void dispatch_gather_axis(
    const array& src,
    const array& inds,
    array& out,
    const int axis,
    Stream stream) {
  switch (out.dtype()) {
    case bool_:
      gather_axis<bool, IdxT>(src, inds, out, axis, stream);
      break;
    case uint8:
      gather_axis<uint8_t, IdxT>(src, inds, out, axis, stream);
      break;
    case uint16:
      gather_axis<uint16_t, IdxT>(src, inds, out, axis, stream);
      break;
    case uint32:
      gather_axis<uint32_t, IdxT>(src, inds, out, axis, stream);
      break;
    case uint64:
      gather_axis<uint64_t, IdxT>(src, inds, out, axis, stream);
      break;
    case int8:
      gather_axis<int8_t, IdxT>(src, inds, out, axis, stream);
      break;
    case int16:
      gather_axis<int16_t, IdxT>(src, inds, out, axis, stream);
      break;
    case int32:
      gather_axis<int32_t, IdxT>(src, inds, out, axis, stream);
      break;
    case int64:
      gather_axis<int64_t, IdxT>(src, inds, out, axis, stream);
      break;
    case float16:
      gather_axis<float16_t, IdxT>(src, inds, out, axis, stream);
      break;
    case float32:
      gather_axis<float, IdxT>(src, inds, out, axis, stream);
      break;
    case float64:
      gather_axis<double, IdxT>(src, inds, out, axis, stream);
      break;
    case bfloat16:
      gather_axis<bfloat16_t, IdxT>(src, inds, out, axis, stream);
      break;
    case complex64:
      gather_axis<complex64_t, IdxT>(src, inds, out, axis, stream);
      break;
  }
}

void GatherAxis::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& src = inputs[0];
  auto& inds = inputs[1];
  switch (inds.dtype()) {
    case uint8:
      dispatch_gather_axis<uint8_t>(src, inds, out, axis_, stream());
      break;
    case uint16:
      dispatch_gather_axis<uint16_t>(src, inds, out, axis_, stream());
      break;
    case uint32:
      dispatch_gather_axis<uint32_t>(src, inds, out, axis_, stream());
      break;
    case uint64:
      dispatch_gather_axis<uint64_t>(src, inds, out, axis_, stream());
      break;
    case int8:
      dispatch_gather_axis<int8_t>(src, inds, out, axis_, stream());
      break;
    case int16:
      dispatch_gather_axis<int16_t>(src, inds, out, axis_, stream());
      break;
    case int32:
      dispatch_gather_axis<int32_t>(src, inds, out, axis_, stream());
      break;
    case int64:
      dispatch_gather_axis<int64_t>(src, inds, out, axis_, stream());
      break;
    default:
      throw std::runtime_error(
          "[GatherAxis::eval_cpu] Cannot gather with indices type.");
      break;
  }
}

template <typename InT, typename IdxT, typename OpT>
void scatter(
    const array& updates,
    array& out,
    const std::vector<array>& inds,
    const std::vector<int>& axes,
    const OpT& op,
    Stream stream) {
  int nind = inds.size();
  auto inds_ndim = updates.ndim() - out.ndim();
  size_t n_updates = nind ? inds[0].size() : 1;

  Shape update_shape(
      updates.shape().begin() + inds_ndim, updates.shape().end());
  size_t update_size = 1;
  for (auto us : update_shape) {
    update_size *= us;
  }

  std::vector<ContiguousIterator> its(inds.begin(), inds.end());
  ContiguousIterator update_it(updates);
  ContiguousIterator out_it(update_shape, out.strides(), out.ndim());

  std::vector<const IdxT*> ind_ptrs;
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(updates);
  for (auto& idx : inds) {
    ind_ptrs.push_back(idx.data<IdxT>());
    encoder.set_input_array(idx);
  }
  encoder.set_output_array(out);
  encoder.dispatch([out_ptr = out.data<InT>(),
                    upd_ptr = updates.data<InT>(),
                    ind_ptrs = std::move(ind_ptrs),
                    axes,
                    n_updates,
                    update_size,
                    op = std::move(op),
                    out_shape = out.shape(),
                    out_strides = out.strides(),
                    out_it = std::move(out_it),
                    update_it = std::move(update_it),
                    its = std::move(its)]() mutable {
    for (int i = 0; i < n_updates; ++i) {
      size_t out_offset = 0;
      for (int j = 0; j < ind_ptrs.size(); ++j) {
        auto ax = axes[j];
        auto idx_loc = its[j].loc;
        its[j].step();
        auto idx_val = offset_neg_idx(ind_ptrs[j][idx_loc], out_shape[ax]);
        out_offset += (idx_val * out_strides[ax]);
      }
      update_it.seek(i * update_size);
      for (int j = 0; j < update_size; ++j) {
        op(upd_ptr[update_it.loc], out_ptr + out_offset + out_it.loc);
        update_it.step();
        out_it.step();
      }
      out_it.reset();
      update_it.reset();
    }
  });
}

template <typename InT, typename IdxT>
void dispatch_scatter_inds(
    array& out,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType rtype,
    Stream stream) {
  switch (rtype) {
    case Scatter::None:
      scatter<InT, IdxT>(
          updates,
          out,
          indices,
          axes,
          [](auto x, auto* y) { (*y) = x; },
          stream);
      break;
    case Scatter::Sum:
      scatter<InT, IdxT>(
          updates,
          out,
          indices,
          axes,
          [](auto x, auto* y) { (*y) += x; },
          stream);
      break;
    case Scatter::Prod:
      scatter<InT, IdxT>(
          updates,
          out,
          indices,
          axes,
          [](auto x, auto* y) { (*y) *= x; },
          stream);
      break;
    case Scatter::Max:
      scatter<InT, IdxT>(
          updates,
          out,
          indices,
          axes,
          [](auto x, auto* y) { (*y) = (*y > x) ? *y : x; },
          stream);
      break;
    case Scatter::Min:
      scatter<InT, IdxT>(
          updates,
          out,
          indices,
          axes,
          [](auto x, auto* y) { (*y) = (*y < x) ? *y : x; },
          stream);
      break;
  }
}

template <typename InT>
void dispatch_scatter(
    array& out,
    const std::vector<array>& inds,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType rtype,
    Stream stream) {
  if (inds.empty()) {
    dispatch_scatter_inds<InT, uint8_t>(
        out, inds, updates, axes, rtype, stream);
    return;
  }

  switch (inds[0].dtype()) {
    case uint8:
      dispatch_scatter_inds<InT, uint8_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case uint16:
      dispatch_scatter_inds<InT, uint16_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case uint32:
      dispatch_scatter_inds<InT, uint32_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case uint64:
      dispatch_scatter_inds<InT, uint64_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case int8:
      dispatch_scatter_inds<InT, int8_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case int16:
      dispatch_scatter_inds<InT, int16_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case int32:
      dispatch_scatter_inds<InT, int32_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    case int64:
      dispatch_scatter_inds<InT, int64_t>(
          out, inds, updates, axes, rtype, stream);
      break;
    default:
      throw std::runtime_error(
          "[Scatter::eval_cpu] Cannot scatter with indices type.");
  }
}

void Scatter::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() >= 2);

  auto& src = inputs[0];
  std::vector<array> inds(inputs.begin() + 1, inputs.end() - 1);
  auto& updates = inputs.back();

  // Copy src into out (copy allocates memory for out)
  auto ctype =
      src.flags().row_contiguous ? CopyType::Vector : CopyType::General;
  copy(src, out, ctype, stream());

  switch (src.dtype()) {
    case bool_:
      dispatch_scatter<bool>(out, inds, updates, axes_, reduce_type_, stream());
      break;
    case uint8:
      dispatch_scatter<uint8_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case uint16:
      dispatch_scatter<uint16_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case uint32:
      dispatch_scatter<uint32_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case uint64:
      dispatch_scatter<uint64_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case int8:
      dispatch_scatter<int8_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case int16:
      dispatch_scatter<int16_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case int32:
      dispatch_scatter<int32_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case int64:
      dispatch_scatter<int64_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case float16:
      dispatch_scatter<float16_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case float32:
      dispatch_scatter<float>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case float64:
      dispatch_scatter<double>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case bfloat16:
      dispatch_scatter<bfloat16_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
    case complex64:
      dispatch_scatter<complex64_t>(
          out, inds, updates, axes_, reduce_type_, stream());
      break;
  }
}

template <typename T, typename IdxT, typename OpT>
void scatter_axis(
    array& out,
    const array idx,
    const array& upd,
    int axis,
    const OpT& op,
    Stream stream) {
  auto strides = idx.strides();
  strides.erase(strides.begin() + axis);
  auto shape = idx.shape();
  shape.erase(shape.begin() + axis);
  ContiguousIterator idx_it(shape, strides, upd.ndim() - 1);

  strides = upd.strides();
  strides.erase(strides.begin() + axis);
  ContiguousIterator upd_it(shape, strides, upd.ndim() - 1);

  auto idx_ptr = idx.data<IdxT>();
  auto upd_ptr = upd.data<T>();
  auto dst_ptr = out.data<T>();
  auto idx_ax_stride = idx.strides(axis);
  auto upd_ax_stride = upd.strides(axis);
  auto dst_ax_stride = out.strides(axis);
  auto idx_ax_size = idx.shape(axis);
  auto dst_ax_size = out.shape(axis);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(idx);
  encoder.set_input_array(upd);
  encoder.set_output_array(out);

  size_t size_pre = 1;
  size_t size_post = 1;
  for (int i = 0; i < axis; ++i) {
    size_pre *= idx.shape(i);
  }
  for (int i = axis + 1; i < idx.ndim(); ++i) {
    size_post *= idx.shape(i);
  }
  encoder.dispatch([idx_ptr,
                    upd_ptr,
                    dst_ptr,
                    size_pre,
                    size_post,
                    idx_ax_size,
                    dst_ax_size,
                    idx_ax_stride,
                    upd_ax_stride,
                    dst_ax_stride,
                    idx_it = std::move(idx_it),
                    upd_it = std::move(upd_it),
                    op = std::move(op)]() mutable {
    size_t stride_pre = size_post * dst_ax_size;
    for (size_t i = 0; i < size_pre; i++) {
      for (size_t k = 0; k < size_post; k++) {
        for (int j = 0; j < idx_ax_size; ++j) {
          auto ind_val = offset_neg_idx(
              idx_ptr[idx_it.loc + j * idx_ax_stride], dst_ax_size);
          op(upd_ptr[upd_it.loc + j * upd_ax_stride],
             dst_ptr + k + ind_val * dst_ax_stride);
        }
        idx_it.step();
        upd_it.step();
      }
      dst_ptr += stride_pre;
    }
  });
}

template <typename InT, typename IdxT>
void dispatch_scatter_axis_op(
    array& out,
    const array& idx,
    const array& updates,
    int axis,
    ScatterAxis::ReduceType rtype,
    Stream stream) {
  switch (rtype) {
    case ScatterAxis::None:
      scatter_axis<InT, IdxT>(
          out, idx, updates, axis, [](auto x, auto* y) { (*y) = x; }, stream);
      break;
    case ScatterAxis::Sum:
      scatter_axis<InT, IdxT>(
          out, idx, updates, axis, [](auto x, auto* y) { (*y) += x; }, stream);
      break;
  }
}

template <typename InT>
void dispatch_scatter_axis(
    array& out,
    const array& idx,
    const array& updates,
    int axis,
    ScatterAxis::ReduceType rtype,
    Stream stream) {
  switch (idx.dtype()) {
    case uint8:
      dispatch_scatter_axis_op<InT, uint8_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case uint16:
      dispatch_scatter_axis_op<InT, uint16_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case uint32:
      dispatch_scatter_axis_op<InT, uint32_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case uint64:
      dispatch_scatter_axis_op<InT, uint64_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case int8:
      dispatch_scatter_axis_op<InT, int8_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case int16:
      dispatch_scatter_axis_op<InT, int16_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case int32:
      dispatch_scatter_axis_op<InT, int32_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    case int64:
      dispatch_scatter_axis_op<InT, int64_t>(
          out, idx, updates, axis, rtype, stream);
      break;
    default:
      throw std::runtime_error(
          "[ScatterAxis::eval_cpu] Cannot scatter with indices type.");
  }
}

void ScatterAxis::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() >= 2);

  auto& src = inputs[0];
  auto& idx = inputs[1];
  auto& updates = inputs[2];

  // Copy src into out (copy allocates memory for out)
  auto ctype =
      src.flags().row_contiguous ? CopyType::Vector : CopyType::General;
  copy(src, out, ctype, stream());

  switch (src.dtype()) {
    case bool_:
      dispatch_scatter_axis<bool>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case uint8:
      dispatch_scatter_axis<uint8_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case uint16:
      dispatch_scatter_axis<uint16_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case uint32:
      dispatch_scatter_axis<uint32_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case uint64:
      dispatch_scatter_axis<uint64_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case int8:
      dispatch_scatter_axis<int8_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case int16:
      dispatch_scatter_axis<int16_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case int32:
      dispatch_scatter_axis<int32_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case int64:
      dispatch_scatter_axis<int64_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case float16:
      dispatch_scatter_axis<float16_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case float32:
      dispatch_scatter_axis<float>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case float64:
      dispatch_scatter_axis<double>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case bfloat16:
      dispatch_scatter_axis<bfloat16_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
    case complex64:
      dispatch_scatter_axis<complex64_t>(
          out, idx, updates, axis_, reduce_type_, stream());
      break;
  }
}

} // namespace mlx::core
