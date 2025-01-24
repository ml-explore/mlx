// Copyright © 2023-2024 Apple Inc.

#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/simd/simd.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

namespace {

template <typename SrcT, typename DstT>
void copy_single(const array& src, array& dst) {
  auto val = static_cast<DstT>(src.data<SrcT>()[0]);
  auto dst_ptr = dst.data<DstT>();
  for (int i = 0; i < dst.size(); ++i) {
    dst_ptr[i] = val;
  }
}

template <typename SrcT, typename DstT>
void copy_vector(const array& src, array& dst) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  size_t size = src.data_size();
  std::copy(src_ptr, src_ptr + src.data_size(), dst_ptr);
}

template <typename SrcT, typename DstT, int D>
inline void copy_dims(
    const SrcT* src,
    DstT* dst,
    const Shape& shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int axis) {
  auto stride_src = i_strides[axis];
  auto stride_dst = o_strides[axis];
  auto N = shape[axis];

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      copy_dims<SrcT, DstT, D - 1>(
          src, dst, shape, i_strides, o_strides, axis + 1);
    } else {
      *dst = static_cast<DstT>(*src);
    }
    src += stride_src;
    dst += stride_dst;
  }
}

template <typename SrcT, typename DstT>
void copy_general_general(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset) {
  if (data_shape.empty()) {
    auto val = static_cast<DstT>(*(src.data<SrcT>() + i_offset));
    auto dst_ptr = dst.data<DstT>() + o_offset;
    *dst_ptr = val;
    return;
  }
  auto [shape, strides] =
      collapse_contiguous_dims(data_shape, {i_strides, o_strides});
  auto src_ptr = src.data<SrcT>() + i_offset;
  auto dst_ptr = dst.data<DstT>() + o_offset;
  int ndim = shape.size();
  if (ndim == 1) {
    copy_dims<SrcT, DstT, 1>(
        src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    return;
  } else if (ndim == 2) {
    copy_dims<SrcT, DstT, 2>(
        src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    return;
  } else if (ndim == 3) {
    copy_dims<SrcT, DstT, 3>(
        src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
    return;
  }
  ContiguousIterator in(shape, strides[0], ndim - 3);
  ContiguousIterator out(shape, strides[1], ndim - 3);
  auto stride = std::accumulate(
      shape.end() - 3, shape.end(), 1, std::multiplies<int64_t>());
  for (int64_t elem = 0; elem < src.size(); elem += stride) {
    copy_dims<SrcT, DstT, 3>(
        src_ptr + in.loc,
        dst_ptr + out.loc,
        shape,
        strides[0],
        strides[1],
        ndim - 3);
    in.step();
    out.step();
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_general(const array& src, array& dst) {
  copy_general_general<SrcT, DstT>(
      src, dst, src.shape(), src.strides(), dst.strides(), 0, 0);
}

template <typename SrcT, typename DstT>
void copy_general(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides&,
    int64_t i_offset,
    int64_t o_offset) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      data_shape,
      i_strides,
      make_contiguous_strides(data_shape),
      i_offset,
      o_offset);
}

template <typename SrcT, typename DstT>
inline void copy_general(const array& src, array& dst) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      src.shape(),
      src.strides(),
      make_contiguous_strides(src.shape()),
      0,
      0);
}

template <typename SrcT, typename DstT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args&&... args) {
  switch (ctype) {
    case CopyType::Scalar:
      copy_single<SrcT, DstT>(src, dst);
      return;
    case CopyType::Vector:
      copy_vector<SrcT, DstT>(src, dst);
      return;
    case CopyType::General:
      copy_general<SrcT, DstT>(src, dst, std::forward<Args>(args)...);
      return;
    case CopyType::GeneralGeneral:
      copy_general_general<SrcT, DstT>(src, dst, std::forward<Args>(args)...);
      return;
  }
}

template <typename SrcT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args&&... args) {
  switch (dst.dtype()) {
    case bool_:
      copy<SrcT, bool>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint8:
      copy<SrcT, uint8_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint16:
      copy<SrcT, uint16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint32:
      copy<SrcT, uint32_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint64:
      copy<SrcT, uint64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int8:
      copy<SrcT, int8_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int16:
      copy<SrcT, int16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int32:
      copy<SrcT, int32_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int64:
      copy<SrcT, int64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case float16:
      copy<SrcT, float16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case float32:
      copy<SrcT, float>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case bfloat16:
      copy<SrcT, bfloat16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case complex64:
      copy<SrcT, complex64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
  }
}

template <typename... Args>
inline void copy_inplace_dispatch(
    const array& src,
    array& dst,
    CopyType ctype,
    Args&&... args) {
  switch (src.dtype()) {
    case bool_:
      copy<bool>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint8:
      copy<uint8_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint16:
      copy<uint16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint32:
      copy<uint32_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case uint64:
      copy<uint64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int8:
      copy<int8_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int16:
      copy<int16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int32:
      copy<int32_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case int64:
      copy<int64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case float16:
      copy<float16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case float32:
      copy<float>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case bfloat16:
      copy<bfloat16_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
    case complex64:
      copy<complex64_t>(src, dst, ctype, std::forward<Args>(args)...);
      break;
  }
}

} // namespace

void copy_inplace(const array& src, array& dst, CopyType ctype) {
  copy_inplace_dispatch(src, dst, ctype);
}

void copy(const array& src, array& dst, CopyType ctype) {
  // Allocate the output
  switch (ctype) {
    case CopyType::Vector:
      if (src.is_donatable() && src.itemsize() == dst.itemsize()) {
        dst.copy_shared_buffer(src);
      } else {
        auto size = src.data_size();
        dst.set_data(
            allocator::malloc_or_wait(size * dst.itemsize()),
            size,
            src.strides(),
            src.flags());
      }
      break;
    case CopyType::Scalar:
    case CopyType::General:
    case CopyType::GeneralGeneral:
      dst.set_data(allocator::malloc_or_wait(dst.nbytes()));
      break;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_inplace(src, dst, ctype);
}

void copy_inplace(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype) {
  switch (ctype) {
    case CopyType::General:
    case CopyType::GeneralGeneral:
      copy_inplace_dispatch(
          src,
          dst,
          ctype,
          data_shape,
          i_strides,
          o_strides,
          i_offset,
          o_offset);
      break;
    case CopyType::Scalar:
    case CopyType::Vector:
      copy_inplace_dispatch(src, dst, ctype);
  }
}

} // namespace mlx::core
