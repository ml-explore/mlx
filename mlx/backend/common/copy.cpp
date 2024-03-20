// Copyright © 2023-2024 Apple Inc.

#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"

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
  std::copy(src_ptr, src_ptr + src.data_size(), dst_ptr);
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general_dim1(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    int64_t i_offset) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  stride_t src_idx = i_offset;
  stride_t dst_idx = 0;
  for (int i = 0; i < data_shape[0]; ++i) {
    dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
    src_idx += i_strides[0];
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_dim1(const array& src, array& dst) {
  return copy_general_dim1<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), 0);
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general_dim2(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    int64_t i_offset) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  stride_t src_idx = i_offset;
  stride_t dst_idx = 0;
  for (int i = 0; i < data_shape[0]; ++i) {
    for (int j = 0; j < data_shape[1]; ++j) {
      dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
      src_idx += i_strides[1];
    }
    src_idx += i_strides[0] - i_strides[1] * data_shape[1];
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_dim2(const array& src, array& dst) {
  return copy_general_dim2<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), 0);
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general_dim3(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    int64_t i_offset) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  stride_t src_idx = i_offset;
  stride_t dst_idx = 0;
  for (int i = 0; i < data_shape[0]; ++i) {
    for (int j = 0; j < data_shape[1]; ++j) {
      for (int k = 0; k < data_shape[2]; ++k) {
        dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
        src_idx += i_strides[2];
      }
      src_idx += i_strides[1] - i_strides[2] * data_shape[2];
    }
    src_idx += i_strides[0] - i_strides[1] * data_shape[1];
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_dim3(const array& src, array& dst) {
  return copy_general_dim3<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), 0);
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general_dim4(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    int64_t i_offset) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  stride_t src_idx = i_offset;
  stride_t dst_idx = 0;
  for (int i = 0; i < data_shape[0]; ++i) {
    for (int j = 0; j < data_shape[1]; ++j) {
      for (int k = 0; k < data_shape[2]; ++k) {
        for (int ii = 0; ii < data_shape[3]; ++ii) {
          dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
          src_idx += i_strides[3];
        }
        src_idx += i_strides[2] - i_strides[3] * data_shape[3];
      }
      src_idx += i_strides[1] - i_strides[2] * data_shape[2];
    }
    src_idx += i_strides[0] - i_strides[1] * data_shape[1];
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_dim4(const array& src, array& dst) {
  return copy_general_dim4<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), 0);
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    int64_t i_offset) {
  switch (src.ndim()) {
    case 1:
      copy_general_dim1<SrcT, DstT, stride_t>(
          src, dst, data_shape, i_strides, i_offset);
      return;
    case 2:
      copy_general_dim2<SrcT, DstT, stride_t>(
          src, dst, data_shape, i_strides, i_offset);
      return;
    case 3:
      copy_general_dim3<SrcT, DstT, stride_t>(
          src, dst, data_shape, i_strides, i_offset);
      return;
    case 4:
      copy_general_dim4<SrcT, DstT, stride_t>(
          src, dst, data_shape, i_strides, i_offset);
      return;
  }

  auto src_ptr = src.data<SrcT>() + i_offset;
  auto dst_ptr = dst.data<DstT>();
  for (size_t i = 0; i < dst.size(); ++i) {
    stride_t src_elem = elem_to_loc(i, data_shape, i_strides);
    dst_ptr[i] = static_cast<DstT>(src_ptr[src_elem]);
  }
}

template <typename SrcT, typename DstT>
inline void copy_general(const array& src, array& dst) {
  return copy_general<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), 0);
}

template <typename SrcT, typename DstT, typename stride_t>
inline void copy_general(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    int64_t i_offset,
    int64_t o_offset) {
  return copy_general<SrcT, DstT, stride_t>(
      src, dst, data_shape, i_strides, i_offset);
}

template <typename SrcT, typename DstT, typename stride_t, int D>
inline void copy_general_general_dims(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    stride_t i_offset,
    stride_t o_offset) {
  if constexpr (D > 1) {
    int axis = src.ndim() - D;
    auto stride_src = i_strides[axis];
    auto stride_dst = o_strides[axis];
    auto N = data_shape[axis];
    for (int i = 0; i < N; i++) {
      copy_general_general_dims<SrcT, DstT, stride_t, D - 1>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      i_offset += stride_src;
      o_offset += stride_dst;
    }
  } else {
    int axis = src.ndim() - 1;
    auto stride_src = i_strides[axis];
    auto stride_dst = o_strides[axis];
    auto N = data_shape[axis];
    const SrcT* src_ptr = src.data<SrcT>() + i_offset;
    DstT* dst_ptr = dst.data<DstT>() + o_offset;
    for (int i = 0; i < N; i++) {
      *dst_ptr = static_cast<DstT>(*src_ptr);
      src_ptr += stride_src;
      dst_ptr += stride_dst;
    }
  }
}

template <typename SrcT, typename DstT, typename stride_t>
void copy_general_general(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    stride_t i_offset,
    stride_t o_offset) {
  switch (src.ndim()) {
    case 1:
      copy_general_general_dims<SrcT, DstT, stride_t, 1>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      return;
    case 2:
      copy_general_general_dims<SrcT, DstT, stride_t, 2>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      return;
    case 3:
      copy_general_general_dims<SrcT, DstT, stride_t, 3>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      return;
    case 4:
      copy_general_general_dims<SrcT, DstT, stride_t, 4>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      return;
    case 5:
      copy_general_general_dims<SrcT, DstT, stride_t, 5>(
          src, dst, data_shape, i_strides, o_strides, i_offset, o_offset);
      return;
  }

  int size = std::accumulate(
      data_shape.begin() - 5, data_shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < src.size(); i += size) {
    stride_t src_offset = i_offset + elem_to_loc(i, data_shape, i_strides);
    stride_t dst_offset = o_offset + elem_to_loc(i, dst.shape(), o_strides);
    copy_general_general_dims<SrcT, DstT, stride_t, 5>(
        src, dst, data_shape, i_strides, o_strides, src_offset, dst_offset);
  }
}

template <typename SrcT, typename DstT>
inline void copy_general_general(const array& src, array& dst) {
  return copy_general_general<SrcT, DstT, size_t>(
      src, dst, src.shape(), src.strides(), dst.strides(), 0, 0);
}

template <typename SrcT, typename DstT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args... args) {
  switch (ctype) {
    case CopyType::Scalar:
      copy_single<SrcT, DstT>(src, dst);
      return;
    case CopyType::Vector:
      copy_vector<SrcT, DstT>(src, dst);
      return;
    case CopyType::General:
      copy_general<SrcT, DstT>(src, dst, args...);
      return;
    case CopyType::GeneralGeneral:
      copy_general_general<SrcT, DstT>(src, dst, args...);
  }
}

template <typename SrcT, typename... Args>
void copy(const array& src, array& dst, CopyType ctype, Args... args) {
  switch (dst.dtype()) {
    case bool_:
      copy<SrcT, bool>(src, dst, ctype, args...);
      break;
    case uint8:
      copy<SrcT, uint8_t>(src, dst, ctype, args...);
      break;
    case uint16:
      copy<SrcT, uint16_t>(src, dst, ctype, args...);
      break;
    case uint32:
      copy<SrcT, uint32_t>(src, dst, ctype, args...);
      break;
    case uint64:
      copy<SrcT, uint64_t>(src, dst, ctype, args...);
      break;
    case int8:
      copy<SrcT, int8_t>(src, dst, ctype, args...);
      break;
    case int16:
      copy<SrcT, int16_t>(src, dst, ctype, args...);
      break;
    case int32:
      copy<SrcT, int32_t>(src, dst, ctype, args...);
      break;
    case int64:
      copy<SrcT, int64_t>(src, dst, ctype, args...);
      break;
    case float16:
      copy<SrcT, float16_t>(src, dst, ctype, args...);
      break;
    case float32:
      copy<SrcT, float>(src, dst, ctype, args...);
      break;
    case bfloat16:
      copy<SrcT, bfloat16_t>(src, dst, ctype, args...);
      break;
    case complex64:
      copy<SrcT, complex64_t>(src, dst, ctype, args...);
      break;
  }
}

template <typename... Args>
inline void copy_inplace_dispatch(
    const array& src,
    array& dst,
    CopyType ctype,
    Args... args) {
  switch (src.dtype()) {
    case bool_:
      copy<bool>(src, dst, ctype, args...);
      break;
    case uint8:
      copy<uint8_t>(src, dst, ctype, args...);
      break;
    case uint16:
      copy<uint16_t>(src, dst, ctype, args...);
      break;
    case uint32:
      copy<uint32_t>(src, dst, ctype, args...);
      break;
    case uint64:
      copy<uint64_t>(src, dst, ctype, args...);
      break;
    case int8:
      copy<int8_t>(src, dst, ctype, args...);
      break;
    case int16:
      copy<int16_t>(src, dst, ctype, args...);
      break;
    case int32:
      copy<int32_t>(src, dst, ctype, args...);
      break;
    case int64:
      copy<int64_t>(src, dst, ctype, args...);
      break;
    case float16:
      copy<float16_t>(src, dst, ctype, args...);
      break;
    case float32:
      copy<float>(src, dst, ctype, args...);
      break;
    case bfloat16:
      copy<bfloat16_t>(src, dst, ctype, args...);
      break;
    case complex64:
      copy<complex64_t>(src, dst, ctype, args...);
      break;
  }
}

} // namespace

void copy_inplace(const array& src, array& dst, CopyType ctype) {
  return copy_inplace_dispatch(src, dst, ctype);
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

template <typename stride_t>
void copy_inplace(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<stride_t>& i_strides,
    const std::vector<stride_t>& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype) {
  switch (ctype) {
    case CopyType::General:
    case CopyType::GeneralGeneral:
      return copy_inplace_dispatch(
          src,
          dst,
          ctype,
          data_shape,
          i_strides,
          o_strides,
          i_offset,
          o_offset);

    case CopyType::Scalar:
    case CopyType::Vector:
      return copy_inplace_dispatch(src, dst, ctype);
  }
}

template <>
void copy_inplace<int64_t>(
    const array& src,
    array& dst,
    const std::vector<int>& data_shape,
    const std::vector<int64_t>& i_strides,
    const std::vector<int64_t>& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype) {
  switch (ctype) {
    case CopyType::General:
    case CopyType::GeneralGeneral:
      return copy_inplace_dispatch(
          src,
          dst,
          ctype,
          data_shape,
          i_strides,
          o_strides,
          i_offset,
          o_offset);

    case CopyType::Scalar:
    case CopyType::Vector:
      return copy_inplace_dispatch(src, dst, ctype);
  }
}

} // namespace mlx::core
