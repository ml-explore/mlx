// Copyright © 2023 Apple Inc.

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

template <typename SrcT, typename DstT>
void copy_general_dim1(const array& src, array& dst) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  size_t src_idx = 0;
  size_t dst_idx = 0;
  for (size_t i = 0; i < src.shape()[0]; ++i) {
    dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
    src_idx += src.strides()[0];
  }
}

template <typename SrcT, typename DstT>
void copy_general_dim2(const array& src, array& dst) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  size_t src_idx = 0;
  size_t dst_idx = 0;
  for (size_t i = 0; i < src.shape()[0]; ++i) {
    for (size_t j = 0; j < src.shape()[1]; ++j) {
      dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
      src_idx += src.strides()[1];
    }
    src_idx += src.strides()[0] - src.strides()[1] * src.shape()[1];
  }
}

template <typename SrcT, typename DstT>
void copy_general_dim3(const array& src, array& dst) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  size_t src_idx = 0;
  size_t dst_idx = 0;
  for (size_t i = 0; i < src.shape()[0]; ++i) {
    for (size_t j = 0; j < src.shape()[1]; ++j) {
      for (size_t k = 0; k < src.shape()[2]; ++k) {
        dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
        src_idx += src.strides()[2];
      }
      src_idx += src.strides()[1] - src.strides()[2] * src.shape()[2];
    }
    src_idx += src.strides()[0] - src.strides()[1] * src.shape()[1];
  }
}

template <typename SrcT, typename DstT>
void copy_general_dim4(const array& src, array& dst) {
  const SrcT* src_ptr = src.data<SrcT>();
  DstT* dst_ptr = dst.data<DstT>();
  size_t src_idx = 0;
  size_t dst_idx = 0;
  for (size_t i = 0; i < src.shape()[0]; ++i) {
    for (size_t j = 0; j < src.shape()[1]; ++j) {
      for (size_t k = 0; k < src.shape()[2]; ++k) {
        for (size_t ii = 0; ii < src.shape()[3]; ++ii) {
          dst_ptr[dst_idx++] = static_cast<DstT>(src_ptr[src_idx]);
          src_idx += src.strides()[3];
        }
        src_idx += src.strides()[2] - src.strides()[3] * src.shape()[3];
      }
      src_idx += src.strides()[1] - src.strides()[2] * src.shape()[2];
    }
    src_idx += src.strides()[0] - src.strides()[1] * src.shape()[1];
  }
}

template <typename SrcT, typename DstT>
void copy_general(const array& src, array& dst) {
  switch (src.ndim()) {
    case 1:
      copy_general_dim1<SrcT, DstT>(src, dst);
      return;
    case 2:
      copy_general_dim2<SrcT, DstT>(src, dst);
      return;
    case 3:
      copy_general_dim3<SrcT, DstT>(src, dst);
      return;
    case 4:
      copy_general_dim4<SrcT, DstT>(src, dst);
      return;
  }

  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  for (size_t i = 0; i < dst.size(); ++i) {
    size_t src_elem = elem_to_loc(i, src.shape(), src.strides());
    dst_ptr[i] = static_cast<DstT>(src_ptr[src_elem]);
  }
}

template <typename SrcT, typename DstT, int D>
inline void copy_general_general_dims(
    const array& src,
    array& dst,
    size_t offset_src,
    size_t offset_dst) {
  if constexpr (D > 1) {
    int axis = src.ndim() - D;
    auto stride_src = src.strides()[axis];
    auto stride_dst = dst.strides()[axis];
    auto N = src.shape(axis);
    for (int i = 0; i < N; i++) {
      copy_general_general_dims<SrcT, DstT, D - 1>(
          src, dst, offset_src, offset_dst);
      offset_src += stride_src;
      offset_dst += stride_dst;
    }
  } else {
    int axis = src.ndim() - 1;
    auto stride_src = src.strides()[axis];
    auto stride_dst = dst.strides()[axis];
    auto N = src.shape(axis);
    const SrcT* src_ptr = src.data<SrcT>() + offset_src;
    DstT* dst_ptr = dst.data<DstT>() + offset_dst;
    for (int i = 0; i < N; i++) {
      *dst_ptr = static_cast<DstT>(*src_ptr);
      src_ptr += stride_src;
      dst_ptr += stride_dst;
    }
  }
}

template <typename SrcT, typename DstT>
void copy_general_general(const array& src, array& dst) {
  switch (src.ndim()) {
    case 1:
      copy_general_general_dims<SrcT, DstT, 1>(src, dst, 0, 0);
      return;
    case 2:
      copy_general_general_dims<SrcT, DstT, 2>(src, dst, 0, 0);
      return;
    case 3:
      copy_general_general_dims<SrcT, DstT, 3>(src, dst, 0, 0);
      return;
    case 4:
      copy_general_general_dims<SrcT, DstT, 4>(src, dst, 0, 0);
      return;
    case 5:
      copy_general_general_dims<SrcT, DstT, 5>(src, dst, 0, 0);
      return;
  }

  int size = std::accumulate(
      src.shape().begin() - 5, src.shape().end(), 1, std::multiplies<int>());
  for (int i = 0; i < src.size(); i += size) {
    size_t offset_src = elem_to_loc(i, src.shape(), src.strides());
    size_t offset_dst = elem_to_loc(i, dst.shape(), dst.strides());
    copy_general_general_dims<SrcT, DstT, 5>(src, dst, offset_src, offset_dst);
  }
}

template <typename SrcT, typename DstT>
void copy(const array& src, array& dst, CopyType ctype) {
  switch (ctype) {
    case CopyType::Scalar:
      copy_single<SrcT, DstT>(src, dst);
      return;
    case CopyType::Vector:
      copy_vector<SrcT, DstT>(src, dst);
      return;
    case CopyType::General:
      copy_general<SrcT, DstT>(src, dst);
      return;
    case CopyType::GeneralGeneral:
      copy_general_general<SrcT, DstT>(src, dst);
  }
}

template <typename SrcT>
void copy(const array& src, array& dst, CopyType ctype) {
  switch (dst.dtype()) {
    case bool_:
      copy<SrcT, bool>(src, dst, ctype);
      break;
    case uint8:
      copy<SrcT, uint8_t>(src, dst, ctype);
      break;
    case uint16:
      copy<SrcT, uint16_t>(src, dst, ctype);
      break;
    case uint32:
      copy<SrcT, uint32_t>(src, dst, ctype);
      break;
    case uint64:
      copy<SrcT, uint64_t>(src, dst, ctype);
      break;
    case int8:
      copy<SrcT, int8_t>(src, dst, ctype);
      break;
    case int16:
      copy<SrcT, int16_t>(src, dst, ctype);
      break;
    case int32:
      copy<SrcT, int32_t>(src, dst, ctype);
      break;
    case int64:
      copy<SrcT, int64_t>(src, dst, ctype);
      break;
    case float16:
      copy<SrcT, float16_t>(src, dst, ctype);
      break;
    case float32:
      copy<SrcT, float>(src, dst, ctype);
      break;
    case bfloat16:
      copy<SrcT, bfloat16_t>(src, dst, ctype);
      break;
    case complex64:
      copy<SrcT, complex64_t>(src, dst, ctype);
      break;
  }
}

} // namespace

void copy_inplace(const array& src, array& dst, CopyType ctype) {
  switch (src.dtype()) {
    case bool_:
      copy<bool>(src, dst, ctype);
      break;
    case uint8:
      copy<uint8_t>(src, dst, ctype);
      break;
    case uint16:
      copy<uint16_t>(src, dst, ctype);
      break;
    case uint32:
      copy<uint32_t>(src, dst, ctype);
      break;
    case uint64:
      copy<uint64_t>(src, dst, ctype);
      break;
    case int8:
      copy<int8_t>(src, dst, ctype);
      break;
    case int16:
      copy<int16_t>(src, dst, ctype);
      break;
    case int32:
      copy<int32_t>(src, dst, ctype);
      break;
    case int64:
      copy<int64_t>(src, dst, ctype);
      break;
    case float16:
      copy<float16_t>(src, dst, ctype);
      break;
    case float32:
      copy<float>(src, dst, ctype);
      break;
    case bfloat16:
      copy<bfloat16_t>(src, dst, ctype);
      break;
    case complex64:
      copy<complex64_t>(src, dst, ctype);
      break;
  }
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

} // namespace mlx::core
