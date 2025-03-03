// Copyright © 2023-2024 Apple Inc.

#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core {

namespace {

template <typename SrcT, typename DstT>
void copy_single(const array& src, array& dst, Stream stream) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  encoder.dispatch([src_ptr, dst_ptr, size = dst.size()]() {
    auto val = static_cast<DstT>(src_ptr[0]);
    std::fill_n(dst_ptr, size, val);
  });
}

template <typename SrcT, typename DstT>
void copy_vector(const array& src, array& dst, Stream stream) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  size_t size = src.data_size();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  encoder.dispatch([src_ptr, dst_ptr, size = src.data_size()]() {
    std::copy(src_ptr, src_ptr + size, dst_ptr);
  });
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
    Stream stream,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    const std::optional<array>& dynamic_i_offset,
    const std::optional<array>& dynamic_o_offset) {
  auto src_ptr = src.data<SrcT>() + i_offset;
  auto dst_ptr = dst.data<DstT>() + o_offset;
  auto i_offset_ptr =
      dynamic_i_offset ? dynamic_i_offset->data<int64_t>() : nullptr;
  auto o_offset_ptr =
      dynamic_o_offset ? dynamic_o_offset->data<int64_t>() : nullptr;

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  encoder.dispatch([src_ptr,
                    dst_ptr,
                    size = src.size(),
                    data_shape = data_shape,
                    i_strides = i_strides,
                    o_strides = o_strides,
                    i_offset_ptr,
                    o_offset_ptr]() mutable {
    if (data_shape.empty()) {
      auto val = static_cast<DstT>(*src_ptr);
      *dst_ptr = val;
      return;
    }
    auto [shape, strides] =
        collapse_contiguous_dims(data_shape, {i_strides, o_strides});

    int ndim = shape.size();
    if (ndim < 3) {
      if (i_offset_ptr) {
        src_ptr += i_offset_ptr[0];
      }
      if (o_offset_ptr) {
        dst_ptr += o_offset_ptr[0];
      }

      if (ndim == 1) {
        copy_dims<SrcT, DstT, 1>(
            src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
      } else if (ndim == 2) {
        copy_dims<SrcT, DstT, 2>(
            src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
      } else if (ndim == 3) {
        copy_dims<SrcT, DstT, 3>(
            src_ptr, dst_ptr, shape, strides[0], strides[1], 0);
      }
      return;
    }
    if (i_offset_ptr) {
      src_ptr += i_offset_ptr[0];
    }
    if (o_offset_ptr) {
      dst_ptr += o_offset_ptr[0];
    }

    ContiguousIterator in(shape, strides[0], ndim - 3);
    ContiguousIterator out(shape, strides[1], ndim - 3);
    auto stride = std::accumulate(
        shape.end() - 3, shape.end(), 1, std::multiplies<int64_t>());
    for (int64_t elem = 0; elem < size; elem += stride) {
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
  });
}

template <typename SrcT, typename DstT>
inline void copy_general_general(const array& src, array& dst, Stream stream) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      stream,
      src.shape(),
      src.strides(),
      dst.strides(),
      0,
      0,
      std::nullopt,
      std::nullopt);
}

template <typename SrcT, typename DstT>
void copy_general(
    const array& src,
    array& dst,
    Stream stream,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides&,
    int64_t i_offset,
    int64_t o_offset,
    const std::optional<array>& dynamic_i_offset,
    const std::optional<array>& dynamic_o_offset) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      stream,
      data_shape,
      i_strides,
      make_contiguous_strides(data_shape),
      i_offset,
      o_offset,
      dynamic_i_offset,
      dynamic_o_offset);
}

template <typename SrcT, typename DstT>
inline void copy_general(const array& src, array& dst, Stream stream) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
      stream,
      src.shape(),
      src.strides(),
      make_contiguous_strides(src.shape()),
      0,
      0,
      std::nullopt,
      std::nullopt);
}

template <typename SrcT, typename DstT, typename... Args>
void copy(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream,
    Args&&... args) {
  switch (ctype) {
    case CopyType::Scalar:
      copy_single<SrcT, DstT>(src, dst, stream);
      return;
    case CopyType::Vector:
      copy_vector<SrcT, DstT>(src, dst, stream);
      return;
    case CopyType::General:
      copy_general<SrcT, DstT>(src, dst, stream, std::forward<Args>(args)...);
      return;
    case CopyType::GeneralGeneral:
      copy_general_general<SrcT, DstT>(
          src, dst, stream, std::forward<Args>(args)...);
      return;
  }
}

template <typename SrcT, typename... Args>
void copy(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream,
    Args&&... args) {
  switch (dst.dtype()) {
    case bool_:
      copy<SrcT, bool>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint8:
      copy<SrcT, uint8_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint16:
      copy<SrcT, uint16_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint32:
      copy<SrcT, uint32_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint64:
      copy<SrcT, uint64_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int8:
      copy<SrcT, int8_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int16:
      copy<SrcT, int16_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int32:
      copy<SrcT, int32_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int64:
      copy<SrcT, int64_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float16:
      copy<SrcT, float16_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float32:
      copy<SrcT, float>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float64:
      copy<SrcT, double>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case bfloat16:
      copy<SrcT, bfloat16_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case complex64:
      copy<SrcT, complex64_t>(
          src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
  }
}

template <typename... Args>
inline void copy_inplace_dispatch(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream,
    Args&&... args) {
  switch (src.dtype()) {
    case bool_:
      copy<bool>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint8:
      copy<uint8_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint16:
      copy<uint16_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint32:
      copy<uint32_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case uint64:
      copy<uint64_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int8:
      copy<int8_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int16:
      copy<int16_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int32:
      copy<int32_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case int64:
      copy<int64_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float16:
      copy<float16_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float32:
      copy<float>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case float64:
      copy<double>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case bfloat16:
      copy<bfloat16_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
    case complex64:
      copy<complex64_t>(src, dst, ctype, stream, std::forward<Args>(args)...);
      break;
  }
}

} // namespace

void copy_inplace(const array& src, array& dst, CopyType ctype, Stream stream) {
  copy_inplace_dispatch(src, dst, ctype, stream);
}

void copy(const array& src, array& dst, CopyType ctype, Stream stream) {
  bool donated = set_copy_output_data(src, dst, ctype);
  if (donated && src.dtype() == dst.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_inplace(src, dst, ctype, stream);
}

void copy_inplace(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    Stream stream,
    const std::optional<array>& dynamic_i_offset, /* = std::nullopt */
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  switch (ctype) {
    case CopyType::General:
    case CopyType::GeneralGeneral:
      copy_inplace_dispatch(
          src,
          dst,
          ctype,
          stream,
          data_shape,
          i_strides,
          o_strides,
          i_offset,
          o_offset,
          dynamic_i_offset,
          dynamic_o_offset);
      break;
    case CopyType::Scalar:
    case CopyType::Vector:
      copy_inplace_dispatch(src, dst, ctype, stream);
  }
}

} // namespace mlx::core
