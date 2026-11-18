// Copyright © 2023-2024 Apple Inc.

#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/backend/cpu/threading.h"
#include "mlx/dtype_utils.h"

namespace mlx::core {

namespace {

template <typename SrcT, typename DstT>
void copy_single(const array& src, array& dst) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  int64_t size = dst.size();
  auto val = static_cast<DstT>(src_ptr[0]);
  cpu::parallel_for(
      size, cpu::elementwise_grain<DstT>(), [&](int64_t begin, int64_t end) {
        std::fill_n(dst_ptr + begin, end - begin, val);
      });
}

template <typename SrcT, typename DstT>
void copy_vector(const array& src, array& dst) {
  auto src_ptr = src.data<SrcT>();
  auto dst_ptr = dst.data<DstT>();
  int64_t size = src.data_size();
  constexpr int64_t grain = cpu::elementwise_grain<
      std::conditional_t<(sizeof(SrcT) > sizeof(DstT)), SrcT, DstT>>();
  cpu::parallel_for(size, grain, [&](int64_t begin, int64_t end) {
    if constexpr (std::is_same_v<SrcT, DstT>) {
      std::copy(src_ptr + begin, src_ptr + end, dst_ptr + begin);
    } else {
      std::transform(
          src_ptr + begin, src_ptr + end, dst_ptr + begin, [](SrcT x) {
            return static_cast<DstT>(x);
          });
    }
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

// Tile size (in elements) for transpose-like copies.
constexpr int64_t kCopyTile = 32;

// dst[i * d_m + j] = src[i * s_m + j * s_n] for i < M, j < N, processed in
// kCopyTile x kCopyTile tiles.
template <typename SrcT, typename DstT>
inline void copy_2d_tiled(
    const SrcT* src,
    DstT* dst,
    int64_t M,
    int64_t N,
    int64_t s_m,
    int64_t s_n,
    int64_t d_m) {
  for (int64_t j0 = 0; j0 < N; j0 += kCopyTile) {
    int64_t j1 = std::min(j0 + kCopyTile, N);
    for (int64_t i = 0; i < M; ++i) {
      const SrcT* s = src + i * s_m;
      DstT* d = dst + i * d_m;
      for (int64_t j = j0; j < j1; ++j) {
        d[j] = static_cast<DstT>(s[j * s_n]);
      }
    }
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
    int64_t o_offset,
    const std::optional<array>& dynamic_i_offset,
    const std::optional<array>& dynamic_o_offset) {
  auto src_ptr = src.data<SrcT>() + i_offset;
  auto dst_ptr = dst.data<DstT>() + o_offset;
  auto i_offset_ptr =
      dynamic_i_offset ? dynamic_i_offset->data<int64_t>() : nullptr;
  auto o_offset_ptr =
      dynamic_o_offset ? dynamic_o_offset->data<int64_t>() : nullptr;
  if (data_shape.empty()) {
    auto val = static_cast<DstT>(*src_ptr);
    *dst_ptr = val;
    return;
  }
  auto [shape, strides] =
      collapse_contiguous_dims(data_shape, {i_strides, o_strides});
  if (i_offset_ptr) {
    src_ptr += i_offset_ptr[0];
  }
  if (o_offset_ptr) {
    dst_ptr += o_offset_ptr[0];
  }

  int ndim = shape.size();
  int64_t size = std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  if (size == 0) {
    return;
  }
  constexpr int64_t grain = cpu::elementwise_grain<
      std::conditional_t<(sizeof(SrcT) > sizeof(DstT)), SrcT, DstT>>();

  // Transpose-like copy: the innermost destination axis is contiguous but
  // the source walks it with a large stride, while the next axis is
  // contiguous in the source. Copy in square tiles so both sides stay in
  // cache instead of touching a new source cache line per element.
  if (ndim >= 2 && strides[1][ndim - 1] == 1 && strides[0][ndim - 2] == 1 &&
      strides[0][ndim - 1] >= 16) {
    int64_t M = shape[ndim - 2];
    int64_t N = shape[ndim - 1];
    int64_t s_m = strides[0][ndim - 2];
    int64_t s_n = strides[0][ndim - 1];
    int64_t d_m = strides[1][ndim - 2];
    int64_t outer = size / (M * N);
    int64_t n_blocks = (M + kCopyTile - 1) / kCopyTile;
    int64_t work_grain =
        std::max<int64_t>(grain / std::max<int64_t>(kCopyTile * N, 1), 1);
    cpu::parallel_for(
        outer * n_blocks, work_grain, [&](int64_t begin, int64_t end) {
          ContiguousIterator in(shape, strides[0], ndim - 2);
          ContiguousIterator out(shape, strides[1], ndim - 2);
          int64_t slab = begin / n_blocks;
          in.seek(slab);
          out.seek(slab);
          for (int64_t w = begin; w < end; ++w) {
            int64_t s = w / n_blocks;
            if (s != slab) {
              in.step();
              out.step();
              slab = s;
            }
            int64_t m0 = (w % n_blocks) * kCopyTile;
            int64_t m1 = std::min(m0 + kCopyTile, M);
            copy_2d_tiled(
                src_ptr + in.loc + m0 * s_m,
                dst_ptr + out.loc + m0 * d_m,
                m1 - m0,
                N,
                s_m,
                s_n,
                d_m);
          }
        });
    return;
  }

  if (ndim < 3) {
    int64_t outer = shape[0];
    int64_t inner = size / outer;
    cpu::parallel_for(
        outer,
        std::max<int64_t>(grain / std::max<int64_t>(inner, 1), 1),
        [&](int64_t begin, int64_t end) {
          Shape shape_ = shape;
          shape_[0] = end - begin;
          auto s = src_ptr + begin * strides[0][0];
          auto d = dst_ptr + begin * strides[1][0];
          if (ndim == 1) {
            copy_dims<SrcT, DstT, 1>(s, d, shape_, strides[0], strides[1], 0);
          } else {
            copy_dims<SrcT, DstT, 2>(s, d, shape_, strides[0], strides[1], 0);
          }
        });
    return;
  }

  // Iterate over the leading ndim - 3 axes and copy 3D blocks.
  int64_t block = std::accumulate(
      shape.end() - 3, shape.end(), int64_t{1}, std::multiplies<int64_t>());
  int64_t n_outer = size / block;
  cpu::parallel_for(
      n_outer,
      std::max<int64_t>(grain / std::max<int64_t>(block, 1), 1),
      [&](int64_t begin, int64_t end) {
        ContiguousIterator in(shape, strides[0], ndim - 3);
        ContiguousIterator out(shape, strides[1], ndim - 3);
        in.seek(begin);
        out.seek(begin);
        for (int64_t i = begin; i < end; ++i) {
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
inline void copy_general_general(const array& src, array& dst) {
  copy_general_general<SrcT, DstT>(
      src,
      dst,
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
      data_shape,
      i_strides,
      make_contiguous_strides(data_shape),
      i_offset,
      o_offset,
      dynamic_i_offset,
      dynamic_o_offset);
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
      0,
      std::nullopt,
      std::nullopt);
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
  dispatch_all_types(dst.dtype(), [&](auto type_tag) {
    using DstT = MLX_GET_TYPE(type_tag);
    copy<SrcT, DstT>(src, dst, ctype, std::forward<Args>(args)...);
  });
}

template <typename... Args>
inline void copy_inplace_dispatch(
    const array& src,
    array& dst,
    CopyType ctype,
    Args&&... args) {
  dispatch_all_types(src.dtype(), [&](auto type_tag) {
    using SrcT = MLX_GET_TYPE(type_tag);
    copy<SrcT>(src, dst, ctype, std::forward<Args>(args)...);
  });
}

} // namespace

void copy_cpu_inplace(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream) {
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  encoder.dispatch(
      [src = array::unsafe_weak_copy(src),
       dst = array::unsafe_weak_copy(dst),
       ctype]() mutable { copy_inplace_dispatch(src, dst, ctype); });
}

void copy_cpu(const array& src, array& dst, CopyType ctype, Stream stream) {
  bool donated = set_copy_output_data(src, dst, ctype);
  if (donated && src.dtype() == dst.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_cpu_inplace(src, dst, ctype, stream);
}

void copy_cpu_inplace(
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
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(src);
  encoder.set_output_array(dst);
  auto weak_copy_if_set = [](auto x) -> std::optional<array> {
    if (x) {
      return array::unsafe_weak_copy(*x);
    } else {
      return std::nullopt;
    }
  };
  encoder.dispatch(
      [src = array::unsafe_weak_copy(src),
       dst = array::unsafe_weak_copy(dst),
       data_shape,
       i_strides,
       o_strides,
       i_offset,
       o_offset,
       ctype,
       dynamic_i_offset = weak_copy_if_set(dynamic_i_offset),
       dynamic_o_offset = weak_copy_if_set(dynamic_o_offset)]() mutable {
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
                o_offset,
                dynamic_i_offset,
                dynamic_o_offset);
            break;
          case CopyType::Scalar:
          case CopyType::Vector:
            copy_inplace_dispatch(src, dst, ctype);
        }
      });
}

array contiguous_copy_cpu(const array& arr, Stream stream) {
  array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
  copy_cpu(arr, arr_copy, CopyType::General, stream);
  return arr_copy;
}

} // namespace mlx::core
