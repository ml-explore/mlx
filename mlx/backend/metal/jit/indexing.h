// Copyright © 2023-2024 Apple Inc.

constexpr std::string_view gather_kernels = R"(
[[kernel]] void gather{0}_{3}_{6}(
    const device {1}* src [[buffer(0)]],
    device {1}* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant size_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const constant int* idx_shapes [[buffer(7)]],
    const constant size_t* idx_strides [[buffer(8)]],
    const constant int& idx_ndim [[buffer(9)]],
    {4}
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {{
  Indices<{2}, {3}> idxs{{
    {{ {5} }}, idx_shapes, idx_strides, idx_ndim}};

  return gather_impl<{1}, {2}, {3}, {6}>(
      src,
      out,
      src_shape,
      src_strides,
      src_ndim,
      slice_sizes,
      axes,
      idxs,
      index,
      grid_dim);
}}
)";

constexpr std::string_view scatter_kernels = R"(
[[kernel]] void scatter_1d_index{0}_{4}(
    const device {1}* updates [[buffer(1)]],
    device mlx_atomic<{1}>* out [[buffer(2)]],
    const constant int* out_shape [[buffer(3)]],
    const constant size_t* out_strides [[buffer(4)]],
    const constant size_t& upd_size [[buffer(5)]],
    {5}
    uint2 gid [[thread_position_in_grid]]) {{
  const array<const device {2}*, {4}> idx_buffers = {{ {6} }};
  return scatter_1d_index_impl<{1}, {2}, {3}, {4}>(
      updates, out, out_shape, out_strides, upd_size, idx_buffers, gid);
}}

[[kernel]] void scatter{0}_{4}(
    const device {1}* updates [[buffer(1)]],
    device mlx_atomic<{1}>* out [[buffer(2)]],
    const constant int* upd_shape [[buffer(3)]],
    const constant size_t* upd_strides [[buffer(4)]],
    const constant size_t& upd_ndim [[buffer(5)]],
    const constant size_t& upd_size [[buffer(6)]],
    const constant int* out_shape [[buffer(7)]],
    const constant size_t* out_strides [[buffer(8)]],
    const constant size_t& out_ndim [[buffer(9)]],
    const constant int* axes [[buffer(10)]],
    const constant int* idx_shapes [[buffer(11)]],
    const constant size_t* idx_strides [[buffer(12)]],
    const constant int& idx_ndim [[buffer(13)]],
    {5}
    uint2 gid [[thread_position_in_grid]]) {{
  Indices<{2}, {4}> idxs{{ {{ {6} }}, idx_shapes, idx_strides, idx_ndim}};

  return scatter_impl<{1}, {2}, {3}, {4}>(
      updates,
      out,
      upd_shape,
      upd_strides,
      upd_ndim,
      upd_size,
      out_shape,
      out_strides,
      out_ndim,
      axes,
      idxs,
      gid);
}}
)";
