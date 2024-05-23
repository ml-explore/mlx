// Copyright © 2024 Apple Inc.

constexpr std::string_view reduce_init_kernels = R"(
[[kernel]] void {0}(
    device {1}* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {{
  out[tid] = {2}<{1}>::init;
}}
)";

constexpr std::string_view reduce_kernels = R"(
template [[host_name("all_{0}")]] [[kernel]] void
all_reduce<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device mlx_atomic<{2}>* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
template [[host_name("colGeneral_{0}")]] [[kernel]] void
col_reduce_general<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device mlx_atomic<{2}>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup {2}* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]);
template [[host_name("colSmall_{0}")]] [[kernel]] void
col_reduce_small<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    const constant size_t& non_col_reductions [[buffer(8)]],
    const constant int* non_col_shapes [[buffer(9)]],
    const constant size_t* non_col_strides [[buffer(10)]],
    const constant int& non_col_ndim [[buffer(11)]],
    uint tid [[thread_position_in_grid]]);
template [[host_name("rowGeneralSmall_{0}")]] [[kernel]] void
row_reduce_general_small<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint lid [[thread_position_in_grid]]);
template [[host_name("rowGeneralMed_{0}")]] [[kernel]] void
row_reduce_general_med<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[dispatch_simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
template [[host_name("rowGeneral_{0}")]] [[kernel]] void
row_reduce_general<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device mlx_atomic<{2}>* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
)";

constexpr std::string_view reduce_non_atomic_kernels = R"(
template [[host_name("allNoAtomics_{0}")]] [[kernel]] void
all_reduce_no_atomics<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_group_id [[threadgroup_position_in_grid]]);

template [[host_name("colGeneralNoAtomics_{0}")]] [[kernel]] void
  col_reduce_general_no_atomics<{1}, {2}, {3}<{2}>>(
      const device {1}* in [[buffer(0)]],
      device {2}* out [[buffer(1)]],
      const constant size_t& reduction_size [[buffer(2)]],
      const constant size_t& reduction_stride [[buffer(3)]],
      const constant size_t& out_size [[buffer(4)]],
      const constant int* shape [[buffer(5)]],
      const constant size_t* strides [[buffer(6)]],
      const constant int& ndim [[buffer(7)]],
      threadgroup {2}* local_data [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint3 gid [[thread_position_in_grid]],
      uint3 lsize [[threads_per_threadgroup]],
      uint3 gsize [[threads_per_grid]]);
template [[host_name("colSmall_{0}")]] [[kernel]] void
col_reduce_small<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    const constant size_t& non_col_reductions [[buffer(8)]],
    const constant int* non_col_shapes [[buffer(9)]],
    const constant size_t* non_col_strides [[buffer(10)]],
    const constant int& non_col_ndim [[buffer(11)]],
    uint tid [[thread_position_in_grid]]);
template [[host_name("rowGeneralSmall_{0}")]] [[kernel]] void
row_reduce_general_small<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint lid [[thread_position_in_grid]]);
template [[host_name("rowGeneralNoAtomics_{0}")]] [[kernel]] void
row_reduce_general_no_atomics<{1}, {2}, {3}<{2}>>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
)";
