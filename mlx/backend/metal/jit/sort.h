// Copyright © 2024 Apple Inc.

constexpr std::string_view block_sort_kernels = R"(
template [[host_name("carg_{0}")]] [[kernel]] void
block_sort<{1}, {2}, true, {3}, {4}>(
    const device {1}* inp [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant int& size_sorted_axis [[buffer(2)]],
    const constant int& stride_sorted_axis [[buffer(3)]],
    const constant int& stride_segment_axis [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
template [[host_name("ncarg_{0}")]] [[kernel]] void
block_sort_nc<{1}, {2}, true, {3}, {4}>(
    const device {1}* inp [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant int& size_sorted_axis [[buffer(2)]],
    const constant int& stride_sorted_axis [[buffer(3)]],
    const constant int& nc_dim [[buffer(4)]],
    const device int* nc_shape [[buffer(5)]],
    const device size_t* nc_strides [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
template [[host_name("c_{0}")]] [[kernel]] void
block_sort<{1}, {2}, false, {3}, {4}>(
    const device {1}* inp [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant int& size_sorted_axis [[buffer(2)]],
    const constant int& stride_sorted_axis [[buffer(3)]],
    const constant int& stride_segment_axis [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
template [[host_name("nc_{0}")]] [[kernel]] void
block_sort_nc<{1}, {2}, false, {3}, {4}>(
    const device {1}* inp [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant int& size_sorted_axis [[buffer(2)]],
    const constant int& stride_sorted_axis [[buffer(3)]],
    const constant int& nc_dim [[buffer(4)]],
    const device int* nc_shape [[buffer(5)]],
    const device size_t* nc_strides [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
)";

constexpr std::string_view multiblock_sort_kernels = R"(
template [[host_name("sort_{0}")]] [[kernel]] void
mb_block_sort<{1}, {2}, true, {3}, {4}>(
    const device {1}* inp [[buffer(0)]],
    device {1}* out_vals [[buffer(1)]],
    device {2}* out_idxs [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    const constant int& stride_sorted_axis [[buffer(4)]],
    const constant int& nc_dim [[buffer(5)]],
    const device int* nc_shape [[buffer(6)]],
    const device size_t* nc_strides [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
template [[host_name("partition_{0}")]] [[kernel]] void
mb_block_partition<{1}, {2}, true, {3}, {4}>(
    device {2}* block_partitions [[buffer(0)]],
    const device {1}* dev_vals [[buffer(1)]],
    const device {2}* dev_idxs [[buffer(2)]],
    const constant int& size_sorted_axis [[buffer(3)]],
    const constant int& merge_tiles [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_dims [[threads_per_threadgroup]]);
template [[host_name("merge_{0}")]] [[kernel]] void
mb_block_merge<{1}, {2}, true, {3}, {4}>(
    const device {2}* block_partitions [[buffer(0)]],
    const device {1}* dev_vals_in [[buffer(1)]],
    const device {2}* dev_idxs_in [[buffer(2)]],
    device {1}* dev_vals_out [[buffer(3)]],
    device {2}* dev_idxs_out [[buffer(4)]],
    const constant int& size_sorted_axis [[buffer(5)]],
    const constant int& merge_tiles [[buffer(6)]],
    const constant int& num_tiles [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
)";
