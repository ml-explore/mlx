// Copyright © 2024 Apple Inc.

constexpr std::string_view softmax_kernels = R"(
template [[host_name("block_{0}")]] [[kernel]] void
softmax_single_row<{1}, {2}>(
    const device {1}* in,
    device {1}* out,
    constant int& axis_size,
    uint gid [[thread_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
template [[host_name("looped_{0}")]] [[kernel]] void
softmax_looped<{1}, {2}>(
    const device {1}* in,
    device {1}* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);
)";
