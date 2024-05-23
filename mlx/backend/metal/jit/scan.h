// Copyright © 2024 Apple Inc.

constexpr std::string_view scan_kernels = R"(
template [[host_name("contig_{0}")]] [[kernel]] void
contiguous_scan<{1}, {2}, {3}<{2}>, 4, {4}, {5}>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]);

template [[host_name("strided_{0}")]] [[kernel]] void
strided_scan<{1}, {2}, {3}<{2}>, 4, {4}, {5}>(
    const device {1}* in [[buffer(0)]],
    device {2}* out [[buffer(1)]],
    const constant size_t& axis_size [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]]);
)";
