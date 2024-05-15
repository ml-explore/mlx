// Copyright © 2024 Apple Inc.

constexpr std::string_view copy_kernels = R"(
template [[host_name("s_{0}")]] [[kernel]] void copy_s<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    uint index [[thread_position_in_grid]]);
template [[host_name("v_{0}")]] [[kernel]] void copy_v<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    uint index [[thread_position_in_grid]]);

template [[host_name("g4_{0}")]] [[kernel]] void
copy_g_nd<{1}, {2}, 4>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("gg4_{0}")]] [[kernel]] void
copy_gg_nd<{1}, {2}, 4>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]);
template [[host_name("g5_{0}")]] [[kernel]] void
copy_g_nd<{1}, {2}, 5>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("gg5_{0}")]] [[kernel]] void
copy_gg_nd<{1}, {2}, 5>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]);
template [[host_name("g1_{0}")]] [[kernel]] void copy_g_nd1<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    uint index [[thread_position_in_grid]]);
template [[host_name("g2_{0}")]] [[kernel]] void copy_g_nd2<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]);
template [[host_name("g3_{0}")]] [[kernel]] void copy_g_nd3<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("gg1_{0}")]] [[kernel]] void
copy_gg_nd1<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    constant const int64_t& dst_stride [[buffer(4)]],
    uint index [[thread_position_in_grid]]);
template [[host_name("gg2_{0}")]] [[kernel]] void
copy_gg_nd2<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]);
template [[host_name("gg3_{0}")]] [[kernel]] void
copy_gg_nd3<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]);

template [[host_name("g_{0}")]] [[kernel]] void copy_g<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("gg_{0}")]] [[kernel]] void copy_gg<{1}, {2}>(
    device const {1}* src [[buffer(0)]],
    device {2}* dst [[buffer(1)]],
    constant const int* src_shape [[buffer(2)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    constant const int& ndim [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]);
)";
