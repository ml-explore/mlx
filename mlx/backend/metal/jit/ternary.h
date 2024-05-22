// Copyright © 2024 Apple Inc.

constexpr std::string_view ternary_kernels = R"(
template [[host_name("v_{0}")]] [[kernel]] void ternary_v<{1}, {2}>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    uint index [[thread_position_in_grid]]);

template [[host_name("g_{0}")]] [[kernel]] void ternary_g<{1}, {2}>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const size_t* c_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);

template [[host_name("g1_{0}")]] [[kernel]] void
ternary_g_nd1<{1}, {2}>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t& a_strides,
    constant const size_t& b_strides,
    constant const size_t& c_strides,
    uint index [[thread_position_in_grid]]);
template [[host_name("g2_{0}")]] [[kernel]] void
ternary_g_nd2<{1}, {2}>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    constant const size_t c_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]);
template [[host_name("g3_{0}")]] [[kernel]] void
ternary_g_nd3<{1}, {2}>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    constant const size_t c_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("g4_{0}")]] [[kernel]] void
ternary_g_nd<{1}, {2}, 4>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int shape[4],
    constant const size_t a_strides[4],
    constant const size_t b_strides[4],
    constant const size_t c_strides[4],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("g5_{0}")]] [[kernel]] void
ternary_g_nd<{1}, {2}, 5>(
    device const bool* a,
    device const {1}* b,
    device const {1}* c,
    device {1}* d,
    constant const int shape[5],
    constant const size_t a_strides[5],
    constant const size_t b_strides[5],
    constant const size_t c_strides[5],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
)";
