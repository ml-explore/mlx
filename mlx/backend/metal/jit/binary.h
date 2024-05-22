// Copyright © 2024 Apple Inc.

constexpr std::string_view binary_kernels = R"(
template [[host_name("ss{0}")]] [[kernel]]
void binary_ss<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    uint index [[thread_position_in_grid]]);
template [[host_name("vs{0}")]] [[kernel]]
void binary_vs<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    uint index [[thread_position_in_grid]]);
template [[host_name("sv{0}")]] [[kernel]]
void binary_sv<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    uint index [[thread_position_in_grid]]);
template [[host_name("vv{0}")]] [[kernel]]
void binary_vv<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    uint index [[thread_position_in_grid]]);
template [[host_name("g4{0}")]] [[kernel]] void
binary_g_nd<{1}, {2}, {3}, 4>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int shape[4],
    constant const size_t a_strides[4],
    constant const size_t b_strides[4],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
template [[host_name("g5{0}")]] [[kernel]] void
binary_g_nd<{1}, {2}, {3}, 5>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int shape[5],
    constant const size_t a_strides[5],
    constant const size_t b_strides[5],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);

template [[host_name("g1{0}")]] [[kernel]] void
binary_g_nd1<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]);
template [[host_name("g2{0}")]] [[kernel]] void
binary_g_nd2<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]);
template [[host_name("g3{0}")]] [[kernel]] void
binary_g_nd3<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);

template [[host_name("gn{0}")]] [[kernel]]
void binary_g<{1}, {2}, {3}>(
    device const {1}* a,
    device const {1}* b,
    device {2}* c,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]);
)";
