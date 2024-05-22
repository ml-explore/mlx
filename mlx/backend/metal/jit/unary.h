// Copyright © 2024 Apple Inc.

constexpr std::string_view unary_kernels = R"(
template [[host_name("v{0}")]] [[kernel]] void unary_v<{1}, {2}>(
    device const {1}* in,
    device {1}* out,
    uint index [[thread_position_in_grid]]);

template [[host_name("g{0}")]] [[kernel]] void unary_g<{1}, {2}>(
    device const {1}* in,
    device {1}* out,
    device const int* in_shape,
    device const size_t* in_strides,
    device const int& ndim,
    uint index [[thread_position_in_grid]]);
)";
