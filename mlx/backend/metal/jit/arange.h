// Copyright © 2024 Apple Inc.

constexpr std::string_view arange_kernels = R"(
template [[host_name("{0}")]] [[kernel]] void arange<{1}>(
    constant const {1}& start,
    constant const {1}& step,
    device {1}* out,
    uint index [[thread_position_in_grid]]);
)";
