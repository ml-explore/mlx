// Copyright © 2024 Apple Inc.

constexpr std::string_view steel_conv_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
implicit_gemm_conv_2d<{itype}, {bm}, {bn}, {bk}, {wm}, {wn}, {n_channels}, {small_filter}>(
    const device {itype}* A [[buffer(0)]],
    const device {itype}* B [[buffer(1)]],
    device {itype}* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);
)";

constexpr std::string_view steel_conv_general_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
    implicit_gemm_conv_2d_general<{itype}, {bm}, {bn}, {bk}, {wm}, {wn}>(
        const device {itype}* A [[buffer(0)]],
        const device {itype}* B [[buffer(1)]],
        device {itype}* C [[buffer(2)]],
        const constant MLXConvParams<2>* params [[buffer(3)]],
        const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],
        const constant Conv2DGeneralJumpParams* jump_params [[buffer(5)]],
        const constant Conv2DGeneralBaseInfo* base_h [[buffer(6)]],
        const constant Conv2DGeneralBaseInfo* base_w [[buffer(7)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint3 lid [[thread_position_in_threadgroup]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]);
)";
