// Copyright © 2024 Apple Inc.

constexpr std::string_view steel_gemm_fused_kernels = R"(
template [[host_name("{name}")]]
[[kernel]] void gemm<{itype}, {bm}, {bn}, {bk}, {wm}, {wn}, {trans_a}, {trans_b}, float>(
    const device {itype} *A [[buffer(0)]],
    const device {itype} *B [[buffer(1)]],
    const device {itype} *C [[buffer(2), function_constant(use_out_source)]],
    device {itype} *D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(5), function_constant(use_out_source)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    const constant uint32_t* lhs_indices [[buffer(10), function_constant(do_gather)]],
    const constant uint32_t* rhs_indices [[buffer(11), function_constant(do_gather)]],
    const constant uint32_t* C_indices [[buffer(12), function_constant(gather_bias)]],
    const constant int* operand_shape [[buffer(13), function_constant(do_gather)]],
    const constant size_t* operand_strides [[buffer(14), function_constant(do_gather)]],
    const constant packed_int3& operand_batch_ndim [[buffer(15), function_constant(do_gather)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
)";

constexpr std::string_view steel_gemm_masked_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
block_masked_gemm<
    {itype},
    {outmasktype},
    {opmasktype},
    {bm},
    {bn},
    {bk},
    {wm},
    {wn},
    {trans_a},
    {trans_b},
    {mn_aligned},
    {k_aligned}>(
    const device {itype}* A [[buffer(0)]],
    const device {itype}* B [[buffer(1)]],
    device {itype}* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int* batch_shape [[buffer(6)]],
    const constant size_t* batch_strides [[buffer(7)]],
    const device {outmasktype}* out_mask [[buffer(10)]],
    const device {opmasktype}* lhs_mask [[buffer(11)]],
    const device {opmasktype}* rhs_mask [[buffer(12)]],
    const constant int* mask_strides [[buffer(13)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
)";

constexpr std::string_view steel_gemm_splitk_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
gemm_splitk<
    {itype},
    {otype},
    {bm},
    {bn},
    {bk},
    {wm},
    {wn},
    {trans_a},
    {trans_b},
    {mn_aligned},
    {k_aligned}>(
    const device {itype}* A [[buffer(0)]],
    const device {itype}* B [[buffer(1)]],
    device {otype}* C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]);
)";

constexpr std::string_view steel_gemm_splitk_accum_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
gemm_splitk_accum<{atype}, {otype}>(
    const device {atype}* C_split [[buffer(0)]],
    device {otype}* D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]);
)";

constexpr std::string_view steel_gemm_splitk_accum_axbpy_kernels = R"(
template [[host_name("{name}")]] [[kernel]] void
gemm_splitk_accum_axpby<{atype}, {otype}>(
    const device {atype}* C_split [[buffer(0)]],
    device {otype}* D [[buffer(1)]],
    const constant int& k_partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    const device {otype}* C [[buffer(5)]],
    const constant int& ldc [[buffer(6)]],
    const constant int& fdc [[buffer(7)]],
    const constant float& alpha [[buffer(8)]],
    const constant float& beta [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]);
)";
