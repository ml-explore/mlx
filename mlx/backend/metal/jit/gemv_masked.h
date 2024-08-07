// Copyright © 2024 Apple Inc.

constexpr std::string_view gemv_masked_kernel = R"(
template [[host_name("{name}")]] [[kernel]] void
gemv_{trans}masked<{itype}, {outm_t}, {opm_t}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, {nc}>(
    const device {itype}* mat [[buffer(0)]],
    const device {itype}* in_vec [[buffer(1)]],
    device {itype}* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& marix_ld [[buffer(6)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant size_t* vector_batch_stride [[buffer(11)]],
    const constant size_t* matrix_batch_stride [[buffer(12)]],
    const device {outm_t}* out_mask [[buffer(20)]],
    const device {opm_t}* mat_mask [[buffer(21)]],
    const device {opm_t}* vec_mask [[buffer(22)]],
    const constant int* mask_strides [[buffer(23)]],
    const constant size_t* mask_batch_strides [[buffer(24)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);
)";
