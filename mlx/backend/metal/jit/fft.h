// Copyright © 2024 Apple Inc.

constexpr std::string_view fft_kernel = R"(
template [[host_name("{name}")]] [[kernel]] void
fft<{tg_mem_size}, {in_T}, {out_T}>(
    const device {in_T}* in [[buffer(0)]],
    device {out_T}* out [[buffer(1)]],
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]);
)";

constexpr std::string_view rader_fft_kernel = R"(
template [[host_name("{name}")]] [[kernel]] void
rader_fft<{tg_mem_size}, {in_T}, {out_T}>(
    const device {in_T}* in [[buffer(0)]],
    device {out_T}* out [[buffer(1)]],
    const device float2* raders_b_q [[buffer(2)]],
    const device short* raders_g_q [[buffer(3)]],
    const device short* raders_g_minus_q [[buffer(4)]],
    constant const int& n,
    constant const int& batch_size,
    constant const int& rader_n,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]);
)";

constexpr std::string_view bluestein_fft_kernel = R"(
template [[host_name("{name}")]] [[kernel]] void
bluestein_fft<{tg_mem_size}, {in_T}, {out_T}>(
    const device {in_T}* in [[buffer(0)]],
    device {out_T}* out [[buffer(1)]],
    const device float2* w_q [[buffer(2)]],
    const device float2* w_k [[buffer(3)]],
    constant const int& length,
    constant const int& n,
    constant const int& batch_size,
    uint3 elem [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]);
)";

constexpr std::string_view four_step_fft_kernel = R"(
template [[host_name("{name}")]] [[kernel]] void
four_step_fft<{tg_mem_size}, {in_T}, {out_T}, {step}, {real}>(
      const device {in_T}* in [[buffer(0)]],
      device {out_T}* out [[buffer(1)]],
      constant const int& n1,
      constant const int& n2,
      constant const int& batch_size,
      uint3 elem [[thread_position_in_grid]],
      uint3 grid [[threads_per_grid]]);
)";