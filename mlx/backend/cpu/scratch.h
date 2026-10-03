// Copyright © 2026 Apple Inc.

#pragma once

#include <cstddef>
#include <vector>

namespace mlx::core::cpu {

// Thread-local float scratch for dequantize/convert staging in the matmul
// paths. Retention policy: buffers up to kMaxRetainedFloats are kept for
// reuse across calls (the same layer shapes recur every prefill); larger
// buffers are kept only while requests match their capacity and released
// as soon as a smaller request arrives, so a one-off huge GEMM does not
// pin gigabytes on the stream thread for the rest of the process.
inline float* scratch_buffer(size_t n_floats, int slot = 0) {
  thread_local std::vector<float> bufs[2];
  constexpr size_t kMaxRetainedFloats = size_t{64} << 20; // 256 MB
  auto& buf = bufs[slot == 0 ? 0 : 1];
  if (buf.capacity() > kMaxRetainedFloats && n_floats < buf.capacity()) {
    std::vector<float>().swap(buf);
  }
  if (buf.size() < n_floats) {
    buf.resize(n_floats);
  }
  return buf.data();
}

} // namespace mlx::core::cpu
