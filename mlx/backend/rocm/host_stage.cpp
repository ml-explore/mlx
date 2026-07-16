// Copyright © 2024 Apple Inc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "mlx/array.h"
#include "mlx/io/load.h"

namespace mlx::core::rocm {

namespace {

// One pinned bounce buffer, reused across saves. 64 MB keeps the DMA engine
// saturated (measured 36.9 GB/s pinned vs 4.0 GB/s pageable on MI300X) while
// staying small enough to never contend with training for host RAM.
constexpr size_t kChunk = 64ull << 20;

std::mutex g_pin_mu;
char* g_pin = nullptr;

// Returns the shared pinned buffer, allocating on first use. Null on failure.
char* pin_buffer() {
  if (g_pin == nullptr) {
    void* p = nullptr;
    if (hipHostMalloc(&p, kChunk, hipHostMallocDefault) != hipSuccess) {
      (void)hipGetLastError();
      return nullptr;
    }
    g_pin = static_cast<char*>(p);
  }
  return g_pin;
}

} // namespace

// Stage an array's bytes to the host through the DMA engine instead of letting
// the CPU dereference them.
//
// The ROCm allocator hands back real VRAM (hipMalloc). On MI300X that VRAM is
// mapped into the host address space through a large PCIe BAR, so
// `arr.data<char>()` is a *valid* host pointer -- reading it just runs at
// uncached-MMIO speed. Measured on gfx942 (xnack-):
//
//   CPU direct BAR read     :    54 MB/s   <- what ostream->write() did
//   hipMemcpy DeviceToHost  :  10.4 GB/s
//   hipMemcpy Default       :  50.4 GB/s
//
// A 3 GB model save took 102 s and a 12.1 GB optimizer save 408 s (~8.5 min per
// checkpoint, ~31% of wall-clock at save_every=1000). Chunked through a pinned
// bounce buffer the same write is disk-bound (~1.7 GB/s) instead of BAR-bound.
//
// Contract: returns false ONLY before emitting any bytes (non-device pointer,
// no pinned buffer, sync failure) so the caller can safely write directly.
// Once the first chunk is written we are committed -- a later failure throws
// rather than return false, because falling back at that point would append a
// second copy of the array and silently corrupt the file.
bool staged_write(io::Writer& out, const array& a) {
  const char* src = a.data<char>();
  const size_t n = a.nbytes();
  if (src == nullptr || n == 0) {
    return false;
  }

  // Only stage true device allocations; host/pinned pointers are already fast.
  hipPointerAttribute_t attr{};
  if (hipPointerGetAttributes(&attr, static_cast<const void*>(src)) !=
      hipSuccess) {
    (void)hipGetLastError(); // not a HIP pointer: let the caller write it
    return false;
  }
  if (attr.type != hipMemoryTypeDevice) {
    return false;
  }

  // The caller eval()s before saving, but that only guarantees MLX's own
  // stream is drained. Sync the device so the DMA cannot observe a half-written
  // buffer: MLX ROCm streams are hipStreamNonBlocking, so a plain hipMemcpy on
  // the NULL stream does NOT order against them.
  if (hipDeviceSynchronize() != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }

  std::lock_guard<std::mutex> lk(g_pin_mu);
  char* pin = pin_buffer();
  if (pin == nullptr) {
    return false; // no pinned memory: fall back to the direct write
  }

  // Committed from here on: every early return above happened before any
  // out.write(), and every failure below throws.
  for (size_t off = 0; off < n; off += kChunk) {
    const size_t c = std::min(kChunk, n - off);
    // hipMemcpyDefault infers direction from pointer attributes and picks the
    // fastest engine (measured 50.4 GB/s vs 10.4 GB/s for an explicit D2H).
    hipError_t err = hipMemcpy(pin, src + off, c, hipMemcpyDefault);
    if (err != hipSuccess) {
      std::ostringstream oss;
      oss << "staged_write: hipMemcpy D2H failed at offset " << off << " of "
          << n << " (" << hipGetErrorString(err)
          << "). Checkpoint is incomplete; not falling back (would duplicate "
             "already-written bytes).";
      throw std::runtime_error(oss.str());
    }
    out.write(pin, c);
  }
  return true;
}

} // namespace mlx::core::rocm
