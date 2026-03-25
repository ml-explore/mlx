// Copyright © 2026 Apple Inc.

#include "mlx/stream.h"
#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/scheduler.h"

#include <array>
#include <optional>
#include <shared_mutex>

namespace mlx::core {

namespace {

auto& default_stream_storage(Device d) {
  // Each device has its own default stream in each thread.
  static thread_local auto default_streams = []() {
    std::array<std::vector<std::optional<Stream>>, 2> streams;
    streams[static_cast<size_t>(Device::cpu)].resize(cpu::device_count());
    streams[static_cast<size_t>(Device::gpu)].resize(gpu::device_count());
    return streams;
  }();
  return default_streams[static_cast<size_t>(d.type)].at(d.index);
}

auto& all_streams() {
  static std::tuple<std::vector<Stream>, std::shared_mutex> streams_and_mtx;
  return streams_and_mtx;
}

} // namespace

Stream default_stream(Device d) {
  if (!gpu::is_available() && d.type == Device::gpu) {
    throw std::invalid_argument(
        "[default_stream] Cannot get gpu stream without gpu backend.");
  }
  auto& s = default_stream_storage(d);
  if (!s.has_value()) {
    s = new_stream(d.type);
  }
  return s.value();
}

void set_default_stream(Stream s) {
  if (!gpu::is_available() && s.device == Device::gpu) {
    throw std::invalid_argument(
        "[set_default_stream] Cannot set gpu stream without gpu backend.");
  }
  default_stream_storage(s.device) = s;
}

std::vector<Stream> get_streams() {
  auto& [streams, mtx] = all_streams();
  std::shared_lock lock(mtx);
  return streams;
}

Stream new_stream(Device d) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[new_stream] Cannot make gpu stream without gpu backend.");
  }
  auto& [streams, mtx] = all_streams();
  std::unique_lock lock(mtx);
  int index = streams.size();
  auto& s = streams.emplace_back(index, d);
  scheduler::scheduler().new_thread(d.type);
  if (d == Device::gpu) {
    gpu::new_stream(s);
  }
  return s;
}

} // namespace mlx::core
