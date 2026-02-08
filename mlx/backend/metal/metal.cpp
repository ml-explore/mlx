// Copyright © 2023-2024 Apple Inc.
#include <atomic>
#include <memory>

#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::metal {
namespace {
std::atomic<bool> g_residency_sets_enabled{true};
} // namespace

bool is_available() {
  return true;
}

bool residency_sets_enabled() {
  return g_residency_sets_enabled.load(std::memory_order_relaxed);
}

void set_residency_sets_enabled(bool enabled) {
  g_residency_sets_enabled.store(enabled, std::memory_order_relaxed);
}

void start_capture(std::string path, NS::Object* object) {
  auto pool = new_scoped_memory_pool();
  set_residency_sets_enabled(false);
  // Detach queue residency sets before capture starts to keep traces replayable
  // when collecting derived counters.
  metal::device(mlx::core::Device::gpu).on_capture_start();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  descriptor->release();
  if (!started) {
    set_residency_sets_enabled(true);
    metal::device(mlx::core::Device::gpu).on_capture_stop();
    metal::allocator().on_capture_stop();
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
  set_residency_sets_enabled(true);
  metal::device(mlx::core::Device::gpu).on_capture_stop();
  metal::allocator().on_capture_stop();
}

bool is_capture_active() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  return manager->isCapturing();
}

} // namespace mlx::core::metal
