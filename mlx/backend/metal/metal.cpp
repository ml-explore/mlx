// Copyright © 2023-2024 Apple Inc.
#include <atomic>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace mlx::core::metal {

namespace {

std::string g_metallib_path;

int initial_batch_invariant_limit() {
  int limit = env::get_var("MLX_METAL_BATCH_INVARIANT_LIMIT", 0);
  return limit > 0 ? limit : 0;
}

std::atomic<int> g_batch_invariant_limit{initial_batch_invariant_limit()};

} // namespace

bool is_available() {
  return true;
}

void set_batch_invariant_limit(int limit) {
  g_batch_invariant_limit.store(
      limit > 0 ? limit : 0, std::memory_order_relaxed);
}

int get_batch_invariant_limit() {
  return g_batch_invariant_limit.load(std::memory_order_relaxed);
}

void start_capture(std::string path, NS::Object* object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init()->autorelease();
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
  if (!started) {
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
}

void set_metallib_path(const std::string& path) {
  g_metallib_path = path;
}

const std::string& get_metallib_path() {
  return g_metallib_path;
}

} // namespace mlx::core::metal
