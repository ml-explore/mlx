// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include <sys/sysctl.h>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

void start_capture(std::string path, id object) {
  auto pool = new_scoped_memory_pool();

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

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  auto init_device_info = []()
      -> std::unordered_map<std::string, std::variant<std::string, size_t>> {
    auto pool = new_scoped_memory_pool();
    auto raw_device = device(default_device()).mtl_device();
    auto name = std::string(raw_device->name()->utf8String());
    auto arch = std::string(raw_device->architecture()->name()->utf8String());

    size_t memsize = 0;
    size_t length = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &length, NULL, 0);

    size_t rsrc_limit = 0;
    sysctlbyname("iogpu.rsrc_limit", &rsrc_limit, &length, NULL, 0);
    if (rsrc_limit == 0) {
      rsrc_limit = 499000;
    }

    return {
        {"device_name", name},
        {"architecture", arch},
        {"max_buffer_length", raw_device->maxBufferLength()},
        {"max_recommended_working_set_size",
         raw_device->recommendedMaxWorkingSetSize()},
        {"memory_size", memsize},
        {"resource_limit", rsrc_limit}};
  };
  static auto device_info_ = init_device_info();
  return device_info_;
}

} // namespace mlx::core::metal
