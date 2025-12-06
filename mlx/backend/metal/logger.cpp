// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/logger.h"

#ifdef MLX_METAL_LOG_ENABLED

#include <os/log.h>

#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::metal {
struct LogCapture {
  std::mutex mutex;
  bool has_logs{false};
  std::vector<std::string> messages;
};

namespace {

constexpr NS::Integer kDefaultLogBufferSize = 8 * 1024;
constexpr MTL::LogLevel kDefaultLogLevel = MTL::LogLevelDebug;

std::string to_string(NS::String* value) {
  return value ? std::string(value->utf8String()) : std::string();
}

std::string level_to_string(MTL::LogLevel level) {
  switch (level) {
    case MTL::LogLevelDebug:
      return "debug";
    case MTL::LogLevelInfo:
      return "info";
    case MTL::LogLevelNotice:
      return "notice";
    case MTL::LogLevelError:
      return "error";
    case MTL::LogLevelFault:
      return "fault";
    default:
      return "unknown";
  }
}

} // namespace

MetalLogger::MetalLogger() {
  descriptor_ = MTL::LogStateDescriptor::alloc()->init();
  descriptor_->setLevel(kDefaultLogLevel);
  descriptor_->setBufferSize(kDefaultLogBufferSize);
}

MetalLogger::~MetalLogger() {
  if (descriptor_) {
    descriptor_->release();
  }
}

MTL::LogState* MetalLogger::make_log_state(MTL::Device* device) {
  if (!device || !descriptor_) {
    return nullptr;
  }

  NS::Error* error = nullptr;
  auto log_state = device->newLogState(descriptor_, &error);
  if (log_state == nullptr) {
    return nullptr;
  }

  return log_state;
}

void MetalLogger::attach(MTL::CommandBuffer* buffer, MTL::LogState* log_state) {
  if (buffer == nullptr || log_state == nullptr) {
    return;
  }

  auto capture = std::make_shared<LogCapture>();
  {
    std::lock_guard<std::mutex> lock(captures_mutex_);
    captures_.erase(buffer);
    captures_.emplace(buffer, capture);
  }

  log_state->addLogHandler(^void(
      NS::String* subsystem,
      NS::String* category,
      MTL::LogLevel level,
      NS::String* message) {
    std::lock_guard<std::mutex> lock(capture->mutex);
    capture->has_logs = true;

    std::ostringstream os;
    auto subsystem_str = to_string(subsystem);
    auto category_str = to_string(category);
    if (!subsystem_str.empty()) {
      os << subsystem_str;
    }
    if (!category_str.empty()) {
      if (!subsystem_str.empty()) {
        os << "::";
      }
      os << category_str;
    }
    if (!subsystem_str.empty() || !category_str.empty()) {
      os << ": ";
    }
    os << "[" << level_to_string(level) << "] ";
    os << to_string(message);

    capture->messages.push_back(os.str());
  });
}

MTL::CommandBuffer* MetalLogger::make_buffer_with_logging(
    MTL::CommandQueue* queue,
    MTL::Device* device) {
  if (!queue || !device) {
    return nullptr;
  }

  auto log_state = make_log_state(device);
  if (log_state == nullptr) {
    return nullptr;
  }

  auto desc = MTL::CommandBufferDescriptor::alloc()->init();
  desc->setRetainedReferences(false);
  desc->setLogState(log_state);
  auto buffer = queue->commandBuffer(desc);
  desc->release();

  attach(buffer, log_state);

  log_state->release();
  return buffer;
}

void MetalLogger::register_completion(MTL::CommandBuffer* buffer) {
  if (buffer == nullptr) {
    return;
  }

  std::shared_ptr<LogCapture> capture;
  {
    std::lock_guard<std::mutex> lock(captures_mutex_);
    auto it = captures_.find(buffer);
    if (it == captures_.end()) {
      return;
    }
    capture = std::move(it->second);
    captures_.erase(it);
  }

  buffer->addCompletedHandler([capture](MTL::CommandBuffer*) {
    std::vector<std::string> messages;
    {
      std::lock_guard<std::mutex> lock(capture->mutex);
      if (!capture->has_logs) {
        return;
      }
      messages = std::move(capture->messages);
    }

    std::ostringstream os;
    os << "[metal::logger] Shader log messages detected";
    for (const auto& message : messages) {
      os << "\n- " << message;
    }

    auto payload = os.str();
    throw std::runtime_error(payload);
  });
}

} // namespace mlx::core::metal

#endif // MLX_METAL_LOG_ENABLED
