// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/logger.h"

#ifdef MLX_METAL_LOG_ENABLED

#include <os/log.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlx::core::metal {

struct MetalLogger::LogCapture {
  std::mutex mutex;
  bool has_error{false};
  bool has_fault{false};
  std::vector<std::string> messages;
};

namespace {

std::string ns_string_to_std(NS::String* value) {
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

MetalLogger::MetalLogger(MetalLoggerConfig config)
    : config_(config), descriptor_(nullptr) {
  descriptor_ = MTL::LogStateDescriptor::alloc()->init();
  if (descriptor_) {
    descriptor_->setLevel(config_.level);
    descriptor_->setBufferSize(config_.buffer_size);
  }
}

MetalLogger::~MetalLogger() {
  if (descriptor_) {
    descriptor_->release();
  }
};

MTL::LogState* MetalLogger::make_log_state(MTL::Device* device) {
  if (!device || !descriptor_) {
    return nullptr;
  }

  NS::Error* error = nullptr;
  auto* log_state = device->newLogState(descriptor_, &error);
  if (!log_state) {
    return nullptr;
  }

  return log_state;
}

void MetalLogger::attach(MTL::CommandBuffer& buffer, MTL::LogState& log_state) {
  auto capture = std::make_shared<LogCapture>();
  {
    std::lock_guard<std::mutex> lock(captures_mutex_);
    captures_[&buffer] = capture;
  }

  log_state.addLogHandler(^void(
      NS::String* subsystem,
      NS::String* category,
      MTL::LogLevel level,
      NS::String* message) {
    std::lock_guard<std::mutex> lock(capture->mutex);

    std::ostringstream os;
    auto subsystem_str = ns_string_to_std(subsystem);
    auto category_str = ns_string_to_std(category);

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
    os << ns_string_to_std(message);

    auto formatted = os.str();
    if (level == MTL::LogLevelFault) {
      capture->has_fault = true;
    } else if (level == MTL::LogLevelError) {
      capture->has_error = true;
    }

    capture->messages.push_back(std::move(formatted));
  });
}

MTL::CommandBuffer* MetalLogger::create_logged_command_buffer(
    MTL::CommandQueue* queue) {
  assert(queue);

  auto* log_state = make_log_state(queue->device());

  if (!log_state) {
    return queue->commandBuffer();
  }

  auto* desc = MTL::CommandBufferDescriptor::alloc()->init();
  if (!desc) {
    log_state->release();
    return queue->commandBuffer();
  }

  desc->setRetainedReferences(false);
  desc->setLogState(log_state);

  MTL::CommandBuffer* buffer = queue->commandBuffer(desc);
  desc->release();

  attach(*buffer, *log_state);
  log_state->release();
  return buffer;
}

void MetalLogger::install_completion_handler(MTL::CommandBuffer* buffer) {
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
    bool has_error = false;
    bool has_fault = false;

    {
      std::lock_guard<std::mutex> lock(capture->mutex);
      has_error = capture->has_error;
      has_fault = capture->has_fault;
      messages = std::move(capture->messages);
    }

    if (!has_error && !has_fault) {
      return;
    }

    std::ostringstream os;
    os << "[metal::logger] Shader error log messages detected";
    for (const auto& message : messages) {
      os << "\n- " << message;
    }

    throw std::runtime_error(os.str());
  });
}

} // namespace mlx::core::metal

#else // !MLX_METAL_LOG_ENABLED

namespace mlx::core::metal {

MetalLogger::MetalLogger(MetalLoggerConfig) {}

MetalLogger::~MetalLogger() = default;

MTL::CommandBuffer* MetalLogger::create_logged_command_buffer(
    MTL::CommandQueue& queue) {
  return queue.commandBuffer();
}

void MetalLogger::install_completion_handler(MTL::CommandBuffer*) {}
} // namespace mlx::core::metal

#endif // MLX_METAL_LOG_ENABLED
