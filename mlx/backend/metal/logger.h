// Copyright © 2025 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>

#ifdef MLX_METAL_LOG_ENABLED
#include <memory>
#include <mutex>
#include <unordered_map>
#endif

namespace mlx::core::metal {

#ifdef MLX_METAL_LOG_ENABLED
struct LogCapture;

class MetalLogger {
 public:
  MetalLogger();
  MetalLogger(const MetalLogger&) = delete;
  MetalLogger& operator=(const MetalLogger&) = delete;
  ~MetalLogger();

  MTL::LogState* make_log_state(MTL::Device* device);
  void attach(MTL::CommandBuffer* buffer, MTL::LogState* log_state);
  MTL::CommandBuffer* make_buffer_with_logging(
      MTL::CommandQueue* queue,
      MTL::Device* device);
  void register_completion(MTL::CommandBuffer* buffer);

 private:
  MTL::LogStateDescriptor* descriptor_{nullptr};
  std::mutex captures_mutex_;
  std::unordered_map<MTL::CommandBuffer*, std::shared_ptr<LogCapture>>
      captures_;
};
#else
class MetalLogger {
 public:
  explicit MetalLogger(MTL::Device*) {}
  MetalLogger(const MetalLogger&) = delete;
  MetalLogger& operator=(const MetalLogger&) = delete;
  ~MetalLogger() = default;

  MTL::LogState* make_log_state(MTL::Device*) {
    return nullptr;
  }
  void attach(MTL::CommandBuffer*, MTL::LogState*) {}
  MTL::CommandBuffer* make_buffer_with_logging(
      MTL::CommandQueue*,
      MTL::Device*) {
    return nullptr;
  }
  void register_completion(MTL::CommandBuffer*) {}
};
#endif

} // namespace mlx::core::metal
