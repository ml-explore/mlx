// Copyright © 2025 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace mlx::core::metal {

struct MetalLoggerConfig {
  NS::Integer buffer_size = 8 * 1024;
  MTL::LogLevel level = MTL::LogLevelDebug;
};

class MetalLogger {
 public:
  MetalLogger(MetalLoggerConfig config = {});
  MetalLogger(const MetalLogger&) = delete;
  MetalLogger& operator=(const MetalLogger&) = delete;
  ~MetalLogger();

  MTL::CommandBuffer* create_logged_command_buffer(MTL::CommandQueue* queue);

  void install_completion_handler(MTL::CommandBuffer* buffer);

 private:
#ifdef MLX_METAL_LOG_ENABLED
  struct LogCapture;
  MTL::LogState* make_log_state(MTL::Device* device);
  void attach(MTL::CommandBuffer& buffer, MTL::LogState& log_state);

  MetalLoggerConfig config_;
  MTL::LogStateDescriptor* descriptor_;
  std::mutex captures_mutex_;
  std::unordered_map<MTL::CommandBuffer*, std::shared_ptr<LogCapture>>
      captures_;
#endif
};

} // namespace mlx::core::metal
