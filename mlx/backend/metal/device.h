// Copyright © 2023 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include <dlfcn.h>
#include <filesystem>

#include "mlx/device.h"

namespace fs = std::filesystem;

namespace mlx::core::metal {

inline std::string get_colocated_mtllib_path(const std::string& lib_name) {
  Dl_info info;
  std::string mtllib_path;
  std::string lib_ext = lib_name + ".metallib";

  int success = dladdr((void*)get_colocated_mtllib_path, &info);
  if (success) {
    auto mtllib = fs::path(info.dli_fname).remove_filename() / lib_ext;
    mtllib_path = mtllib.c_str();
  }

  return mtllib_path;
}

class Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_;
  };

  void new_queue(int index);
  MTL::CommandBuffer* new_command_buffer(int index);
  MTL::CommandBuffer* get_command_buffer(int index);
  int get_command_buffer_ops(int index);
  void increment_command_buffer_ops(int index);
  void commit_command_buffer(int index);
  MTL::ComputeCommandEncoder* get_command_encoder(int index);
  void end_encoding(int index);

  void register_library(
      const std::string& lib_name,
      const std::string& lib_path);
  void register_library(
      const std::string& lib_name,
      const std::function<std::string(const std::string&)>& lib_path_func =
          get_colocated_mtllib_path);

  MTL::ComputePipelineState* get_kernel(
      const std::string& name,
      const std::string& lib_name = "mlx");

  MTL::ArgumentEncoder* argument_encoder(
      const std::vector<MTL::ArgumentDescriptor*>& arg_descs) const;

 private:
  MTL::Device* device_;
  std::unordered_map<int32_t, MTL::CommandQueue*> queue_map_;
  std::unordered_map<int32_t, std::pair<int, MTL::CommandBuffer*>> buffer_map_;
  std::unordered_map<int32_t, MTL::ComputeCommandEncoder*> encoder_map_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_map_;
  std::unordered_map<std::string, MTL::Library*> library_map_;
  std::mutex mtx_;
};

Device& device(mlx::core::Device);

} // namespace mlx::core::metal
