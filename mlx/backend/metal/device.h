// Copyright © 2023-24 Apple Inc.

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

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

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

  MTL::Library* get_library(const std::string& name);

  MTL::Library* get_library(
      const std::string& name,
      const std::string& source_string,
      bool cache = true);

  MTL::Library* get_library(
      const std::string& name,
      const MTL::StitchedLibraryDescriptor* desc,
      bool cache = true);

  MTL::Function* get_function(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& specialized_name = "",
      const MTLFCList& func_consts = {});

  MTL::Function* get_function(
      const std::string& base_name,
      const std::string& lib_name = "mlx",
      const std::string& specialized_name = "",
      const MTLFCList& func_consts = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& lib_name = "mlx",
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ArgumentEncoder* argument_encoder(
      const std::vector<MTL::ArgumentDescriptor*>& arg_descs) const;

 private:
  MTL::Library* get_library_cache_(const std::string& name);

  MTL::Library* get_library_(const std::string& source_string);
  MTL::Library* get_library_(const MTL::StitchedLibraryDescriptor* desc);

  MTL::Function* get_function_(const std::string& name, MTL::Library* mtl_lib);

  MTL::Function* get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  MTL::LinkedFunctions* get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

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
