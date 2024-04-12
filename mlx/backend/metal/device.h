// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <dlfcn.h>
#include <filesystem>

#include "mlx/array.h"
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

struct CommandEncoder {
  CommandEncoder(MTL::ComputeCommandEncoder* enc)
      : enc(enc), concurrent(false){};
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent = true;
    }
    ~ConcurrentContext() {
      enc.concurrent = false;
      enc.outputs.insert(
          enc.concurrent_outputs.begin(), enc.concurrent_outputs.end());
      enc.concurrent_outputs.clear();
    }

   private:
    CommandEncoder& enc;
  };

  MTL::ComputeCommandEncoder* operator->() {
    return enc;
  }

  void set_input_array(const array& a, int idx, int offset = 0) {
    auto r_buf =
        static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
    if (auto it = outputs.find(r_buf); it != outputs.end()) {
      // Insert a barrier
      enc->memoryBarrier(&r_buf, 1);

      // Remove the output
      outputs.erase(it);
    }
    auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
    auto base_offset = a.data<char>() -
        static_cast<char*>(const_cast<MTL::Buffer*>(a_buf)->contents());
    base_offset += offset;
    enc->setBuffer(a_buf, base_offset, idx);
  }

  void set_output_array(array& a, int idx, int offset = 0) {
    // Add barriers before adding the output to the output set
    set_input_array(a, idx, offset);
    auto buf = static_cast<MTL::Resource*>(a.buffer().ptr());
    if (concurrent) {
      concurrent_outputs.insert(buf);
    } else {
      outputs.insert(buf);
    }
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

 private:
  MTL::ComputeCommandEncoder* enc;
  bool concurrent;
  std::unordered_set<MTL::Resource*> outputs;
  std::unordered_set<MTL::Resource*> concurrent_outputs;
};

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
  CommandEncoder& get_command_encoder(int index);
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
  std::unordered_map<int32_t, std::unique_ptr<CommandEncoder>> encoder_map_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_map_;
  std::unordered_map<std::string, MTL::Library*> library_map_;
  std::mutex mtx_;
};

Device& device(mlx::core::Device);

} // namespace mlx::core::metal
