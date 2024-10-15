// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/device.h"

namespace fs = std::filesystem;

namespace mlx::core::metal {

// Note, this function must be left inline in a header so that it is not
// dynamically linked.
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
  CommandEncoder(MTL::CommandBuffer* cbuf);
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

  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void dispatchThreadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatchThreads(MTL::Size grid_dims, MTL::Size group_dims);

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  ~CommandEncoder();

 private:
  void maybe_split();

  int num_dispatches{0};
  MTL::CommandBuffer* cbuf;
  MTL::ComputeCommandEncoder* enc;
  bool concurrent{false};
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
  MTL::CommandBuffer* get_command_buffer(int index);
  int get_command_buffer_ops(int index);
  void increment_command_buffer_ops(int index);
  void commit_command_buffer(int index);
  CommandEncoder& get_command_encoder(int index);
  void end_encoding(int index);

  void register_library(
      const std::string& lib_name,
      const std::string& lib_path);

  // Note, this should remain in the header so that it is not dynamically
  // linked
  void register_library(const std::string& lib_name) {
    if (auto it = library_map_.find(lib_name); it == library_map_.end()) {
      register_library(lib_name, get_colocated_mtllib_path(lib_name));
    }
  }

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder);

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

  MTL::Library* get_library_(const std::string& name);
  MTL::Library* build_library_(const std::string& source_string);

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

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::Device* device_;
  std::unordered_map<int32_t, MTL::CommandQueue*> queue_map_;
  std::unordered_map<int32_t, std::pair<int, MTL::CommandBuffer*>> buffer_map_;
  std::unordered_map<int32_t, std::unique_ptr<CommandEncoder>> encoder_map_;

  std::shared_mutex kernel_mtx_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_map_;

  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, MTL::Library*> library_map_;
};

Device& device(mlx::core::Device);

} // namespace mlx::core::metal
