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

struct DeviceStream;

struct CommandEncoder {
  explicit CommandEncoder(DeviceStream& stream);
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent_ = true;
    }
    ~ConcurrentContext() {
      enc.concurrent_ = false;
      enc.prev_outputs_.insert(
          enc.concurrent_outputs_.begin(), enc.concurrent_outputs_.end());
      enc.concurrent_outputs_.clear();
    }

   private:
    CommandEncoder& enc;
  };

  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);
  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();
  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);

  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel) {
    enc_->setComputePipelineState(kernel);
  }

  void wait_for_fence(MTL::Fence* fence) {
    enc_->waitForFence(fence);
  }

  void update_fence(MTL::Fence* fence) {
    enc_->updateFence(fence);
  }

  template <typename T>
  void set_vector_bytes(const std::vector<T>& vec, size_t nelems, int idx) {
    enc_->setBytes(vec.data(), nelems * sizeof(T), idx);
  }
  template <typename T>
  void set_vector_bytes(const std::vector<T>& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    return enc_->setBytes(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    return enc_->setBytes(&v, sizeof(T), idx);
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }
  ~CommandEncoder();

  // Inputs to all kernels in the encoder including temporaries
  std::unordered_set<const void*>& inputs() {
    return all_inputs_;
  };

  // Outputs of all kernels in the encoder including temporaries
  std::unordered_set<const void*> outputs() {
    return all_outputs_;
  };

  void barrier();

 private:
  DeviceStream& stream_;
  MTL::ComputeCommandEncoder* enc_;
  bool needs_barrier_{false};
  bool concurrent_{false};
  std::unordered_set<MTL::Resource*> prev_outputs_;
  std::unordered_set<MTL::Resource*> next_outputs_;
  std::unordered_set<MTL::Resource*> concurrent_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;
};

struct Fence {
  Fence(MTL::Fence* fence) : fence(fence) {}
  ~Fence() {
    fence->release();
  }
  MTL::Fence* fence;
};

struct DeviceStream {
  DeviceStream(MTL::CommandQueue* queue) : queue(queue) {};
  ~DeviceStream() {
    queue->release();
    if (buffer != nullptr) {
      buffer->release();
    }
  };
  MTL::CommandQueue* queue;
  // A map of prior command encoder outputs to their corresponding fence
  std::unordered_map<const void*, std::shared_ptr<Fence>> outputs;
  // Used to allow thread-safe access to the outputs map
  std::mutex fence_mtx;

  // Data updated between command buffers
  MTL::CommandBuffer* buffer{nullptr};
  int buffer_ops{0};
  size_t buffer_sizes{0};

  // The command encoder, fence, and temporaries are updated between command
  // encoders
  std::unique_ptr<CommandEncoder> encoder{nullptr};
  std::shared_ptr<Fence> fence;
  std::vector<array> temporaries;
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

  const std::string& get_architecture() {
    return arch_;
  }

  void new_queue(int index);

  MTL::CommandQueue* get_queue(Stream stream);

  MTL::CommandBuffer* get_command_buffer(int index);
  bool command_buffer_needs_commit(int index);
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

  // Record temporary arrays for the given stream index
  void add_temporary(array arr, int index);
  void add_temporaries(std::vector<array> arrays, int index);

  void set_residency_set(const MTL::ResidencySet* residency_set);

 private:
  DeviceStream& get_stream_(int index) {
    return stream_map_.find(index)->second;
  }
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
  std::unordered_map<int32_t, DeviceStream> stream_map_;

  std::shared_mutex kernel_mtx_;
  std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_map_;

  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, MTL::Library*> library_map_;
  const MTL::ResidencySet* residency_set_{nullptr};
  std::string arch_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

Device& device(mlx::core::Device);

} // namespace mlx::core::metal
