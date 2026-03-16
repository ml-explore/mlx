// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Metal/Metal.hpp>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/device.h"

namespace mlx::core::metal {

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

class Device;

class MLX_API CommandEncoder {
 public:
  CommandEncoder(Device& d, int index, const MTL::ResidencySet* residency_set);
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

  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);

  void add_temporary(array arr);
  void add_temporaries(std::vector<array> arrays);

  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();

  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel) {
    get_command_encoder()->setComputePipelineState(kernel);
  }

  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, size_t nelems, int idx) {
    get_command_encoder()->setBytes(
        vec.data(), nelems * sizeof(typename Vec::value_type), idx);
  }
  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    return get_command_encoder()->setBytes(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    return get_command_encoder()->setBytes(&v, sizeof(T), idx);
  }

  void set_threadgroup_memory_length(size_t length, int idx) {
    get_command_encoder()->setThreadgroupMemoryLength(length, idx);
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  void barrier();
  void end_encoding();
  bool needs_commit() const;
  void commit();

  MTL::CommandQueue* get_command_queue() const {
    return queue_.get();
  }
  MTL::CommandBuffer* get_command_buffer() const {
    return buffer_.get();
  }

 private:
  MTL::ComputeCommandEncoder* get_command_encoder();

  Device& device_;

  // Buffer that stores encoded commands.
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  size_t buffer_sizes_{0};

  // Encoder for issuing GPU commands.
  // The members are used within a single ComputeCommandEncoder and will be
  // reset after calling end_encoding().
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  NS::SharedPtr<MTL::Fence> fence_;
  bool needs_barrier_{false};
  bool concurrent_{false};
  std::vector<array> temporaries_;
  std::unordered_set<MTL::Resource*> prev_outputs_;
  std::unordered_set<MTL::Resource*> next_outputs_;
  std::unordered_set<MTL::Resource*> concurrent_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;

  // A map of prior command encoder outputs to their corresponding fence.
  std::unordered_map<const void*, NS::SharedPtr<MTL::Fence>> prev_ce_outputs_;
  std::mutex outputs_mtx_;
};

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_.get();
  };

  const std::string& get_architecture() const {
    return arch_;
  }
  int get_architecture_gen() const {
    return arch_gen_;
  }
  std::tuple<int, int> get_max_ops_mb_per_buffer() const {
    return std::make_tuple(max_ops_per_buffer_, max_mb_per_buffer_);
  }

  MTL::CommandBuffer* get_command_buffer(int index);
  bool command_buffer_needs_commit(int index);
  void commit_command_buffer(int index);
  CommandEncoder& get_command_encoder(int index);
  void end_encoding(int index);

  MTL::Library* get_library(
      const std::string& name,
      const std::string& path = "");

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder);

  void clear_library(const std::string& name);

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  // Record temporary arrays for the given stream index
  void add_temporary(array arr, int index);
  void add_temporaries(std::vector<array> arrays, int index);

  void set_residency_set(const MTL::ResidencySet* residency_set);

 private:
  NS::SharedPtr<MTL::Library> build_library_(const std::string& source_string);

  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      MTL::Library* mtl_lib);
  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  NS::SharedPtr<MTL::LinkedFunctions> get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);
  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  NS::SharedPtr<MTL::Device> device_;
  std::unordered_map<int32_t, CommandEncoder> encoders_;

  std::shared_mutex kernel_mtx_;
  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::Library>> library_map_;
  NS::SharedPtr<MTL::Library> default_library_;
  std::unordered_map<
      MTL::Library*,
      std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>
      library_kernels_;
  const MTL::ResidencySet* residency_set_{nullptr};
  std::string arch_;
  int arch_gen_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

MLX_API Device& device(mlx::core::Device);

NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool();

bool is_nax_available();

} // namespace mlx::core::metal
