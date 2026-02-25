// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/stream.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cudnn.h>

#include <unordered_map>

namespace mlx::core::cu {

class CommandEncoder {
 public:
  struct CaptureContext {
    CaptureContext(CommandEncoder& enc);
    ~CaptureContext();
    CudaGraph graph;
    CommandEncoder& enc;
    bool discard{false};
  };
  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc);
    ~ConcurrentContext();
    CommandEncoder& enc;
  };

  explicit CommandEncoder(Device& d);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  CaptureContext capture_context() {
    return CaptureContext{*this};
  }
  ConcurrentContext concurrent_context() {
    return ConcurrentContext{*this};
  }

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F, typename... Params>
  void
  add_kernel_node(F* func, dim3 grid_dim, dim3 block_dim, Params&&... params) {
    add_kernel_node_ex(func, grid_dim, block_dim, {}, 0, params...);
  }

  template <typename F, typename... Params>
  void add_kernel_node_ex(
      F* func,
      dim3 grid_dim,
      dim3 block_dim,
      dim3 cluster_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    constexpr size_t num = sizeof...(Params);
    void* ptrs[num];
    size_t i = 0;
    ([&](auto&& p) { ptrs[i++] = static_cast<void*>(&p); }(
         std::forward<Params>(params)),
     ...);
    add_kernel_node_raw(
        reinterpret_cast<void*>(func),
        grid_dim,
        block_dim,
        cluster_dim,
        smem_bytes,
        ptrs);
  }

  void add_kernel_node_raw(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      dim3 cluster_dim,
      uint32_t smem_bytes,
      void** params);

  void add_kernel_node_raw(
      CUfunction func,
      dim3 grid_dim,
      dim3 block_dim,
      dim3 cluster_dim,
      uint32_t smem_bytes,
      void** params);

  void add_graph_node(cudaGraph_t child);

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> task);
  bool needs_commit();
  void commit();

  Device& device() {
    return device_;
  }

  CudaStream& stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished
  void synchronize();

 private:
  cudaGraphNode_t add_kernel_node_raw(const cudaKernelNodeParams& params);
  CUgraphNode add_kernel_node_raw(const CUDA_KERNEL_NODE_PARAMS& params);

  struct GraphNode {
    cudaGraphNode_t node;
    // K = kernel
    // E = empty
    // () = subgraph (with metadata)
    // Symbols ':', '-' are reserved as separators
    std::string node_type;
    std::string id;
  };

  void insert_graph_dependencies(GraphNode node);
  void insert_graph_dependencies(std::vector<GraphNode> nodes);

  Device& device_;
  CudaStream stream_;
  CudaGraph graph_;
  Worker worker_;
  int node_count_{0};
  bool in_concurrent_{false};
  std::vector<cudaGraphNode_t> from_nodes_;
  std::vector<cudaGraphNode_t> to_nodes_;
  std::string graph_nodes_key_;
  std::string graph_deps_key_;
  std::vector<GraphNode> concurrent_nodes_;
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  LRUCache<std::string, CudaGraphExec> graph_cache_;
  std::vector<std::uintptr_t> active_deps_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, GraphNode> node_map_;
  size_t bytes_in_graph_{0};
  bool is_graph_updatable_{true};
  int max_ops_per_graph_;
  int max_mb_per_graph_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(Device&&) = default;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current cuda device, this method is thread-safe.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);
  cublasLtHandle_t get_cublaslt_handle();
  cudnnHandle_t get_cudnn_handle();

  int cuda_device() const {
    return device_;
  }
  int compute_capability_major() const {
    return compute_capability_major_;
  }
  int compute_capability_minor() const {
    return compute_capability_minor_;
  }
  bool concurrent_managed_access() const {
    return concurrent_managed_access_ == 1;
  }
  bool host_native_atomic() const {
    return host_native_atomic_ == 1;
  }
  bool managed_memory() const {
    return managed_memory_ == 1;
  }
  bool memory_pools() const {
    return memory_pools_ == 1;
  }

 private:
  int device_;
  int compute_capability_major_;
  int compute_capability_minor_;
  int concurrent_managed_access_;
  int host_native_atomic_;
  int managed_memory_;
  int memory_pools_;
  std::string device_name_;
  cublasLtHandle_t cublaslt_handle_{nullptr};
  cudnnHandle_t cudnn_handle_{nullptr};
  std::unordered_map<int, CommandEncoder> encoders_;
};

Device& device(int cuda_device);
Device& device(mlx::core::Device d);
CommandEncoder& get_command_encoder(Stream s);

} // namespace mlx::core::cu
