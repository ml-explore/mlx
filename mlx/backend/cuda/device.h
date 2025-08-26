// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/lru_cache.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/stream.h"

#include <cublasLt.h>
#include <cuda.h>
#include <cudnn.h>
#include <thrust/execution_policy.h>

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
  void add_kernel_node(
      F* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    constexpr size_t num = sizeof...(Params);
    void* ptrs[num];
    size_t i = 0;
    ([&](auto&& p) { ptrs[i++] = static_cast<void*>(&p); }(
         std::forward<Params>(params)),
     ...);
    add_kernel_node((void*)func, grid_dim, block_dim, smem_bytes, ptrs);
  }

  void add_kernel_node(
      CUfunction func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  void add_kernel_node(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  void add_graph_node(cudaGraph_t child);

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> task);
  void maybe_commit();
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
  void add_kernel_node(const cudaKernelNodeParams& params);
  void add_kernel_node(const CUDA_KERNEL_NODE_PARAMS& params);

  struct GraphNode {
    cudaGraphNode_t node;
    // K = kernel
    // E = empty
    // G = subgraph
    char node_type;
    std::string id;
  };

  void insert_graph_dependencies(GraphNode node);
  void insert_graph_dependencies(std::vector<GraphNode> nodes);

  Device& device_;
  CudaStream stream_;
  CudaGraph graph_;
  Worker worker_;
  char node_count_{0};
  char graph_node_count_{0};
  char empty_node_count_{0};
  bool in_concurrent_{false};
  std::vector<cudaGraphNode_t> from_nodes_;
  std::vector<cudaGraphNode_t> to_nodes_;
  std::string graph_key_;
  std::vector<GraphNode> concurrent_nodes_;
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  LRUCache<std::string, CudaGraphExec> graph_cache_;
  std::vector<std::uintptr_t> active_deps_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, GraphNode> node_map_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current cuda device, required by some cuda calls.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);

  int cuda_device() const {
    return device_;
  }
  int compute_capability_major() const {
    return compute_capability_major_;
  }
  int compute_capability_minor() const {
    return compute_capability_minor_;
  }
  cublasLtHandle_t lt_handle() const {
    return lt_;
  }
  cudnnHandle_t cudnn_handle() const {
    return cudnn_;
  }

 private:
  int device_;
  int compute_capability_major_;
  int compute_capability_minor_;
  cublasLtHandle_t lt_;
  cudnnHandle_t cudnn_;
  std::unordered_map<int, CommandEncoder> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);

// Return an execution policy that does not sync for result.
// Note that not all thrust APIs support async policy, confirm before using.
inline auto thrust_policy(cudaStream_t stream) {
  // TODO: Connect thrust's custom allocator with mlx's allocator.
  return thrust::cuda::par_nosync.on(stream);
}

} // namespace mlx::core::cu
