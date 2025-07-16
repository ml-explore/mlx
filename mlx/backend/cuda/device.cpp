// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/utils.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>
#include <future>
#include <unordered_set>

namespace mlx::core {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
// This should be less than 255
constexpr int default_max_nodes_per_graph = 20;

int cuda_graph_cache_size() {
  static int cache_size = []() {
    return env::get_var("MLX_CUDA_GRAPH_CACHE_SIZE", 100);
  }();
  return cache_size;
}

namespace cu {

Device::Device(int device) : device_(device) {
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &compute_capability_major_, cudaDevAttrComputeCapabilityMajor, device_));
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &compute_capability_minor_, cudaDevAttrComputeCapabilityMinor, device_));
  // Validate the requirements of device.
  int attr = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &attr, cudaDevAttrConcurrentManagedAccess, device_));
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device_));
  }
  // The cublasLt handle is used by matmul.
  make_current();
  cublasLtCreate(&lt_);
}

Device::~Device() {
  cublasLtDestroy(lt_);
}

void Device::make_current() {
  // We need to set/get current CUDA device very frequently, cache it to reduce
  // actual calls of CUDA APIs. This function assumes single-thread in host.
  static int current = 0;
  if (current != device_) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_));
    current = device_;
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    it = encoders_.try_emplace(s.index, *this).first;
  }
  return it->second;
}

CommandEncoder::CaptureContext::CaptureContext(CommandEncoder& enc) : enc(enc) {
  CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));
  CHECK_CUDA_ERROR(
      cudaStreamBeginCapture(enc.stream(), cudaStreamCaptureModeGlobal));
}

CommandEncoder::CaptureContext::~CaptureContext() {
  CHECK_CUDA_ERROR(cudaStreamEndCapture(enc.stream(), &graph));
  size_t num_nodes;
  CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, NULL, &num_nodes));
  if (num_nodes == 1) {
    cudaGraphNode_t captured_node;
    CHECK_CUDA_ERROR(cudaGraphGetNodes(graph, &captured_node, &num_nodes));
    CUDA_KERNEL_NODE_PARAMS params;
    CHECK_CUDA_ERROR(cuGraphKernelNodeGetParams(captured_node, &params));
    cudaGraphNode_t node;
    CHECK_CUDA_ERROR(cuGraphAddKernelNode(&node, enc.graph_, NULL, 0, &params));
    enc.insert_graph_dependencies(GraphNode{node, 'K'});
  } else {
    cudaGraphNode_t node;
    CHECK_CUDA_ERROR(
        cudaGraphAddChildGraphNode(&node, enc.graph_, NULL, 0, graph));
    enc.insert_graph_dependencies(GraphNode{node, 'G'});
  }
  CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
}

CommandEncoder::ConcurrentContext::ConcurrentContext(CommandEncoder& enc)
    : enc(enc) {
  enc.in_concurrent_ = true;
}

CommandEncoder::ConcurrentContext::~ConcurrentContext() {
  enc.in_concurrent_ = false;

  // Use an empty graph node for synchronization
  CommandEncoder::GraphNode empty{NULL, 'E', std::to_string(enc.node_count_++)};
  enc.empty_node_count_++;
  CHECK_CUDA_ERROR(cudaGraphAddEmptyNode(&empty.node, enc.graph_, NULL, 0));

  // Insert the concurrent -> empty node dependencies
  for (auto& from : enc.concurrent_nodes_) {
    enc.from_nodes_.push_back(from.node);
    enc.to_nodes_.push_back(empty.node);
    enc.graph_key_ += from.id;
    enc.graph_key_ += from.node_type;
    enc.graph_key_ += empty.id;
    enc.graph_key_ += empty.node_type;
  }

  // Insert the input -> concurrent node dependencies without updating output
  // nodes
  auto outputs = std::move(enc.active_outputs_);
  enc.insert_graph_dependencies(std::move(enc.concurrent_nodes_));

  // Update output node to be the empty node
  for (auto o : outputs) {
    enc.node_map_.emplace(o, empty).first->second = empty;
  }
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  if (node.node_type == 'G') {
    graph_node_count_++;
  }
  node.id = std::to_string(node_count_++);
  if (in_concurrent_) {
    concurrent_nodes_.push_back(std::move(node));
  } else {
    std::vector<GraphNode> nodes;
    nodes.push_back(std::move(node));
    insert_graph_dependencies(std::move(nodes));
  }
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  std::vector<GraphNode> deps;
  {
    // Dependencies must be added in the same order to produce a consistent
    // topology
    std::unordered_set<cudaGraphNode_t> set_deps;
    for (auto d : active_deps_) {
      if (auto it = node_map_.find(d); it != node_map_.end()) {
        auto [_, inserted] = set_deps.insert(it->second.node);
        if (inserted) {
          deps.push_back(it->second);
        }
      }
    }
  }
  active_deps_.clear();

  for (auto o : active_outputs_) {
    for (auto& node : nodes) {
      node_map_.emplace(o, node).first->second = node;
    }
  }
  active_outputs_.clear();

  for (auto& from : deps) {
    for (auto& to : nodes) {
      from_nodes_.push_back(from.node);
      to_nodes_.push_back(to.node);
      graph_key_ += from.id;
      graph_key_ += from.node_type;
      graph_key_ += to.id;
      graph_key_ += to.node_type;
    }
  }
}

CommandEncoder::CommandEncoder(Device& d) : device_(d), stream_(d) {
  CHECK_CUDA_ERROR(cudaGraphCreate(&graph_, 0));
}

void clear_graphs(std::unordered_map<std::string, cudaGraphExec_t>& graphs) {
  for (auto& [_, graph_exec] : graphs) {
    CHECK_CUDA_ERROR(cudaGraphExecDestroy(graph_exec));
  }
  graphs.clear();
}

CommandEncoder::~CommandEncoder() {
  clear_graphs(graph_cache_);
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_.add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::maybe_commit() {
  if (node_count_ >= env::max_ops_per_buffer(default_max_nodes_per_graph)) {
    commit();
  }
}

void CommandEncoder::add_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    void** params) {
  cudaKernelNodeParams kernel_params = {0};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  cudaGraphNode_t node;
  CHECK_CUDA_ERROR(
      cudaGraphAddKernelNode(&node, graph_, NULL, 0, &kernel_params));
  insert_graph_dependencies(GraphNode{node, 'K'});
}

void CommandEncoder::add_kernel_node(
    CUfunction func,
    dim3 grid_dim,
    dim3 block_dim,
    void** params) {
  CUDA_KERNEL_NODE_PARAMS kernel_params = {0};
  kernel_params.func = func;
  kernel_params.gridDimX = grid_dim.x;
  kernel_params.gridDimY = grid_dim.y;
  kernel_params.gridDimZ = grid_dim.z;
  kernel_params.blockDimX = block_dim.x;
  kernel_params.blockDimY = block_dim.y;
  kernel_params.blockDimZ = block_dim.z;
  kernel_params.kernelParams = params;
  CUgraphNode node;
  CHECK_CUDA_ERROR(
      cuGraphAddKernelNode(&node, graph_, NULL, 0, &kernel_params));
  insert_graph_dependencies(GraphNode{node, 'K'});
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  if (node_count_ > 0) {
    if (!from_nodes_.empty()) {
      CHECK_CUDA_ERROR(cudaGraphAddDependencies(
          graph_, from_nodes_.data(), to_nodes_.data(), from_nodes_.size()));
    }

    graph_key_ += ".";
    graph_key_ += std::to_string(node_count_);
    graph_key_ += ".";
    graph_key_ += std::to_string(graph_node_count_);
    graph_key_ += ".";
    graph_key_ += std::to_string(empty_node_count_);

    cudaGraphExec_t& graph_exec = graph_cache_[graph_key_];

    if (graph_exec != nullptr) {
      cudaGraphExecUpdateResult update_result;
#if CUDART_VERSION >= 12000
      cudaGraphExecUpdateResultInfo info;
      cudaGraphExecUpdate(graph_exec, graph_, &info);
      update_result = info.result;
#else
      cudaGraphNode_t error_node;
      cudaGraphExecUpdate(graph_exec, graph_, &error_node, &update_result);
#endif // CUDART_VERSION >= 12000
      if (update_result != cudaGraphExecUpdateSuccess) {
        cudaGetLastError(); // reset error
        CHECK_CUDA_ERROR(cudaGraphExecDestroy(graph_exec));
        graph_exec = nullptr;
      }
    }
    if (graph_exec == nullptr) {
      CHECK_CUDA_ERROR(
          cudaGraphInstantiate(&graph_exec, graph_, NULL, NULL, 0));
    }
    device_.make_current();
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream_));

    // TODO smarter cache policy
    if (graph_cache_.size() > cuda_graph_cache_size()) {
      clear_graphs(graph_cache_);
    }

    // Reset state
    node_count_ = 0;
    graph_node_count_ = 0;
    from_nodes_.clear();
    to_nodes_.clear();
    graph_key_.clear();
    node_map_.clear();
    CHECK_CUDA_ERROR(cudaGraphDestroy(graph_));
    CHECK_CUDA_ERROR(cudaGraphCreate(&graph_, 0));
  }

  // Put completion handlers in a batch.
  worker_.end_batch();
  worker_.commit(stream_);
}

void CommandEncoder::synchronize() {
  cudaStreamSynchronize(stream_);
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  worker_.end_batch();
  commit();
  f.wait();
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device(s.device).get_command_encoder(s);
}

} // namespace cu

} // namespace mlx::core
