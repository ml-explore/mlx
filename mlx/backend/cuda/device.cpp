// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>
#include <future>
#include <unordered_set>

namespace mlx::core {

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

CommandEncoder::CaptureContext::CaptureContext(CommandEncoder& enc) : enc(enc) {
  CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));
  CHECK_CUDA_ERROR(cudaStreamBeginCaptureToGraph(enc.stream(), graph, NULL, NULL, 0, cudaStreamCaptureModeGlobal));
}

CommandEncoder::CaptureContext::~CaptureContext() {
  CHECK_CUDA_ERROR(cudaStreamEndCapture(enc.stream(), &graph));
  cudaGraphNode_t captured_node;
  CHECK_CUDA_ERROR(cudaGraphAddChildGraphNode(&captured_node, enc.graph_, NULL, 0, graph));
  CHECK_CUDA_ERROR(cudaGraphDestroy(graph));
  if (enc.in_concurrent_) {
    enc.concurrent_nodes_.push_back(captured_node);
  } else {
    enc.insert_graph_dependencies({captured_node});
  }
}
CommandEncoder::ConcurrentContext::ConcurrentContext(CommandEncoder& enc) : enc(enc) {
  enc.in_concurrent_ = true;
}

CommandEncoder::ConcurrentContext::~ConcurrentContext() {
  enc.in_concurrent_ = false;
  enc.insert_graph_dependencies(std::move(enc.concurrent_nodes_));
}

void CommandEncoder::insert_graph_dependencies(std::vector<cudaGraphNode_t> nodes) {
  // Increment number of graph ops
  num_ops_++;

  std::vector<cudaGraphNode_t> deps;
  {
    // Dependencies must be added in the same order to produce a consistent
    // topology
    std::unordered_set<cudaGraphNode_t> set_deps;
    for (auto d : active_deps_) {
      if (auto it = node_map_.find(d); it != node_map_.end()) {
        auto [_, inserted] = set_deps.insert(it->second);
        if (inserted) {
          deps.push_back(it->second);
        }
      }
    }
  }
  active_deps_.clear();

  for (auto o : active_outputs_) {
    for (auto node : nodes) {
      node_map_.emplace(o, node).first->second = node;
    }
  }
  active_outputs_.clear();

  for (auto node : nodes) {
    if (deps.size() == 1) {
      CHECK_CUDA_ERROR(cudaGraphAddDependencies(graph_, deps.data(), &node, deps.size()));
    } else {
      std::vector<cudaGraphNode_t> to_nodes(deps.size(), node);
      CHECK_CUDA_ERROR(cudaGraphAddDependencies(graph_, deps.data(), to_nodes.data(), deps.size()));
    }
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    it = encoders_.try_emplace(s.index, *this).first;
  }
  return it->second;
}

CommandEncoder::CommandEncoder(Device& d) : stream_(d) {
  CHECK_CUDA_ERROR(cudaGraphCreate(&graph_, 0));
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
  if (num_ops_ > 10) {
    commit();
  }
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  if (num_ops_ > 0) {
    // Try in-place update
    if (graph_exec_ != NULL) {
      cudaGraphExecUpdateResultInfo update_result;
      cudaGraphExecUpdate(graph_exec_, graph_, &update_result);
      if (update_result.result != cudaGraphExecUpdateSuccess) {
        cudaGetLastError();
        CHECK_CUDA_ERROR(cudaGraphExecDestroy(graph_exec_));
        graph_exec_ = NULL;
      }
    }
    if (graph_exec_ == NULL) {
      CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
    }
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec_, stream_));

    num_ops_ = 0;
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
