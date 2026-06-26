// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <atomic>
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace mlx::core::rocm {

namespace {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
constexpr int default_max_ops_per_buffer = 2000;

inline bool is_empty_dim(dim3 dim) {
  return (dim.x == 0 && dim.y == 0 && dim.z == 0) ||
      (dim.x == 1 && dim.y == 1 && dim.z == 1);
}

} // namespace

// True while the engine is in a single-token decode step; false during prefill
// (multi-token) and outside generation. Set by set_graph_decode_mode().
static std::atomic<bool> g_graph_decode_mode{false};

bool use_hip_graphs() {
  // Default OFF: the rebuild-every-eval graph path is a net loss vs eager when
  // launches are cheap (build/instantiate overhead exceeds the launch-batching
  // win, and one-shot prefill has no replay to amortize the build).
  // Independent opt-in per region: MLX_HIP_GRAPH_PREFILL / MLX_HIP_GRAPH_DECODE.
  // Legacy MLX_USE_HIP_GRAPHS=1 turns both on.
  static const bool legacy = [] {
    const char* e = std::getenv("MLX_USE_HIP_GRAPHS");
    return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0);
  }();
  static const bool prefill = legacy || [] {
    const char* e = std::getenv("MLX_HIP_GRAPH_PREFILL");
    return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0);
  }();
  static const bool decode = legacy || [] {
    const char* e = std::getenv("MLX_HIP_GRAPH_DECODE");
    return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0);
  }();
  return g_graph_decode_mode.load(std::memory_order_relaxed) ? decode : prefill;
}

// Count of inline (graph-splitting) launches — library GEMM / JIT / memset run
// outside the graph via launch_kernel. Declared extern in device.h.
std::atomic<long> g_inline_launches_{0};
void set_current_prim(const char*) {}
void record_inline_launch() {
  g_inline_launches_.fetch_add(1, std::memory_order_relaxed);
}

// Per-arch op/MB caps for the build graph. Tunable via env.
// The earlier "corrupts at >3 nodes" was actually one bad op (Concatenate, whose
// multi-copy kernels corrupt when co-grouped); it is now graph-split in
// gpu::eval (is_graph_split_op), so large graphs are correct again.
static std::pair<int, int> get_graph_limits() {
  int ops = env::max_ops_per_buffer(50);
  int mb = env::max_mb_per_buffer(200);
  return {ops, mb};
}

Device::Device(int device) : device_(device) {
  make_current();
  {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, device_) == hipSuccess) {
      fprintf(stderr, "[mlx-rocm] bound HIP device %d: %s (%s)\n",
              device_, p.gcnArchName, p.name);
      if (p.sharedMemPerBlock > 0) {
        max_shared_memory_per_block_ = static_cast<int>(p.sharedMemPerBlock);
      }
    }
  }
  // rocBLAS initialization is now lazy - done in get_rocblas_handle()
}

Device::~Device() {
  if (rocblas_) {
    rocblas_destroy_handle(rocblas_);
  }
}

rocblas_handle Device::get_rocblas_handle() {
  if (!rocblas_initialized_) {
    rocblas_initialized_ = true;
    make_current();

    // Check if the GPU architecture is supported by rocBLAS
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_);
    std::string arch_name = props.gcnArchName;

    // List of architectures supported by rocBLAS (based on TensileLibrary
    // files). These are the architectures that have TensileLibrary_lazy_*.dat.
    static const std::vector<std::string> supported_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1200",
        "gfx1201"};

    // Extract base architecture name (remove any suffix like :sramecc+:xnack-)
    std::string base_arch = arch_name;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    bool arch_supported = false;
    for (const auto& supported : supported_archs) {
      if (base_arch == supported) {
        arch_supported = true;
        break;
      }
    }

    if (!arch_supported) {
      rocblas_available_ = false;
      rocblas_ = nullptr;
      std::cerr << "Warning: rocBLAS does not support GPU architecture '"
                << arch_name << "'. "
                << "Matrix multiplication operations will not be available. "
                << "Supported architectures: gfx908, gfx90a, gfx942, gfx950, "
                << "gfx1030, gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, "
                << "gfx1200, gfx1201." << std::endl;
    } else {
      rocblas_status status = rocblas_create_handle(&rocblas_);
      if (status != rocblas_status_success) {
        rocblas_available_ = false;
        rocblas_ = nullptr;
        std::cerr
            << "Warning: rocBLAS initialization failed (status "
            << static_cast<int>(status)
            << "). Matrix multiplication operations will not be available."
            << std::endl;
      }
    }
  }
  if (!rocblas_available_) {
    throw std::runtime_error(
        "rocBLAS is not available on this GPU architecture. "
        "Matrix multiplication operations are not supported.");
  }
  return rocblas_;
}

bool Device::is_rocblas_available() {
  if (!rocblas_initialized_) {
    try {
      get_rocblas_handle();
    } catch (...) {
    }
  }
  return rocblas_available_;
}

bool Device::is_rocblas_bf16_available() {
  if (!rocblas_bf16_probed_) {
    rocblas_bf16_probed_ = true;
    rocblas_bf16_available_ = false;

    if (!is_rocblas_available()) {
      return false;
    }

    // Probe: run a tiny bf16 GEMM and check if the GPU survives.
    // rocBLAS may claim support but crash if the Tensile .co files
    // are corrupt or missing specific kernel variants.
    make_current();
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    hipError_t err;

    err = hipMalloc(&a_ptr, 4 * 4 * 2); // 4x4 bf16
    if (err != hipSuccess)
      return false;
    err = hipMalloc(&b_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      hipFree(a_ptr);
      return false;
    }
    err = hipMalloc(&c_ptr, 4 * 4 * 2);
    if (err != hipSuccess) {
      hipFree(a_ptr);
      hipFree(b_ptr);
      return false;
    }

    (void)hipMemset(a_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(b_ptr, 0, 4 * 4 * 2);
    (void)hipMemset(c_ptr, 0, 4 * 4 * 2);

    float alpha = 1.0f, beta = 0.0f;
    rocblas_status status = rocblas_gemm_ex(
        rocblas_,
        rocblas_operation_none,
        rocblas_operation_none,
        4,
        4,
        4,
        &alpha,
        a_ptr,
        rocblas_datatype_bf16_r,
        4,
        b_ptr,
        rocblas_datatype_bf16_r,
        4,
        &beta,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        c_ptr,
        rocblas_datatype_bf16_r,
        4,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0,
        0);

    // Sync and check if the GPU is still alive
    hipError_t sync_err = hipDeviceSynchronize();
    // Clear any lingering error
    (void)hipGetLastError();

    hipFree(a_ptr);
    hipFree(b_ptr);
    hipFree(c_ptr);

    if (status == rocblas_status_success && sync_err == hipSuccess) {
      rocblas_bf16_available_ = true;
    } else {
      // GPU may be in a bad state — need to reset
      (void)hipDeviceReset();
      // Re-initialize device
      make_current();
      // Re-create rocBLAS handle
      if (rocblas_) {
        rocblas_destroy_handle(rocblas_);
        rocblas_ = nullptr;
      }
      rocblas_status rs = rocblas_create_handle(&rocblas_);
      if (rs != rocblas_status_success) {
        rocblas_available_ = false;
      }
      std::cerr << "Warning: rocBLAS bfloat16 GEMM probe failed on this GPU. "
                << "Using fallback kernels for bf16 matmul." << std::endl;
    }
  }
  return rocblas_bf16_available_;
}

bool Device::has_native_wmma() {
  if (!wmma_probed_) {
    wmma_probed_ = true;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_) != hipSuccess) {
      has_native_wmma_ = false;
      return has_native_wmma_;
    }

    // Strip any ":sramecc+:xnack-" style suffix from gcnArchName.
    std::string base_arch = props.gcnArchName;
    size_t colon_pos = base_arch.find(':');
    if (colon_pos != std::string::npos) {
      base_arch = base_arch.substr(0, colon_pos);
    }

    // rocWMMA arch allowlist (AMD's official support matrix). Keep in sync
    // with detect_rocm_hw_info() in mlx/backend/rocm/quantized/qmm.hip.
    static const std::vector<std::string> rocwmma_archs = {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    };
    for (const auto& a : rocwmma_archs) {
      if (base_arch == a) {
        has_native_wmma_ = true;
        break;
      }
    }
  }
  return has_native_wmma_;
}

void Device::make_current() {
  // HIP's current device is per-thread, so the cache must be too — a process
  // global lets one thread's binding suppress another's, stranding allocations
  // on the wrong device in a multi-GPU / multi-stream-thread run.
  thread_local int current = -1;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

void Device::set_rocblas_stream(hipStream_t stream) {
  if (rocblas_stream_ != stream) {
    rocblas_set_stream(get_rocblas_handle(), stream);
    rocblas_stream_ = stream;
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  // Bind this device current before constructing/returning the encoder. Callers
  // reach this member directly (e.g. QuantizedMatmul::eval_gpu), and the
  // encoder's stream + the kernels launched on it must land on this device, not
  // whatever was current on the calling thread.
  make_current();
  std::lock_guard<std::mutex> lk(encoders_mtx_);
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    auto [inserted_it, success] =
        encoders_.emplace(s.index, std::make_unique<CommandEncoder>(*this));
    it = inserted_it;
  }
  return *it->second;
}

void Device::clear_encoders() {
  encoders_.clear();
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      worker_(std::make_unique<Worker>(d.hip_device())) {
  std::tie(max_ops_per_graph_, max_mb_per_graph_) = get_graph_limits();
  if (use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    set_graph_active(true);
  }
}

CommandEncoder::~CommandEncoder() {
  for (auto& [key, pool] : exec_pool_) {
    for (auto& slot : pool) {
      hipGraphExecDestroy(slot.exec);
      if (slot.source_graph) {
        hipGraphDestroy(slot.source_graph);
      }
    }
  }
  if (build_graph_) {
    hipGraphDestroy(build_graph_);
    build_graph_ = nullptr;
  }
}

void CommandEncoder::add_temporary(const array& arr) {
  auto data = arr.data_shared_ptr();
  const array::Data* ptr = data.get();
  if (temporary_ptrs_.insert(ptr).second) {
    temporaries_.push_back(std::move(data));
  }
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  bytes_in_graph_ += arr.data_size();
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
}

void CommandEncoder::set_output_array(const array& arr) {
  if (!use_hip_graphs()) {
    return;
  }
  auto id = reinterpret_cast<std::uintptr_t>(arr.buffer().ptr());
  active_deps_.push_back(id);
  active_outputs_.push_back(id);
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  node.id = std::to_string(node_count_++);
  std::vector<GraphNode> nodes;
  nodes.push_back(std::move(node));
  insert_graph_dependencies(std::move(nodes));
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  // Two edge sets combined:
  //  (1) REAL data-dependency edges (node_map_: buffer ptr -> producing node).
  //      These give each node a unique data-flow identity, which
  //      hipGraphExecUpdate relies on to re-map kernelParams correctly across
  //      reuse — a bare submission chain leaves same-signature nodes ambiguous
  //      and corrupts under ExecUpdate.
  //  (2) A submission-order CHAIN edge (last_node_ -> node) as a backstop. Not
  //      every migrated kernel registers all of its I/O (the dense GEMM path and
  //      a few others register nothing), so the real-dep graph alone has missing
  //      edges -> races -> coherent-but-wrong output. The chain is a superset of
  //      the true partial order (eager runs serial on one stream), so it fills
  //      every gap and makes the graph bit-correct vs eager, while the real
  //      edges still uniquely identify nodes for ExecUpdate.
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  std::unordered_set<hipGraphNode_t> set_deps;
  for (auto d : active_deps_) {
    if (auto it = node_map_.find(d); it != node_map_.end()) {
      if (set_deps.insert(it->second.node).second) {
        deps.push_back(it->second);
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
      graph_deps_key_ += from.id;
      graph_deps_key_ += "-";
      graph_deps_key_ += to.id;
      graph_deps_key_ += "-";
    }
  }

  // (2) Submission-order CHAIN edge. Serialize every node behind the previously
  // inserted node so the graph's execution order is a superset of eager's
  // serial-on-one-stream order — even for kernels that register no I/O (dense
  // GEMM path and a few others), whose nodes would otherwise have no edges and
  // race their true producers/consumers at any cap > 1. The chain is
  // deterministic for a given node sequence, so identical kernel sequences
  // still produce identical topology and hipGraphExecUpdate stays valid. Skip
  // an edge already present as a real data dep (avoid an exact duplicate edge).
  hipGraphNode_t prev = last_node_;
  for (auto& to : nodes) {
    // A real-dep edge prev->to was already added iff prev is in set_deps; skip
    // then so we never add an exact duplicate edge (HIP rejects duplicates).
    // Internal batch nodes are never external producers, so they always chain.
    if (prev && !set_deps.count(prev)) {
      from_nodes_.push_back(prev);
      to_nodes_.push_back(to.node);
      graph_deps_key_ += "c-";
    }
    prev = to.node;
  }
  last_node_ = prev;
}

void CommandEncoder::add_kernel_node_raw(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params) {
  if (!use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipLaunchKernel(
        func, grid_dim, block_dim, params, smem_bytes, stream_));
    node_count_++;
    return;
  }

  hipKernelNodeParams kernel_params = {};
  kernel_params.func = func;
  kernel_params.gridDim = grid_dim;
  kernel_params.blockDim = block_dim;
  kernel_params.kernelParams = params;
  kernel_params.sharedMemBytes = smem_bytes;
  add_kernel_node_kp(kernel_params);
}

// Key the node by its kernel FUNCTION + FULL launch dims so a reused exec is
// re-pointed only onto a structurally identical graph. The grid/block PRODUCT
// collided distinct shapes (e.g. 2097152x1x1 vs 1024x2048x1); encode each dim.
std::string CommandEncoder::kernel_node_key(const hipKernelNodeParams& kp) {
  std::string key = "K";
  key += std::to_string(reinterpret_cast<std::uintptr_t>(kp.func));
  key += "_";
  key += std::to_string(kp.gridDim.x);
  key += ",";
  key += std::to_string(kp.gridDim.y);
  key += ",";
  key += std::to_string(kp.gridDim.z);
  key += "x";
  key += std::to_string(kp.blockDim.x);
  key += ",";
  key += std::to_string(kp.blockDim.y);
  key += ",";
  key += std::to_string(kp.blockDim.z);
  key += "s";
  key += std::to_string(kp.sharedMemBytes);
  return key;
}

void CommandEncoder::add_kernel_node_kp(const hipKernelNodeParams& kp) {
  // A launch with any zero grid/block dim is a no-op: hipLaunchKernel tolerates
  // it (eager path), but hipGraphAddKernelNode rejects it with "invalid
  // argument". Skip it entirely — it does no work, and skipping is consistent
  // across tokens so the decode topology stays stable for replay.
  if (kp.func == nullptr || kp.gridDim.x == 0 || kp.gridDim.y == 0 ||
      kp.gridDim.z == 0 || kp.blockDim.x == 0 || kp.blockDim.y == 0 ||
      kp.blockDim.z == 0) {
    return;
  }
  // Build-once decode replay: at the first node of a decode-mode token, if the
  // persistent source graph is already built for the (stable) decode topology,
  // switch to param-update mode — re-point the existing source nodes instead of
  // adding new ones; commit() then ExecUpdates a drained pooled exec from it.
  static const bool replay_enabled = [] {
    const char* e = std::getenv("MLX_GRAPH_REPLAY");
    return e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0);
  }();
  static const bool prefill_replay = [] {
    const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
    return e && e[0] == '1';
  }();
  const std::string& rkey = graph_decode_mode() ? decode_key_ : prefill_key_;
  // Build-skip replay (replay_active_) is decode-only: prefill captures library/
  // rejected kernels into the graph each chunk, and capture needs the launch — it
  // can't be build-skipped. Prefill instead uses ExecUpdate (re-capture + refresh
  // a cached exec instead of re-instantiate; see use_execupdate in commit()).
  (void)prefill_replay;
  const bool replay_mode_ok = graph_decode_mode();
  if (replay_enabled && replay_mode_ok && !replay_active_ && node_count_ == 0 &&
      build_node_params_.empty() && use_hip_graphs() && !rkey.empty()) {
    // Only once the normal path has grown the pool to N execs for this topology,
    // so replay always finds a drained slot (completion-handler flags lag the GPU,
    // so the in-flight pipeline can be a few deep) -> always a HIT (re-point via
    // ExecUpdate), never the instantiate-during-replay path (which hangs). N
    // tunable via MLX_GRAPH_REPLAY_SLOTS (default 4).
    static const size_t replay_slots = [] {
      const char* e = std::getenv("MLX_GRAPH_REPLAY_SLOTS");
      return e ? std::max<size_t>(2, std::atoi(e)) : 4;
    }();
    auto it = exec_pool_.find(rkey);
    if (it != exec_pool_.end() && it->second.size() >= replay_slots) {
      replay_active_ = true;
    }
  }

  std::string key = kernel_node_key(kp);
  if (replay_active_) {
    // Buffer only (no AddKernelNode). Use a UNIQUE fake node handle so
    // insert_graph_dependencies' handle-based dedup + chain edges reproduce the
    // EXACT same topology key as the real build. commit() re-points a drained
    // replay exec (ExecKernelNodeSetParams) or self-instantiates a new one.
    build_node_params_.push_back(kp);
    auto fake = reinterpret_cast<hipGraphNode_t>(
        static_cast<std::uintptr_t>(build_node_params_.size()));
    insert_graph_dependencies(GraphNode{fake, key});
    return;
  }

  hipGraphNode_t node;
  hipError_t kn_err = hipGraphAddKernelNode(&node, build_graph_, nullptr, 0, &kp);
  if (kn_err != hipSuccess) {
    // ROCm's hipGraphAddKernelNode rejects some kernels with invalid-argument
    // (certain WMMA / single-block / JIT kernels). In replay mode, don't split —
    // CAPTURE the launch into a child graph node (CUDA's model: capture records
    // any launched kernel that the manual node API won't accept), so the chunk
    // stays ~1 fragment. Else fall back to the eager graph-split.
    (void)hipGetLastError();
    static const bool cap_reject = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    if (cap_reject && !graph_decode_mode()) {
      device_.make_current();
      hipError_t be = hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
      if (be == hipSuccess) {
        (void)hipLaunchKernel(kp.func, kp.gridDim, kp.blockDim, kp.kernelParams,
                              kp.sharedMemBytes, stream_);
        hipGraph_t child = nullptr;
        hipError_t ee = hipStreamEndCapture(stream_, &child);
        if (ee == hipSuccess && child) {
          size_t nn = 0;
          hipGraphGetNodes(child, nullptr, &nn);
          if (nn > 0) {
            add_child_graph_node(child, key);
            hipGraphDestroy(child);
            return;
          }
          hipGraphDestroy(child);
        } else if (child) {
          hipGraphDestroy(child);
        }
        (void)hipGetLastError();
      }
    }
    commit();
    device_.make_current();
    (void)hipLaunchKernel(kp.func, kp.gridDim, kp.blockDim, kp.kernelParams,
                          kp.sharedMemBytes, stream_);
    record_inline_launch();
    return;
  }
  build_nodes_.push_back(node);
  build_node_params_.push_back(kp);
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::add_module_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params,
    std::shared_ptr<void> args_keepalive) {
  if (!use_hip_graphs()) {
    device_.make_current();
    CHECK_HIP_ERROR(hipModuleLaunchKernel(
        reinterpret_cast<hipFunction_t>(func),
        grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z,
        smem_bytes, stream_, params, nullptr));
    node_count_++;
    return;
  }
  // Graph path: the node references `params` (which point into the kept-alive
  // KernelArgs) until commit instantiates the graph. A module hipFunction_t is a
  // valid hipKernelNodeParams.func on ROCm 7.13 (see device.h note).
  if (args_keepalive) {
    graph_node_args_.push_back(std::move(args_keepalive));
  }
  add_kernel_node_raw(func, grid_dim, block_dim, smem_bytes, params);
}

void CommandEncoder::add_child_graph_node(
    hipGraph_t child,
    const std::string& key) {
  // hipGraphExecUpdate does NOT refresh kernel params nested inside child-graph
  // nodes (confirmed: it returns success but the child keeps stale kernargs), so
  // embedding a child node silently breaks graph reuse across tokens. Flatten the
  // child's kernels into build_graph_ as TOP-LEVEL kernel nodes via the normal
  // add path so ExecUpdate refreshes them. The child's kernelParams point at the
  // caller's arg storage (kept alive like any other op), and chain-edge ordering
  // serializes the flattened nodes correctly. Fall back to embedding the child
  // as-is only if it contains non-kernel nodes (none observed in practice).
  size_t n = 0;
  hipGraphGetNodes(child, nullptr, &n);
  std::vector<hipGraphNode_t> cnodes(n);
  if (n) {
    hipGraphGetNodes(child, cnodes.data(), &n);
  }
  // Topologically order the child's kernels (Kahn) so chain-edge serialization
  // never places a consumer before its producer.
  size_t ne = 0;
  hipGraphGetEdges(child, nullptr, nullptr, &ne);
  std::vector<hipGraphNode_t> cfrom(ne), cto(ne);
  if (ne) {
    hipGraphGetEdges(child, cfrom.data(), cto.data(), &ne);
  }
  bool all_kernels = n > 0;
  for (size_t i = 0; i < n; i++) {
    hipGraphNodeType t;
    if (hipGraphNodeGetType(cnodes[i], &t) != hipSuccess ||
        t != hipGraphNodeTypeKernel) {
      all_kernels = false;
      break;
    }
  }
  if (all_kernels) {
    std::unordered_map<hipGraphNode_t, int> indeg;
    std::unordered_map<hipGraphNode_t, std::vector<hipGraphNode_t>> succ;
    for (auto nd : cnodes) indeg[nd] = 0;
    for (size_t i = 0; i < ne; i++) {
      succ[cfrom[i]].push_back(cto[i]);
      indeg[cto[i]]++;
    }
    std::vector<hipGraphNode_t> order;
    order.reserve(n);
    for (auto nd : cnodes)
      if (indeg[nd] == 0) order.push_back(nd);
    for (size_t h = 0; h < order.size(); h++)
      for (auto s : succ[order[h]])
        if (--indeg[s] == 0) order.push_back(s);
    if (order.size() == n) {
      bool ok = true;
      std::vector<hipKernelNodeParams> kps(n);
      for (size_t i = 0; i < n && ok; i++) {
        kps[i] = {};
        if (hipGraphKernelNodeGetParams(order[i], &kps[i]) != hipSuccess)
          ok = false;
      }
      if (ok) {
        for (size_t i = 0; i < n; i++) {
          add_kernel_node_kp(kps[i]); // preserve full params (kernelParams+extra)
        }
        return;
      }
    }
  }
  // Fallback: embed the child as-is (non-kernel nodes or topo failure).
  hipGraphNode_t node;
  CHECK_HIP_ERROR(
      hipGraphAddChildGraphNode(&node, build_graph_, nullptr, 0, child));
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::maybe_commit() {
  if (use_hip_graphs()) {
    if (needs_commit()) {
      commit();
    }
    return;
  }
  if (node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer)) {
    commit();
  }
}

bool CommandEncoder::needs_commit() {
  if (!use_hip_graphs()) {
    return node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer);
  }
  // Decode-mode: never split mid-forward — the whole single-token forward becomes
  // one graph, committed once at finalize, refreshed via ExecUpdate. Decode's live
  // intermediates are small, so the per-graph caps (which bound prefill) aren't
  // needed here.
  if (graph_decode_mode()) {
    return false;
  }
  return (node_count_ > max_ops_per_graph_) ||
      ((bytes_in_graph_ >> 20) > static_cast<size_t>(max_mb_per_graph_));
}

void CommandEncoder::decode_pure_end() {
  decode_pure_mode_ = 0;
  decode_pure_idx_ = 0;
  for (int p = 0; p < 2; ++p) {
    for (auto& pe : decode_pure_chain_[p]) {
      if (pe.exec) hipGraphExecDestroy(pe.exec);
      if (pe.graph) hipGraphDestroy(pe.graph);
    }
    decode_pure_chain_[p].clear();
  }
}

void CommandEncoder::decode_pure_begin_record(int slot) {
  // Drop any prior chain for this parity slot (keep the other parity's chain).
  for (auto& pe : decode_pure_chain_[slot & 1]) {
    if (pe.exec) hipGraphExecDestroy(pe.exec);
    if (pe.graph) hipGraphDestroy(pe.graph);
  }
  decode_pure_chain_[slot & 1].clear();
  decode_pure_mode_ = 1;
  decode_pure_slot_ = slot & 1;
  decode_pure_idx_ = 0;
}

void CommandEncoder::decode_pure_begin_replay(int slot) {
  static const bool dbg = std::getenv("MLX_PURE_CHAINDBG") != nullptr;
  if (dbg && decode_pure_mode_ == 2) {
    fprintf(stderr, "[chain] prev token consumed %zu / %zu recorded (slot %d)\n",
            decode_pure_idx_, decode_pure_chain_[decode_pure_slot_].size(),
            decode_pure_slot_);
  }
  decode_pure_mode_ = 2;
  decode_pure_slot_ = slot & 1;
  decode_pure_idx_ = 0;
}

void CommandEncoder::decode_pure_relaunch_all(int slot) {
  device_.make_current();
  for (auto& pe : decode_pure_chain_[slot & 1]) {
    CHECK_HIP_ERROR(hipGraphLaunch(pe.exec, stream_));
  }
  worker_->commit(stream_);
}

// Reset the per-chunk build accounting (mirrors the tail of the normal commit
// path). recreate_graph=true destroys and re-creates build_graph_ (used on
// replay, where the forward populated it with nodes we discard).
bool CommandEncoder::commit_pure() {
  if (decode_pure_mode_ == 0 || !use_hip_graphs() || node_count_ == 0) {
    return false;
  }
  device_.make_current();

  auto reset_build = [&](bool recreate_graph) {
    if (recreate_graph) {
      hipGraphDestroy(build_graph_);
      CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    }
    from_nodes_.clear();
    to_nodes_.clear();
    graph_nodes_key_.clear();
    graph_deps_key_.clear();
    node_map_.clear();
    active_deps_.clear();
    active_outputs_.clear();
    last_node_ = nullptr;
    bytes_in_graph_ = 0;
    build_nodes_.clear();
    build_node_params_.clear();
    node_count_ = 0;
  };

  if (decode_pure_mode_ == 2) {
    // REPLAY: relaunch the next recorded exec verbatim; discard the freshly
    // built nodes. The deterministic arena guarantees this token's buffer
    // addresses match the recorded exec's baked pointers.
    auto& chain = decode_pure_chain_[decode_pure_slot_];
    if (decode_pure_idx_ >= chain.size()) {
      decode_pure_mode_ = 0;   // topology drift (more commits than recorded)
      return false;            // fall back to the normal build path
    }
    auto& pe = chain[decode_pure_idx_++];
    graph_node_args_.clear();  // this token's packs are unused
    uint64_t mg = graph_current_gen();
    graph_advance_gen();
    CHECK_HIP_ERROR(hipGraphLaunch(pe.exec, stream_));
    free_graph_generation_async(mg);
    reset_build(/*recreate_graph=*/true);
    worker_->commit(stream_);
    return true;
  }

  // RECORD: nodes were added to build_graph_ via the normal add path. Add the
  // dependency edges, instantiate, launch, and KEEP the exec + its source graph
  // + kernarg packs alive in the chain. A fresh build_graph_ is made for the
  // next chunk (the recorded exec owns this one).
  if (!from_nodes_.empty()) {
    CHECK_HIP_ERROR(hipGraphAddDependencies(
        build_graph_, from_nodes_.data(), to_nodes_.data(),
        from_nodes_.size()));
  }
  hipGraphExec_t exec = nullptr;
  CHECK_HIP_ERROR(hipGraphInstantiate(&exec, build_graph_, nullptr, nullptr, 0));
  uint64_t mg = graph_current_gen();
  graph_advance_gen();
  CHECK_HIP_ERROR(hipGraphLaunch(exec, stream_));
  free_graph_generation_async(mg);
  decode_pure_chain_[decode_pure_slot_].push_back(
      PureExec{exec, build_graph_, std::move(graph_node_args_)});
  graph_node_args_.clear();
  CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));  // fresh for next chunk
  reset_build(/*recreate_graph=*/false);
  worker_->commit(stream_);
  return true;
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  temporary_ptrs_.clear();

  // Pure-relaunch decode (deterministic arena): record/replay the per-token
  // graph chain without any SetParams/ExecUpdate. Returns true when handled.
  if (commit_pure()) {
    return;
  }

  // Build-once decode replay: this token was buffered (no AddKernelNode). If a
  // DRAINED pooled exec for the stable decode topology exists, re-point its
  // persistent source graph's params (hipGraphKernelNodeSetParams — safe, the
  // slot is idle) and hipGraphExecUpdate it (the standard by-pointer replay), no
  // node-by-node rebuild. On a miss, materialize into build_graph_ and fall
  // through to the normal instantiate path.
  if (use_hip_graphs() && replay_active_ && node_count_ > 0) {
    const std::string& replay_key =
        graph_decode_mode() ? decode_key_ : prefill_key_;
    std::string full_key = graph_nodes_key_ + ":" + graph_deps_key_;
    ExecSlot* slot = nullptr;
    if (full_key == replay_key) {
      auto it = exec_pool_.find(replay_key);
      if (it != exec_pool_.end()) {
        for (auto& s : it->second) {
          if (s.inflight->load(std::memory_order_acquire) == 0 &&
              s.source_graph &&
              s.src_nodes.size() == build_node_params_.size()) {
            slot = &s;
            break;
          }
        }
        // Anti-hang (prefill): no drained slot means the clone+instantiate miss
        // path below would hipGraphInstantiate while execs are live on the stream
        // — that hangs on ROCm. Prefill chunks are coarse/sequential, so drain the
        // stream (every exec goes idle) and reuse ANY matching slot via SetParams.
        if (!slot && !graph_decode_mode()) {
          (void)hipStreamSynchronize(stream_);
          for (auto& s : it->second) {
            if (s.source_graph &&
                s.src_nodes.size() == build_node_params_.size()) {
              s.inflight->store(0, std::memory_order_release);  // drained by sync
              slot = &s;
              break;
            }
          }
        }
      }
    }
    bool ok = slot != nullptr;
    if (ok) {
      device_.make_current();
      // Re-point the exec's nodes DIRECTLY (hipGraphExecKernelNodeSetParams).
      // These slots are force-grown (instantiated, never ExecUpdate'd), so
      // src_nodes are the exec's own instantiation nodes -> valid. Avoids both the
      // node-by-node rebuild AND ExecUpdate-from-a-mutated-source (which returns
      // success but produces a faulting exec).
      for (size_t i = 0; i < build_node_params_.size(); ++i) {
        if (hipGraphExecKernelNodeSetParams(
                slot->exec, slot->src_nodes[i], &build_node_params_[i]) !=
            hipSuccess) {
          (void)hipGetLastError();
          ok = false;
          break;
        }
      }
    }
    if (ok) {
      // This token's arg Packs must outlive the launch (kernelParams by pointer).
      slot->packs = std::move(graph_node_args_);
      graph_node_args_.clear();
      auto inflight = slot->inflight;
      inflight->store(1, std::memory_order_release);
      add_completed_handler(
          [inflight]() { inflight->store(0, std::memory_order_release); });
      uint64_t my_gen = graph_current_gen();
      graph_advance_gen();
      CHECK_HIP_ERROR(hipGraphLaunch(slot->exec, stream_));
      free_graph_generation_async(my_gen);
      from_nodes_.clear();
      to_nodes_.clear();
      graph_nodes_key_.clear();
      graph_deps_key_.clear();
      node_map_.clear();
      active_deps_.clear();
      active_outputs_.clear();
      last_node_ = nullptr;
      bytes_in_graph_ = 0;
      build_nodes_.clear();
      build_node_params_.clear();
      replay_active_ = false;
      node_count_ = 0;
      worker_->commit(stream_);
      return;
    }
    // Miss (all slots in flight): grow the ping-pong pool by CLONING an existing
    // slot's known-good source graph (hipGraphClone of a graph that already
    // instantiated cleanly — avoids rebuilding nodes from buffered params), set
    // this token's params on the clone, instantiate + launch.
    replay_active_ = false;
    hipGraph_t base_graph = nullptr;
    std::vector<hipGraphNode_t> base_nodes;
    {
      auto pit = exec_pool_.find(replay_key);
      if (full_key == replay_key && pit != exec_pool_.end()) {
        for (auto& s : pit->second) {
          if (s.source_graph &&
              s.src_nodes.size() == build_node_params_.size()) {
            base_graph = s.source_graph;
            base_nodes = s.src_nodes;
            break;
          }
        }
      }
    }
    if (base_graph) {
      device_.make_current();
      hipGraph_t g2 = nullptr;
      CHECK_HIP_ERROR(hipGraphClone(&g2, base_graph));
      std::vector<hipGraphNode_t> n2(base_nodes.size());
      bool cloned = true;
      for (size_t i = 0; i < base_nodes.size(); ++i) {
        if (hipGraphNodeFindInClone(&n2[i], base_nodes[i], g2) != hipSuccess) {
          cloned = false;
          break;
        }
      }
      if (cloned) {
        for (size_t i = 0; i < n2.size(); ++i)
          CHECK_HIP_ERROR(
              hipGraphKernelNodeSetParams(n2[i], &build_node_params_[i]));
        hipGraphExec_t e2 = nullptr;
        CHECK_HIP_ERROR(hipGraphInstantiate(&e2, g2, nullptr, nullptr, 0));
        auto inflight = std::make_shared<std::atomic<int>>(1);
        exec_pool_[replay_key].push_back(
            {e2, inflight, g2, std::move(graph_node_args_), n2});
        graph_node_args_.clear();
        add_completed_handler(
            [inflight]() { inflight->store(0, std::memory_order_release); });
        uint64_t mg = graph_current_gen();
        graph_advance_gen();
        CHECK_HIP_ERROR(hipGraphLaunch(e2, stream_));
        free_graph_generation_async(mg);
        from_nodes_.clear();
        to_nodes_.clear();
        graph_nodes_key_.clear();
        graph_deps_key_.clear();
        node_map_.clear();
        active_deps_.clear();
        active_outputs_.clear();
        last_node_ = nullptr;
        bytes_in_graph_ = 0;
        build_nodes_.clear();
        build_node_params_.clear();
        node_count_ = 0;
        worker_->commit(stream_);
        return;
      }
      hipGraphDestroy(g2);
    }
    // Fallback (no cloneable base): rebuild nodes from buffered params + edges,
    // fall through to the normal instantiate path.
    std::vector<hipGraphNode_t> real(build_node_params_.size());
    for (size_t i = 0; i < build_node_params_.size(); ++i) {
      CHECK_HIP_ERROR(hipGraphAddKernelNode(
          &real[i], build_graph_, nullptr, 0, &build_node_params_[i]));
    }
    std::vector<hipGraphNode_t> nf, nt;
    nf.reserve(from_nodes_.size());
    nt.reserve(to_nodes_.size());
    for (size_t e = 0; e < from_nodes_.size(); ++e) {
      auto fi = reinterpret_cast<std::uintptr_t>(from_nodes_[e]) - 1;
      auto ti = reinterpret_cast<std::uintptr_t>(to_nodes_[e]) - 1;
      if (fi < real.size() && ti < real.size() && fi != ti) {
        nf.push_back(real[fi]);
        nt.push_back(real[ti]);
      }
    }
    from_nodes_ = std::move(nf);
    to_nodes_ = std::move(nt);
    build_nodes_ = std::move(real);
    // fall through to the normal block below (node_count_ unchanged from buffering)
  }

  if (use_hip_graphs() && node_count_ > 0) {
    if (!from_nodes_.empty()) {
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          build_graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          from_nodes_.size()));
    }
    device_.make_current();

    hipGraphExec_t graph_exec = nullptr;
    bool build_graph_adopted = false; // build_graph_ became a slot's source graph
    std::shared_ptr<std::atomic<int>> inflight;
    ExecSlot* used_slot = nullptr;
    // Reuse a drained exec (inflight==0) for an identical kernel sequence. In
    // decode-mode the whole-forward graph recurs every token, so refresh the
    // cached exec's params in one hipGraphExecUpdate; otherwise (or if ExecUpdate
    // fails) reinstantiate into the slot. The slot owns the source graph + arg
    // Packs for the exec's life (CLR stores kernelParams by pointer).
    static const bool prefill_replay_cu = [] {
      const char* e = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
      return e && e[0] == '1';
    }();
    const bool use_execupdate = graph_decode_mode() || prefill_replay_cu;
    auto& pool = exec_pool_[graph_nodes_key_ + ":" + graph_deps_key_];
    // For the stable decode topology, grow the pool to N execs (skip reuse until
    // then) so replay always finds a drained slot despite completion-flag lag.
    static const size_t replay_slots = [] {
      const char* e = std::getenv("MLX_GRAPH_REPLAY_SLOTS");
      return e ? std::max<size_t>(2, std::atoi(e)) : 4;
    }();
    const std::string& grow_key =
        graph_decode_mode() ? decode_key_ : prefill_key_;
    const bool force_grow = use_execupdate && !grow_key.empty() &&
        (graph_nodes_key_ + ":" + graph_deps_key_) == grow_key &&
        pool.size() < replay_slots;
    for (auto& slot : pool) {
      if (force_grow) break;
      if (slot.inflight->load(std::memory_order_acquire) != 0) {
        continue;
      }
      bool refreshed = false;
      if (use_execupdate) {
        hipGraphExecUpdateResult ur;
        hipGraphNode_t en;
        if (hipGraphExecUpdate(slot.exec, build_graph_, &en, &ur) == hipSuccess &&
            ur == hipGraphExecUpdateSuccess) {
          refreshed = true;
          build_graph_adopted = true; // exec now bound to build_graph_'s nodes
          slot.src_nodes = build_nodes_;
          if (slot.source_graph) hipGraphDestroy(slot.source_graph);
          slot.source_graph = build_graph_;
        } else {
          (void)hipGetLastError();
        }
      }
      if (refreshed) {
        graph_exec = slot.exec;
      } else {
        // Reinstantiate into this slot from the new build graph.
        hipGraphExecDestroy(slot.exec);
        CHECK_HIP_ERROR(
            hipGraphInstantiate(&slot.exec, build_graph_, nullptr, nullptr, 0));
        graph_exec = slot.exec;
        slot.src_nodes = build_nodes_;
        if (slot.source_graph) hipGraphDestroy(slot.source_graph);
        slot.source_graph = build_graph_;
        build_graph_adopted = true;
      }
      inflight = slot.inflight;
      used_slot = &slot;
      break;
    }
    if (graph_exec == nullptr) {
      CHECK_HIP_ERROR(
          hipGraphInstantiate(&graph_exec, build_graph_, nullptr, nullptr, 0));
      inflight = std::make_shared<std::atomic<int>>(0);
      pool.push_back({graph_exec, inflight, build_graph_, {}, build_nodes_});
      used_slot = &pool.back();
      build_graph_adopted = true;
    }
    inflight->store(1, std::memory_order_release);

    // Reclaim this chunk's deferred-free buffers once it has drained, bounding
    // graph-mode memory to a sliding window of chunks instead of a whole forward
    // (which OOMs a 32GB card). Free with a generation LAG so a buffer is only
    // released after the next few chunks have also launched — covers cross-chunk
    // / in-place references that a lag-0 free races (use-after-free). Tunable via
    // MLX_GRAPH_FREE_LAG.
    // Opt-in (MLX_GRAPH_FREE_LAG): per-chunk reclaim is currently racy (frees a
    // buffer the next chunk still references → UAF), so OFF by default — frees
    // flush safely at the per-token synchronize. The real fix is 100% buffer
    // reuse (deterministic per-forward addresses), not freeing.
    static const long free_lag = [] {
      const char* e = std::getenv("MLX_GRAPH_FREE_LAG");
      return e ? std::atol(e) : -1;
    }();
    uint64_t my_gen = graph_current_gen();
    graph_advance_gen();
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    if (free_lag >= 0 && static_cast<long>(my_gen) > free_lag) {
      uint64_t fg = my_gen - free_lag;
      add_completed_handler([fg]() { free_graph_generation(fg); });
    }
    // Reclaim this chunk's stream-ordered POOL buffers now, via hipFreeAsync
    // queued right after the launch on stream_ (retires after the graph; no
    // blocking drain). This is the common case on the discrete pool and keeps the
    // peak bounded without stalling the pipeline.
    free_graph_generation_async(my_gen);
    // Backstop ONLY for the non-stream-ordered remainder (unified/slab buffers,
    // rare on the discrete path): if that residual backlog still exceeds a cap,
    // drain + flush. MLX_GRAPH_DEFER_MAX_MB (default 2048; 0 disables).
    static const size_t defer_cap = [] {
      const char* e = std::getenv("MLX_GRAPH_DEFER_MAX_MB");
      return static_cast<size_t>(e ? std::atoll(e) : 2048) << 20;
    }();
    if (defer_cap && graph_deferred_bytes() > defer_cap) {
      (void)hipStreamSynchronize(stream_);
      flush_graph_deferred_frees();
    }
    // The completion handler fires after the stream drains this commit's launch;
    // clear inflight so the slot can be reused next token.
    add_completed_handler(
        [inflight]() { inflight->store(0, std::memory_order_release); });

    // Lock in the stable decode topology key once it recurs on two consecutive
    // tokens (the first decode token after prefill differs). Thereafter, matching
    // tokens replay (build-once) — re-point a drained slot's params, no rebuild.
    if (graph_decode_mode() && used_slot && decode_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_decode_key_) {
        decode_key_ = fk;
      } else {
        pending_decode_key_ = std::move(fk);
      }
    }
    // Same for prefill: lock the full-size-chunk topology once it recurs (chunk 2
    // matches chunk 1), then later chunks build-once/replay it.
    if (!graph_decode_mode() && used_slot && prefill_key_.empty()) {
      std::string fk = graph_nodes_key_ + ":" + graph_deps_key_;
      if (fk == pending_prefill_key_) {
        prefill_key_ = fk;
      } else {
        pending_prefill_key_ = std::move(fk);
      }
    }

    // Reset build state for the next chunk.
    from_nodes_.clear();
    to_nodes_.clear();
    graph_nodes_key_.clear();
    graph_deps_key_.clear();
    node_map_.clear();
    active_deps_.clear();
    active_outputs_.clear();
    last_node_ = nullptr;
    bytes_in_graph_ = 0;
    build_nodes_.clear();
    build_node_params_.clear();

    // The exec references the current build's Packs by pointer (CLR stores
    // kernelParams by pointer) and is relaunched in later tokens, so the Packs
    // must outlive the slot's next use. Move them into the slot, releasing its
    // prior Packs — the slot's prior launch has drained (only inflight==0
    // reused). source_graph was adopted above when ExecUpdate/instantiate bound
    // the exec to build_graph_; if not adopted, destroy it. Build next fresh.
    used_slot->packs = std::move(graph_node_args_);
    graph_node_args_.clear();
    if (!build_graph_adopted) {
      hipGraphDestroy(build_graph_);
    }
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
  }

  node_count_ = 0;

  // Put completion handlers in a batch.
  worker_->commit(stream_);
}

void CommandEncoder::synchronize() {
  (void)hipStreamSynchronize(stream_);
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
  (void)hipStreamSynchronize(stream_);
  // Stream is fully drained. Non-cached (no-reuse) execs reference these Packs
  // until now; cached-exec Packs live in their ExecSlot (clr#138) and are NOT
  // in these vectors, so clearing here is safe.
  graph_node_args_.clear();
  graph_node_args_prev_.clear();
  if (use_hip_graphs()) flush_graph_deferred_frees();
}

// Global flag: true while any stream on this process is recording a HIP graph.
// Lazy library inits (e.g. hipblasLtCreate) abort the process if first called
// during capture, so they consult this to defer to a non-capturing path.
std::atomic<bool> g_stream_capturing{false};
bool stream_capturing() {
  return g_stream_capturing.load(std::memory_order_relaxed);
}
void set_stream_capturing(bool v) {
  g_stream_capturing.store(v, std::memory_order_relaxed);
}

std::atomic<bool> g_graph_active{false};
bool graph_active() {
  return g_graph_active.load(std::memory_order_relaxed);
}
void set_graph_active(bool v) {
  g_graph_active.store(v, std::memory_order_relaxed);
}

// Decode-mode: a single-token forward accrues into ONE graph (no mid-forward
// commit) that is refreshed via hipGraphExecUpdate and launched once per token.
// Set by the generation loop for Lstep==1 steps; prefill leaves it off so its
// large intermediates stay bounded by the per-graph caps. Disable entirely with
// MLX_GRAPH_DECODE=0.
bool graph_decode_mode() {
  static const bool enabled = [] {
    const char* e = std::getenv("MLX_GRAPH_DECODE");
    return !(e && std::string(e) == "0");
  }();
  return enabled && g_graph_decode_mode.load(std::memory_order_relaxed);
}
void set_graph_decode_mode(bool v) {
  g_graph_decode_mode.store(v, std::memory_order_relaxed);
}


std::unordered_map<int, Device>& get_devices() {
  static std::unordered_map<int, Device> devices;
  return devices;
}

Device& device(mlx::core::Device device) {
  auto& devices = get_devices();
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    // Set blocking sync flags on THIS device (per index, not a single global
    // bool: if device 0 were touched first the global gate would leave device 1
    // unflagged). Must happen while this device is current and before its
    // context is created — i.e. before the Device is constructed. Iterating every
    // device would create a context/queue on the other GPU too; on a multi-GPU
    // host that cross-device coexistence is what wedges the discrete GPU's queue
    // over a TB5 link, so touch only this device.
    hipSetDevice(device.index);
    hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  // Bind the HIP current device to this stream's device. HIP's current device is
  // per-thread; everything that touches a stream goes through here (eval, kernel
  // launches, event record/wait, commit, completion callbacks). Without binding,
  // operations for a non-default GPU (--device 1) execute against device 0 — the
  // stream/event/kernel land on the wrong device and the queue hangs. With
  // HIP_VISIBLE_DEVICES the only device IS index 0 so the bug is hidden.
  auto& d = device(s.device);
  d.make_current();
  return d.get_command_encoder(s);
}

void clear_all_encoders() {
  auto& devices = get_devices();
  for (auto& [idx, dev] : devices) {
    dev.clear_encoders();
  }
}

} // namespace mlx::core::rocm

// --- Pure-relaunch decode bridge (called from the engine's decode loop) ------
namespace mlx::core {
void decode_pure_record(int slot) {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_pure_begin_record(slot);
}
void decode_pure_replay(int slot) {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_pure_begin_replay(slot);
}
void decode_pure_relaunch_all(int slot) {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_pure_relaunch_all(slot);
}
void decode_pure_off() {
  rocm::get_command_encoder(default_stream(default_device()))
      .decode_pure_end();
}
size_t decode_pure_chain_len(int slot) {
  return rocm::get_command_encoder(default_stream(default_device()))
      .decode_pure_chain_len(slot);
}
} // namespace mlx::core
