// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <atomic>
#include <dlfcn.h>
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

bool use_hip_graphs() {
  // Default ON: batching kernel launches into a HIP graph collapses thousands of
  // per-token launch submissions into a handful — the win is launch/PCIe traffic
  // and latency on a discrete GPU over a slow link (e.g. a TB5 eGPU), where each
  // hipLaunchKernel is a separate command crossing the link. Explicit opt-out via
  // MLX_USE_HIP_GRAPHS=0; any other value (or unset) leaves graphs enabled.
  static bool use_graphs = [] {
    const char* e = std::getenv("MLX_USE_HIP_GRAPHS");
    if (e && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0))
      return false;
    return true;
  }();
  return use_graphs;
}

// Reuse-path diagnostics (MLX_GRAPH_REUSE_STATS).
std::atomic<long> g_reuse_update_{0};
std::atomic<long> g_reuse_reinst_{0};
std::atomic<long> g_reuse_new_{0};
// Launch-count diagnostics: graph launches (hipGraphLaunch, one per commit) vs
// inline launches (library GEMM / JIT / memset routed through launch_kernel,
// which graph-splits). MLX_GRAPH_REUSE_STATS prints these too.
std::atomic<long> g_graph_launches_{0};
std::atomic<long> g_inline_launches_{0};
std::atomic<long> g_kernel_nodes_{0};
// Per-phase CPU time accumulators (microseconds), MLX_GRAPH_REUSE_STATS.
std::atomic<long> g_t_addnode_us_{0};   // hipGraphAddKernelNode (construction)
std::atomic<long> g_t_deps_us_{0};      // dependency wiring
std::atomic<long> g_t_reuse_us_{0};     // instantiate / execupdate / setparams
std::atomic<long> g_t_launch_us_{0};    // hipGraphLaunch
std::atomic<long> g_key_collisions_{0}; // matched slot != current build sig
std::atomic<long> g_sp_sig_mismatch_{0}; // SetParams per-node sig mismatch (A1)
// Change tracking: across reused slots, how many node arg0 values changed vs
// total node-reuses, and the set of distinct kernel funcs that ever changed.
std::atomic<long> g_arg_total_{0};
std::atomic<long> g_arg_changed_{0};
std::atomic<long> g_arg_untracked_{0};
std::mutex g_changed_funcs_mtx_;
std::unordered_map<uint64_t, long> g_changed_funcs_;
static inline long now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static thread_local const char* g_cur_prim_ = "?";
static std::mutex g_inline_by_op_mtx_;
static std::unordered_map<std::string, long> g_inline_by_op_;
static const bool g_reuse_stats_on_ =
    std::getenv("MLX_GRAPH_REUSE_STATS") != nullptr;
bool graph_reuse_stats_on() {
  return g_reuse_stats_on_;
}
void set_current_prim(const char* name) {
  g_cur_prim_ = name ? name : "?";
}
void record_inline_launch() {
  g_inline_launches_.fetch_add(1, std::memory_order_relaxed);
  if (g_reuse_stats_on_) {
    std::lock_guard<std::mutex> lk(g_inline_by_op_mtx_);
    g_inline_by_op_[g_cur_prim_]++;
  }
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

void CommandEncoder::add_kernel_node_kp(const hipKernelNodeParams& kp) {
  hipGraphNode_t node;
  long _t0 = g_reuse_stats_on_ ? now_us() : 0;
  CHECK_HIP_ERROR(hipGraphAddKernelNode(&node, build_graph_, nullptr, 0, &kp));
  if (g_reuse_stats_on_) g_t_addnode_us_ += now_us() - _t0;
  g_kernel_nodes_.fetch_add(1, std::memory_order_relaxed);
  build_nodes_.push_back(node);
  build_node_params_.push_back(kp);
  build_arghash_.push_back(pending_arghash_);
  // Key the node by its kernel FUNCTION + FULL launch dims. Reuse (SetParams /
  // reinstantiate) only re-points params of a structurally IDENTICAL exec, so the
  // key must distinguish any difference that changes the launch. Using the grid/
  // block PRODUCT collided distinct shapes (e.g. 2097152x1x1 vs 1024x2048x1) onto
  // one key; reinstantiate rebuilds and survives that, but SetParams then writes
  // params onto a mismatched exec node -> the kernel launches the old grid over
  // new (smaller) buffers -> out-of-bounds. Encode each dim separately.
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
  insert_graph_dependencies(GraphNode{node, key});
}

void CommandEncoder::add_module_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params,
    std::shared_ptr<void> args_keepalive,
    uint64_t arghash) {
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
  pending_arghash_ = arghash; // from KernelArgs::arg_hash() (0 if not provided)
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
          pending_arghash_ = 0; // child kernarg layout unknown to the tracker
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

void CommandEncoder::commit() {
  // During graph capture, record ONLY the compute kernels into the graph. The
  // host-function completion callbacks (which release temporaries) are not
  // executed under stream capture and would otherwise be baked into the graph
  // as host nodes that fire on every replay. Temporaries are arena-backed while
  // capturing (the arena is freed in bulk on end, never per-buffer), so we can
  // simply drop our references without scheduling a cleanup task. NOTE: this
  // relies on the DecodeArena being active during capture — otherwise dropping
  // these refs would return live buffers to the pool while later recorded
  // kernels still reference them.
  if (capturing_) {
    // Keep capture-time buffers alive (unique, stable addresses) until the
    // graph is destroyed — do NOT free them (which would alias graph nodes) and
    // do NOT schedule a host-function completion (it can't fire under capture).
    for (auto& d : temporaries_)
      capture_held_.push_back(std::move(d));
    temporaries_.clear();
    temporary_ptrs_.clear();
    node_count_ = 0;
    return;
  }

  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  temporary_ptrs_.clear();

  if (use_hip_graphs() && node_count_ > 0) {
    long _td0 = g_reuse_stats_on_ ? now_us() : 0;
    if (!from_nodes_.empty()) {
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          build_graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          from_nodes_.size()));
    }
    if (g_reuse_stats_on_) g_t_deps_us_ += now_us() - _td0;

    device_.make_current();

    long _tr0 = g_reuse_stats_on_ ? now_us() : 0;
    static const bool no_reuse = std::getenv("MLX_GRAPH_NO_REUSE") != nullptr;
    hipGraphExec_t graph_exec = nullptr;
    bool cached_exec = false;
    bool build_graph_adopted = false; // build_graph_ became a slot's source graph
    std::shared_ptr<std::atomic<int>> inflight;
    ExecSlot* used_slot = nullptr;
    if (!no_reuse) {
      // Reuse a drained exec (inflight==0) for an identical kernel sequence. We
      // DON'T use hipGraphExecUpdate — it returns success but mis-maps params in
      // the model's complex DAG. Instead we refresh each node's params by handle
      // (hipGraphExecKernelNodeSetParams), keyed by our own insertion order, which
      // is deterministic. The slot keeps the source graph + ordered node handles +
      // arg Packs alive for the exec's life. MLX_GRAPH_REINST_SLOT forces the
      // reinstantiate fallback (diagnostic).
      // SetParams reuse is opt-in (MLX_GRAPH_SETPARAMS): it currently corrupts
      // the kernarg of some complex by-value-struct kernels (e.g. copy_gg_byval)
      // — hipGraphExecKernelNodeSetParams returns success but nulls the arg
      // segment. Default to reinstantiate-into-slot, which is correct.
      static const bool use_setparams =
          std::getenv("MLX_GRAPH_SETPARAMS") != nullptr;
      // Pure relaunch: reuse the cached exec with NO param update — correct only
      // if per-token buffer addresses are stable. Diagnostic for the minimal-call
      // path.
      static const bool use_relaunch =
          std::getenv("MLX_GRAPH_RELAUNCH") != nullptr;
      // ExecUpdate: refresh a cached exec's params in one call (vs reinstantiate).
      // Forced on in decode-mode (the whole-forward graph is reused every token);
      // also opt-in globally via MLX_GRAPH_EXECUPDATE.
      const bool use_execupdate =
          std::getenv("MLX_GRAPH_EXECUPDATE") != nullptr || graph_decode_mode();
      // Bisect: only ExecUpdate commits whose node-count is within [min,max];
      // others reinstantiate. Sweep to isolate which commit class EU breaks on.
      static const size_t eu_min = [] {
        const char* e = std::getenv("MLX_GRAPH_EU_MIN");
        return e ? (size_t)std::atoll(e) : 0;
      }();
      static const size_t eu_max = [] {
        const char* e = std::getenv("MLX_GRAPH_EU_MAX");
        return e ? (size_t)std::atoll(e) : (size_t)-1;
      }();
      const bool eu_size_ok =
          build_node_params_.size() >= eu_min &&
          build_node_params_.size() <= eu_max;
      static const bool stats = std::getenv("MLX_GRAPH_REUSE_STATS") != nullptr;
      std::vector<std::array<uint64_t, 7>> cur_sig;
      cur_sig.reserve(build_node_params_.size());
      for (auto& p : build_node_params_) cur_sig.push_back(node_sig(p));
      // Per-node arg-hash captured safely at build time. Used to measure how
      // many nodes' kernargs change per token.
      std::vector<uint64_t>& cur_args = build_arghash_;
      auto& pool = exec_pool_[graph_nodes_key_ + ":" + graph_deps_key_];
      for (auto& slot : pool) {
        if (slot.inflight->load(std::memory_order_acquire) != 0) {
          continue;
        }
        static const bool sp_debug =
            std::getenv("MLX_GRAPH_SP_DEBUG") != nullptr;
        // Change measurement: how many nodes' primary buffer changed since this
        // slot's last use? Drives "refresh only changed nodes" (and tells us
        // whether the by-value-struct copy kernels are among the changers).
        if (stats && slot.last_args.size() == cur_args.size()) {
          long changed = 0, untracked = 0;
          for (size_t i = 0; i < cur_args.size(); i++) {
            if (cur_args[i] == 0) untracked++; // module/JIT node, hash not taken
            if (slot.last_args[i] != cur_args[i]) {
              changed++;
              std::lock_guard<std::mutex> lk(g_changed_funcs_mtx_);
              g_changed_funcs_[reinterpret_cast<uint64_t>(
                  build_node_params_[i].func)]++;
            }
          }
          g_arg_total_ += static_cast<long>(cur_args.size());
          g_arg_changed_ += changed;
          g_arg_untracked_ += untracked;
        }
        slot.last_args = cur_args;
        // Collision check: does the matched slot's signature equal this build's?
        if (stats) {
          bool same = slot.src_sig.size() == cur_sig.size();
          for (size_t i = 0; same && i < cur_sig.size(); i++)
            same = slot.src_sig[i] == cur_sig[i];
          if (!same) g_key_collisions_++;
        }
        bool refreshed = false;
        if (use_execupdate && eu_size_ok) {
          hipGraphExecUpdateResult ur;
          hipGraphNode_t en;
          if (hipGraphExecUpdate(slot.exec, build_graph_, &en, &ur) ==
                  hipSuccess &&
              ur == hipGraphExecUpdateSuccess) {
            graph_exec = slot.exec;
            refreshed = true;
            if (stats) g_reuse_update_++;
            inflight = slot.inflight;
            used_slot = &slot;
            build_graph_adopted = true; // exec now bound to build_graph_'s nodes
            slot.src_nodes = build_nodes_;
            slot.src_sig = cur_sig;
            if (slot.source_graph) hipGraphDestroy(slot.source_graph);
            slot.source_graph = build_graph_;
            break;
          }
          (void)hipGetLastError();
        }
        if (use_relaunch &&
            slot.src_nodes.size() == build_node_params_.size()) {
          graph_exec = slot.exec; // relaunch as-is
          refreshed = true;
          if (stats) g_reuse_update_++;
          inflight = slot.inflight;
          used_slot = &slot;
          // packs of THIS build must stay alive in case the cached exec aliases
          // any current address; handled by the common bottom transfer.
          break;
        }
        if (use_setparams &&
            slot.src_nodes.size() == build_node_params_.size()) {
          // Step 4 (plan A1): src_nodes[i] (captured at instantiate) must be the
          // SAME kernel as build_node_params_[i] (this commit) at every index, or
          // SetParams writes one kernel's params onto another's exec node. Size
          // match is necessary but not sufficient — assert per-node signature.
          bool sig_ok = slot.src_sig.size() == cur_sig.size();
          for (size_t i = 0; sig_ok && i < cur_sig.size(); i++)
            sig_ok = (slot.src_sig[i] == cur_sig[i]);
          if (!sig_ok) {
            g_sp_sig_mismatch_++;
            if (sp_debug)
              fprintf(stderr, "[sp] sig-mismatch -> reinstantiate fallback\n");
            // leave refreshed=false -> reinstantiate fallback below
          } else {
          refreshed = true;
          if (sp_debug) (void)hipGetLastError(); // clear stale error
          for (size_t i = 0; i < slot.src_nodes.size(); i++) {
            hipError_t e = hipGraphExecKernelNodeSetParams(
                slot.exec, slot.src_nodes[i], &build_node_params_[i]);
            hipError_t pe = sp_debug ? hipGetLastError() : hipSuccess;
            if (e != hipSuccess || pe != hipSuccess) {
              auto& kp = build_node_params_[i];
              fprintf(stderr,
                      "[sp] node %zu/%zu ret=%d pend=%d(%s) func=%p "
                      "grid=%ux%ux%u block=%ux%ux%u smem=%u\n",
                      i, slot.src_nodes.size(), (int)e, (int)pe,
                      hipGetErrorString(pe == hipSuccess ? e : pe), kp.func,
                      kp.gridDim.x, kp.gridDim.y, kp.gridDim.z, kp.blockDim.x,
                      kp.blockDim.y, kp.blockDim.z, kp.sharedMemBytes);
              if (e != hipSuccess) {
                refreshed = false;
                break;
              }
            }
          }
          }
        }
        if (refreshed) {
          // exec still instantiated from slot.source_graph/src_nodes (unchanged);
          // only its params were refreshed to point at the current build's Packs.
          graph_exec = slot.exec;
          if (stats) g_reuse_update_++;
        } else {
          // Fallback: reinstantiate into this slot from the new build graph.
          hipGraphExecDestroy(slot.exec);
          CHECK_HIP_ERROR(hipGraphInstantiate(
              &slot.exec, build_graph_, nullptr, nullptr, 0));
          graph_exec = slot.exec;
          slot.src_nodes = build_nodes_;
          slot.src_sig = cur_sig;
          if (slot.source_graph) {
            hipGraphDestroy(slot.source_graph);
          }
          slot.source_graph = build_graph_;
          build_graph_adopted = true;
          if (stats) g_reuse_reinst_++;
        }
        inflight = slot.inflight;
        used_slot = &slot;
        break;
      }
      if (graph_exec == nullptr) {
        CHECK_HIP_ERROR(
            hipGraphInstantiate(&graph_exec, build_graph_, nullptr, nullptr, 0));
        inflight = std::make_shared<std::atomic<int>>(0);
        pool.push_back(
            {graph_exec, inflight, build_graph_, {}, build_nodes_, cur_sig,
             cur_args});
        used_slot = &pool.back();
        build_graph_adopted = true;
        if (stats) g_reuse_new_++;
      }
      inflight->store(1, std::memory_order_release);
      cached_exec = true;
    } else {
      CHECK_HIP_ERROR(
          hipGraphInstantiate(&graph_exec, build_graph_, nullptr, nullptr, 0));
    }

    static const bool dump = std::getenv("MLX_HIP_GRAPH_DUMP") != nullptr;
    if (dump) {
      size_t n = 0;
      hipGraphGetNodes(build_graph_, nullptr, &n);
      std::vector<hipGraphNode_t> nodes(n);
      hipGraphGetNodes(build_graph_, nodes.data(), &n);
      size_t nedges = 0;
      hipGraphGetEdges(build_graph_, nullptr, nullptr, &nedges);
      int k = 0, mcpy = 0, mset = 0, host = 0, child = 0, empty = 0, malloc_n = 0,
          free_n = 0, other = 0;
      for (auto nd : nodes) {
        hipGraphNodeType t;
        if (hipGraphNodeGetType(nd, &t) != hipSuccess) { other++; continue; }
        switch (t) {
          case hipGraphNodeTypeKernel: k++; break;
          case hipGraphNodeTypeMemcpy: mcpy++; break;
          case hipGraphNodeTypeMemset: mset++; break;
          case hipGraphNodeTypeHost: host++; break;
          case hipGraphNodeTypeGraph: child++; break;
          case hipGraphNodeTypeEmpty: empty++; break;
          case hipGraphNodeTypeMemAlloc: malloc_n++; break;
          case hipGraphNodeTypeMemFree: free_n++; break;
          default: other++; break;
        }
      }
      fprintf(stderr,
              "[graph] nodes=%zu edges=%zu kernel=%d memcpy=%d memset=%d "
              "host=%d child=%d empty=%d memAlloc=%d memFree=%d other=%d\n",
              n, nedges, k, mcpy, mset, host, child, empty, malloc_n, free_n,
              other);
      static int dn = 0;
      char path[64];
      snprintf(path, sizeof(path), "/tmp/hipgraph_%d.dot", dn++);
      hipGraphDebugDotPrint(build_graph_, path, 0);
      fprintf(stderr, "[graph] dot -> %s\n", path);
    }

    if (g_reuse_stats_on_) g_t_reuse_us_ += now_us() - _tr0;
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
    long _tl0 = g_reuse_stats_on_ ? now_us() : 0;
    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));
    if (g_reuse_stats_on_) g_t_launch_us_ += now_us() - _tl0;
    g_graph_launches_.fetch_add(1, std::memory_order_relaxed);
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
    // The completion handler fires after the stream drains this commit's launch.
    // Pooled execs: clear inflight so the slot can be ExecUpdate-reused.
    // Non-pooled (no_reuse): destroy the exec to avoid leaking it.
    if (cached_exec) {
      add_completed_handler(
          [inflight]() { inflight->store(0, std::memory_order_release); });
    } else {
      add_completed_handler([graph_exec]() { hipGraphExecDestroy(graph_exec); });
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
    build_arghash_.clear();

    if (cached_exec) {
      // The exec references the current build's Packs by pointer (via SetParams
      // or instantiate) and is relaunched in later tokens, so the Packs must
      // outlive the slot's next use. Move them into the slot, releasing its prior
      // Packs — the slot's prior launch has drained (only inflight==0 reused), so
      // nothing reads them. The slot's source_graph/src_nodes were set above when
      // adopted; if the new build graph was NOT adopted (SetParams refresh kept
      // the original source graph), destroy it. Always build the next chunk fresh.
      used_slot->packs = std::move(graph_node_args_);
      graph_node_args_.clear();
      if (!build_graph_adopted) {
        hipGraphDestroy(build_graph_);
      }
      CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    } else {
      // no_reuse: the exec is destroyed once its launch drains; its Packs stay
      // in graph_node_args_ (freed in synchronize, after the stream is idle).
      // Instantiate copied the node structure, so destroying build_graph_ now is
      // safe.
      hipGraphDestroy(build_graph_);
      CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    }
  }

  node_count_ = 0;

  // Put completion handlers in a batch.
  worker_->commit(stream_);
}

void CommandEncoder::synchronize() {
  // A capturing stream cannot be synchronized, and there is nothing to wait for
  // — recorded kernels do not execute until the captured graph is replayed.
  if (capturing_) {
    return;
  }
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
  static const bool stats = std::getenv("MLX_GRAPH_REUSE_STATS") != nullptr;
  if (stats) {
    fprintf(stderr,
            "[reuse-stats] update=%ld reinst=%ld new=%ld | "
            "graph_launches=%ld inline_launches=%ld kernel_nodes=%ld\n",
            g_reuse_update_.load(), g_reuse_reinst_.load(),
            g_reuse_new_.load(), g_graph_launches_.load(),
            g_inline_launches_.load(), g_kernel_nodes_.load());
    fprintf(stderr, "[key-collisions] %ld\n", g_key_collisions_.load());
    fprintf(stderr, "[sp-sig-mismatch] %ld\n", g_sp_sig_mismatch_.load());
    {
      std::lock_guard<std::mutex> lk(g_changed_funcs_mtx_);
      long tot = g_arg_total_.load(), chg = g_arg_changed_.load();
      long untr = g_arg_untracked_.load();
      fprintf(stderr,
              "[arg-change] changed=%ld/%ld nodes (%.1f%%) untracked(JIT)=%ld "
              "across %zu distinct changing funcs\n",
              chg, tot, tot ? 100.0 * chg / tot : 0.0, untr,
              g_changed_funcs_.size());
      std::vector<std::pair<uint64_t, long>> v(g_changed_funcs_.begin(),
                                               g_changed_funcs_.end());
      std::sort(v.begin(), v.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
      for (size_t i = 0; i < v.size() && i < 16; i++) {
        Dl_info di;
        const char* nm = (dladdr(reinterpret_cast<void*>(v[i].first), &di) &&
                          di.dli_sname)
            ? di.dli_sname
            : "?";
        fprintf(stderr, "  [chg-func] %p x%ld %s\n",
                reinterpret_cast<void*>(v[i].first), v[i].second, nm);
      }
    }
    std::lock_guard<std::mutex> lk(g_inline_by_op_mtx_);
    std::vector<std::pair<std::string, long>> v(
        g_inline_by_op_.begin(), g_inline_by_op_.end());
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    std::string line = "[inline-by-op]";
    for (size_t i = 0; i < v.size() && i < 12; i++)
      line += " " + v[i].first + "=" + std::to_string(v[i].second);
    fprintf(stderr, "%s\n", line.c_str());
    fprintf(stderr,
            "[graph-time-ms] addnode=%ld deps=%ld reuse/instantiate=%ld "
            "launch=%ld\n",
            g_t_addnode_us_.load() / 1000, g_t_deps_us_.load() / 1000,
            g_t_reuse_us_.load() / 1000, g_t_launch_us_.load() / 1000);
  }
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
static std::atomic<bool> g_graph_decode_mode{false};
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

void CommandEncoder::begin_capture() {
  if (capturing_)
    return;
  g_stream_capturing.store(true, std::memory_order_relaxed);
  g_graph_active.store(true, std::memory_order_relaxed);
  device_.make_current();
  // hipStreamBeginCapture records all subsequent operations on this stream
  // into a graph instead of executing them. Use ThreadLocal (not Global) mode
  // so only THIS thread's stream activity is captured — the Worker thread may
  // still be running completion/free callbacks from prior eager steps, and
  // capturing those cross-thread ops bakes spurious nodes into the graph that
  // hang on replay.
  hipError_t err =
      hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal);
  if (err == hipSuccess) {
    capturing_ = true;
  }
}

bool CommandEncoder::end_capture() {
  if (!capturing_)
    return false;
  capturing_ = false;
  g_stream_capturing.store(false, std::memory_order_relaxed);

  hipGraph_t new_graph = nullptr;
  hipError_t err = hipStreamEndCapture(stream_, &new_graph);
  if (err != hipSuccess || new_graph == nullptr) {
    return false;
  }

  // Destroy previous graph if any
  reset_graph();

  graph_ = new_graph;

  // Patch host->device constant-upload memcpy nodes. Stream capture records
  // these with the HOST source pointer, but those host buffers are freed before
  // replay, so on replay the H2D copy reads stale host memory and stalls the
  // GPU queue. While the host data is still valid (right after capture), copy
  // each into a persistent device staging buffer and rewrite the node as
  // device->device so replay reads valid device memory. The staging buffers are
  // intentionally leaked for the lifetime of the graph.
  {
    size_t n = 0;
    hipGraphGetNodes(graph_, nullptr, &n);
    std::vector<hipGraphNode_t> nodes(n);
    hipGraphGetNodes(graph_, nodes.data(), &n);
    for (size_t i = 0; i < n; i++) {
      hipGraphNodeType t;
      if (hipGraphNodeGetType(nodes[i], &t) != hipSuccess ||
          t != hipGraphNodeTypeMemcpy)
        continue;
      hipMemcpy3DParms p{};
      if (hipGraphMemcpyNodeGetParams(nodes[i], &p) != hipSuccess)
        continue;
      if (p.kind != hipMemcpyHostToDevice)
        continue;
      size_t bytes = p.extent.width * std::max<size_t>(p.extent.height, 1) *
          std::max<size_t>(p.extent.depth, 1);
      if (bytes == 0 || p.srcPtr.ptr == nullptr)
        continue;
      void* stage = nullptr;
      if (hipMalloc(&stage, bytes) != hipSuccess)
        continue;
      // Copy the host constant into the staging buffer now (host source is still
      // valid right after capture) and rewrite the node as device->device.
      if (hipMemcpy(stage, p.srcPtr.ptr, bytes, hipMemcpyHostToDevice) !=
          hipSuccess) {
        hipFree(stage);
        continue;
      }
      p.srcPtr = make_hipPitchedPtr(stage, p.srcPtr.pitch ? p.srcPtr.pitch : bytes,
                                    p.extent.width, std::max<size_t>(p.extent.height, 1));
      p.kind = hipMemcpyDeviceToDevice;
      (void)hipGraphMemcpyNodeSetParams(nodes[i], &p);
    }
  }

  static const bool dbg = std::getenv("MLX_GRAPH_DEBUG") != nullptr;
  if (dbg) {
    size_t n = 0;
    hipGraphGetNodes(graph_, nullptr, &n);
    std::vector<hipGraphNode_t> nodes(n);
    hipGraphGetNodes(graph_, nodes.data(), &n);
    int kKernel = 0, kMemcpy = 0, kMemset = 0, kHost = 0, kEmpty = 0,
        kWaitEvent = 0, kEventRecord = 0, kMemAlloc = 0, kMemFree = 0, kOther = 0;
    for (size_t i = 0; i < n; i++) {
      hipGraphNodeType t;
      if (hipGraphNodeGetType(nodes[i], &t) != hipSuccess) { kOther++; continue; }
      switch (t) {
        case hipGraphNodeTypeKernel: kKernel++; break;
        case hipGraphNodeTypeMemcpy: kMemcpy++; break;
        case hipGraphNodeTypeMemset: kMemset++; break;
        case hipGraphNodeTypeHost: kHost++; break;
        case hipGraphNodeTypeEmpty: kEmpty++; break;
        case hipGraphNodeTypeWaitEvent: kWaitEvent++; break;
        case hipGraphNodeTypeEventRecord: kEventRecord++; break;
        case hipGraphNodeTypeMemAlloc: kMemAlloc++; break;
        case hipGraphNodeTypeMemFree: kMemFree++; break;
        default: kOther++; break;
      }
    }
    fprintf(stderr,
            "[capture] nodes=%zu kernel=%d memcpy=%d memset=%d host=%d empty=%d "
            "waitEvent=%d eventRecord=%d memAlloc=%d memFree=%d other=%d\n",
            n, kKernel, kMemcpy, kMemset, kHost, kEmpty, kWaitEvent,
            kEventRecord, kMemAlloc, kMemFree, kOther);
    // Inspect memcpy nodes — host->device copies with a stale host source would
    // fault/stall on replay.
    for (size_t i = 0; i < n; i++) {
      hipGraphNodeType t;
      if (hipGraphNodeGetType(nodes[i], &t) != hipSuccess ||
          t != hipGraphNodeTypeMemcpy)
        continue;
      hipMemcpy3DParms p{};
      if (hipGraphMemcpyNodeGetParams(nodes[i], &p) == hipSuccess) {
        fprintf(stderr, "[capture]   memcpy kind=%d bytes=%zu\n", (int)p.kind,
                p.extent.width * p.extent.height * p.extent.depth);
      }
    }
  }

  err = hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
  if (err != hipSuccess) {
    hipGraphDestroy(graph_);
    graph_ = nullptr;
    graph_exec_ = nullptr;
    return false;
  }
  return true;
}

bool CommandEncoder::replay(bool sync) {
  if (!graph_exec_)
    return false;
  device_.make_current();
  static const bool dbg = std::getenv("MLX_GRAPH_DEBUG") != nullptr;
  if (dbg) fprintf(stderr, "[replay] launching graph (sync=%d)...\n", (int)sync);
  hipError_t err = hipGraphLaunch(graph_exec_, stream_);
  if (dbg) fprintf(stderr, "[replay] launch returned %d (%s)\n",
                   (int)err, hipGetErrorString(err));
  if (err != hipSuccess)
    return false;
  // The captured kernels run asynchronously on stream_. The completion Events
  // that eval() would normally wait on were skipped during capture. When sync
  // is requested, wait here for the replayed work to finish before the caller
  // reads outputs. When async, the caller orders its output reads after this
  // launch on the SAME stream (subsequent MLX eval on the generation stream),
  // so no drain is needed and per-token work can pipeline.
  if (!sync)
    return true;
  err = hipStreamSynchronize(stream_);
  if (dbg) fprintf(stderr, "[replay] sync returned %d (%s)\n",
                   (int)err, hipGetErrorString(err));
  return err == hipSuccess;
}

void CommandEncoder::reset_graph() {
  if (graph_exec_) {
    hipGraphExecDestroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_) {
    hipGraphDestroy(graph_);
    graph_ = nullptr;
  }
  // The captured graph is gone — release the buffers it referenced.
  capture_held_.clear();
  g_graph_active.store(false, std::memory_order_relaxed);
  flush_graph_deferred_frees();
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
