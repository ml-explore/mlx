// Copyright © 2025 Apple Inc.

#include <atomic>
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/utils.h"

#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
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
  static bool use_graphs = std::getenv("MLX_USE_HIP_GRAPHS") != nullptr;
  return use_graphs;
}

// Per-arch op/MB caps for the build graph. Tunable via env.
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
  }
}

CommandEncoder::~CommandEncoder() {
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
  for (auto& node : nodes) {
    graph_nodes_key_ += node.node_type;
    graph_nodes_key_ += "-";
  }
  std::vector<GraphNode> deps;
  {
    std::unordered_set<hipGraphNode_t> set_deps;
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
      graph_deps_key_ += from.id;
      graph_deps_key_ += "-";
      graph_deps_key_ += to.id;
      graph_deps_key_ += "-";
    }
  }
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
  hipGraphNode_t node;
  CHECK_HIP_ERROR(
      hipGraphAddKernelNode(&node, build_graph_, nullptr, 0, &kernel_params));
  insert_graph_dependencies(GraphNode{node, "K"});
}

void CommandEncoder::add_child_graph_node(
    hipGraph_t child,
    const std::string& key) {
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
    if (!from_nodes_.empty()) {
      CHECK_HIP_ERROR(hipGraphAddDependencies(
          build_graph_,
          from_nodes_.data(),
          to_nodes_.data(),
          from_nodes_.size()));
    }

    device_.make_current();

    auto graph_key =
        std::hash<std::string>{}(graph_nodes_key_ + ":" + graph_deps_key_);
    auto cached = graph_cache_.get(graph_key);
    hipGraphExec_t graph_exec = cached ? *cached : nullptr;

    if (graph_exec != nullptr) {
      hipGraphExecUpdateResult update_result;
      hipGraphNode_t error_node;
      hipError_t uerr = hipGraphExecUpdate(
          graph_exec, build_graph_, &error_node, &update_result);
      if (uerr != hipSuccess ||
          update_result != hipGraphExecUpdateSuccess) {
        (void)hipGetLastError();
        hipGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
      }
    }
    if (graph_exec == nullptr) {
      CHECK_HIP_ERROR(hipGraphInstantiate(
          &graph_exec, build_graph_, nullptr, nullptr, 0));
      graph_cache_.put(graph_key, graph_exec);
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

    CHECK_HIP_ERROR(hipGraphLaunch(graph_exec, stream_));

    // Reset build state for the next chunk.
    from_nodes_.clear();
    to_nodes_.clear();
    graph_nodes_key_.clear();
    graph_deps_key_.clear();
    node_map_.clear();
    active_deps_.clear();
    active_outputs_.clear();
    bytes_in_graph_ = 0;
    hipGraphDestroy(build_graph_);
    CHECK_HIP_ERROR(hipGraphCreate(&build_graph_, 0));
    // NOTE: do NOT free graph_node_args_ here. hipGraphLaunch is async and the
    // exec references the kernelParams until the stream drains. They are freed
    // in synchronize() once the stream is idle.
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
  // Stream is fully drained; graph execs no longer reference the kernelParams.
  graph_node_args_.clear();
  graph_node_args_prev_.clear();
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
