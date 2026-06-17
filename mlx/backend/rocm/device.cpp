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

} // namespace

Device::Device(int device) : device_(device) {
  make_current();
  {
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, device_) == hipSuccess) {
      fprintf(stderr, "[mlx-rocm] bound HIP device %d: %s (%s)\n",
              device_, p.gcnArchName, p.name);
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
    : device_(d), stream_(d), worker_(std::make_unique<Worker>()) {}

CommandEncoder::~CommandEncoder() = default;

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

void CommandEncoder::set_input_array(const array& arr) {}

void CommandEncoder::set_output_array(const array& arr) {}

void CommandEncoder::maybe_commit() {
  if (node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer)) {
    commit();
  }
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
}

// Global flag: true while any stream on this process is recording a HIP graph.
// Lazy library inits (e.g. hipblasLtCreate) abort the process if first called
// during capture, so they consult this to defer to a non-capturing path.
std::atomic<bool> g_stream_capturing{false};
bool stream_capturing() {
  return g_stream_capturing.load(std::memory_order_relaxed);
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
  static bool flags_set = false;
  if (!flags_set) {
    flags_set = true;
    // Set blocking sync for all devices to reduce CPU usage
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; i++) {
      hipSetDevice(i);
      hipSetDeviceFlags(hipDeviceScheduleBlockingSync);
    }
    // Restore default device
    hipSetDevice(0);
  }
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device(s.device).get_command_encoder(s);
}

void clear_all_encoders() {
  auto& devices = get_devices();
  for (auto& [idx, dev] : devices) {
    dev.clear_encoders();
  }
}

} // namespace mlx::core::rocm
